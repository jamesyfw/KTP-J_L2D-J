from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from timm.models.layers import DropPath
from functools import partial

from common.model_ktpformer import TPA_Block, Block

# ==========================================
# 3DGCN Components (from HRNet/3DGCN)
# ==========================================
class SemGraphConv(nn.Module):
    """Semantic graph convolution layer"""
    def __init__(self, in_features, out_features, adj, bias=True):
        super(SemGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj = adj
        self.m = (self.adj > 0)
        self.e = nn.Parameter(torch.zeros(1, len(self.m.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e.data, 1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        adj = -9e15 * torch.ones_like(self.adj).to(input.device)
        adj[self.m] = self.e
        adj = F.softmax(adj, dim=1)

        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()
        self.gconv = SemGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))
        x = self.relu(x)
        return x

class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()
        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out

class SemGCN_Extractor(nn.Module):
    """ Modified SemGCN that outputs hidden features instead of 3D coordinates """
    def __init__(self, adj, hid_dim, in_chans=2, num_layers=4, p_dropout=None):
        super(SemGCN_Extractor, self).__init__()
        _gconv_input = [_GraphConv(adj, in_chans, hid_dim, p_dropout=p_dropout)]
        _gconv_layers = []
        
        for i in range(num_layers):
            _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
            
        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_layers = nn.Sequential(*_gconv_layers)
        # Note: We removed gconv_output to keep features at hid_dim size

    def forward(self, x):
        out = self.gconv_input(x)
        out = self.gconv_layers(out)
        return out


# ==========================================
# Hybrid Model: 3DGCN (Spatial) + TPA (Temporal)
# ==========================================
class Hybrid_GCN_TPA(nn.Module):
    def __init__(self, adj, adj_temporal, num_frame=243, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio   # temporal embed_dim
        out_dim = 3                   # output dimension

        # 1. Spatial Extractor: 3DGCN
        self.spatial_gcn = SemGCN_Extractor(adj=adj, hid_dim=embed_dim_ratio, in_chans=in_chans, num_layers=4, p_dropout=drop_rate)
        
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Spatial_norm = norm_layer(embed_dim_ratio)

        # 2. Temporal Extractor: TPA + TTE blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth + 1)]
        
        from common.model_ktpformer import TPAttention
        self.tpattention = TPA_Block(
                adj_temporal, num_frame, dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, attention=TPAttention, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer)
        
        self.Temporal_norm = norm_layer(embed_dim)
        
        self.TTEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+1], norm_layer=norm_layer, comb=False, changedim=False, currentdim=i+1, depth=depth)
            for i in range(depth)])
            
        self.block_depth = depth

        # 3. Output Head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )

    def GCN_forward(self, x):
        b, f, n, c = x.shape
        x = rearrange(x, 'b f n c -> (b f) n c')
        
        # Extract features using 3DGCN
        x = self.spatial_gcn(x)
        
        # Add spatial positional embedding (like KPA does)
        x = x + self.Spatial_pos_embed
        x = self.Spatial_norm(x)
        
        # Reshape for Temporal Processing
        x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)
        return x

    def TPA_forward(self, x):
        x = self.tpattention(x)
        x = self.Temporal_norm(x)
        return x

    def Temporal_blocks_forward(self, x):
        b_n, f, cw = x.shape
        
        for i in range(self.block_depth):
            tteblock = self.TTEblocks[i]
            x = tteblock(x)
            x = self.Temporal_norm(x)
        return x

    def forward(self, x):
        b, f, n, c = x.shape
        
        # 1. 空間特徵提取 (3DGCN)
        x = self.GCN_forward(x)  # shape: (b n) f cw
        
        # 2. 軌跡特徵提取 (TPA)
        x = self.TPA_forward(x)  # shape: (b n) f cw
        
        # 3. 時間注意力區塊 (TTE)
        x = self.Temporal_blocks_forward(x) # shape: (b n) f cw
        
        # 4. 回歸預測 (Head)
        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        x = self.head(x) # shape: b f n 3
        
        x = x.view(b, f, n, -1)
        return x
