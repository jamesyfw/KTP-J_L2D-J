# 六關節 L2D-J 正式驗證（S9+S11、全部4台攝影機、NPZ輸入）
# python "3DL2D_from_images(1-6).py"

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from thop import profile as thop_profile



class PRFKKeypointMonitor:
    def __init__(self, dist_threshold=0.0, dir_threshold=0.0, acc_threshold=0.0, theta_threshold=0.0):
        """
        初始化 PRFK 關鍵點監控器 (按個別關節座標判斷變化)
        :param dist_threshold: 關鍵點移動距離閾值 (預設 20.0)
        """
        self.dist_threshold = dist_threshold
        # 方向向量差異閾值（unit vector L2 距離，範圍 0..2）
        self.dir_threshold = dir_threshold
        # 加速度閾值（以三幀二階差分計算）
        self.acc_threshold = acc_threshold
        
        # COCO Keypoint IDs 定義
        # 下半身關鍵點 ID
        self.LEFT_LEG_IDS = {
            'left_hip': 11,
            'left_knee': 13,
            'left_ankle': 15
        }
        self.RIGHT_LEG_IDS = {
            'right_hip': 12,
            'right_knee': 14,
            'right_ankle': 16
        }
        
        # 反向映射：ID -> 名稱
        self.ID_TO_NAME = {
            11: 'left_hip',
            12: 'right_hip',
            13: 'left_knee',
            14: 'right_knee',
            15: 'left_ankle',
            16: 'right_ankle'
        }
        
        # 儲存上一幀的關鍵點數據 {kp_id: {'x': x, 'y': y}, ...}
        self.prev_keypoints = {}
        # 最近三幀的關鍵點歷史，用於計算加速度
        self.frame_history = []
        # 高加速度狀態：True 時只監測，不做後續判斷
        self.high_acceleration_active = False
        # 骨盆軸方向狀態
        self.prev_theta = None
        self.theta_threshold = theta_threshold  # degrees, 可調

    def _measure_keypoint_shift(self, curr_pos, prev_pos):
        """計算兩個 2D 點之間的軸向位移與歐幾里得距離。"""
        dx = abs(curr_pos['x'] - prev_pos['x'])
        dy = abs(curr_pos['y'] - prev_pos['y'])
        euclidean_distance = float(np.hypot(dx, dy))

        is_changed_x = dx >= self.dist_threshold
        is_changed_y = dy >= self.dist_threshold
        is_changed_euclidean = euclidean_distance >= self.dist_threshold

        return {
            'dx': dx,
            'dy': dy,
            'euclidean_distance': euclidean_distance,
            'is_changed_x': is_changed_x,
            'is_changed_y': is_changed_y,
            'is_changed_euclidean': is_changed_euclidean,
            'is_changed': is_changed_x or is_changed_y or is_changed_euclidean,
        }

    def _bone_unit_vectors_from_points(self, hip, knee, ankle):
        """
        從三個關節座標計算兩段骨向量的 unit vectors：hip->knee, knee->ankle
        回傳 dict {'hip_knee': np.array or None, 'knee_ankle': np.array or None}
        若向量長度過短則回傳 None 以避免數值不穩定。
        """
        try:
            v_hk = np.array([knee['x'] - hip['x'], knee['y'] - hip['y']], dtype=float)
            v_ka = np.array([ankle['x'] - knee['x'], ankle['y'] - knee['y']], dtype=float)
        except Exception:
            return {'hip_knee': None, 'knee_ankle': None}

        eps = 1e-8
        norm_hk = np.linalg.norm(v_hk)
        norm_ka = np.linalg.norm(v_ka)

        u_hk = None
        u_ka = None
        if norm_hk > eps:
            u_hk = v_hk / (norm_hk + eps)
        if norm_ka > eps:
            u_ka = v_ka / (norm_ka + eps)

        return {'hip_knee': u_hk, 'knee_ankle': u_ka}

    def _compute_point_acceleration(self, curr_pos, prev_pos, prev2_pos, dt=1.0):
        """用三幀二階差分計算單一點的加速度向量與大小。"""
        try:
            dt_sq = dt * dt
            ax = (curr_pos['x'] - 2.0 * prev_pos['x'] + prev2_pos['x']) / dt_sq
            ay = (curr_pos['y'] - 2.0 * prev_pos['y'] + prev2_pos['y']) / dt_sq
        except Exception:
            return None

        total = float(np.hypot(ax, ay))
        return {
            'ax': float(ax),
            'ay': float(ay),
            'total': total,
        }

    def _compute_frame_acceleration(self, curr_keypoints, prev_keypoints, prev2_keypoints):
        """計算目前這一幀的整體加速度，使用 lower-body joint 的最大加速度作為 A_total。"""
        acc_by_joint = {}
        for kp_id, curr_pos in curr_keypoints.items():
            if kp_id in prev_keypoints and kp_id in prev2_keypoints:
                acc = self._compute_point_acceleration(curr_pos, prev_keypoints[kp_id], prev2_keypoints[kp_id])
                if acc is not None:
                    acc_by_joint[kp_id] = acc

        if not acc_by_joint:
            return None

        max_joint_id = max(acc_by_joint, key=lambda k: acc_by_joint[k]['total'])
        return {
            'total': acc_by_joint[max_joint_id]['total'],
            'max_joint_id': max_joint_id,
            'by_joint': acc_by_joint,
        }

    def _compare_keypoints(self, curr_keypoints, prev_keypoints=None):
        """
        比較當前幀和上一幀的關鍵點，找出變化的關鍵點
        :param curr_keypoints: dict {kp_id: {'x': x, 'y': y}, ...}
        :return: dict (包含變化的關鍵點及其距離)
        """
        if prev_keypoints is None:
            prev_keypoints = self.prev_keypoints

        changed_keypoints = {}

        for kp_id, curr_pos in curr_keypoints.items():
            if kp_id in prev_keypoints:
                prev_pos = prev_keypoints[kp_id]
                shift_info = self._measure_keypoint_shift(curr_pos, prev_pos)

                if shift_info['is_changed']:
                    kp_name = self.ID_TO_NAME.get(kp_id, f"kp_{kp_id}")
                    changed_keypoints[kp_id] = {
                        'name': kp_name,
                        **shift_info,
                        'prev_pos': prev_pos,
                        'curr_pos': curr_pos
                    }

        return changed_keypoints

    def _detect_changed_legs(self, curr_keypoints, prev_keypoints=None, include_direction=True):
        """
        判斷左腿和右腿是否有任一關節超過閾值
        :param curr_keypoints: dict {kp_id: {'x': x, 'y': y}, ...}
        :return: dict (有變化的腿部與其三個關節座標)
        """
        if prev_keypoints is None:
            prev_keypoints = self.prev_keypoints

        changed_legs = {}

        for leg_name, leg_ids in (("left_leg", self.LEFT_LEG_IDS), ("right_leg", self.RIGHT_LEG_IDS)):
            changed_joints = []

            # 檢查每個關節位移（x/y/euclidean）
            for joint_name, kp_id in leg_ids.items():
                # 情況 A: 上一幀沒有，這一幀有 -> 視為變化
                if kp_id in curr_keypoints and kp_id not in prev_keypoints:
                    changed_joints.append({
                        'name': joint_name,
                        'dx': 999.0,
                        'dy': 999.0,
                        'euclidean_distance': 999.0,
                        'is_changed_x': True,
                        'is_changed_y': True,
                        'is_changed_euclidean': True,
                        'is_changed': True,
                    })
                    continue

                # 情況 B: 兩幀都有 -> 比較座標
                if kp_id in curr_keypoints and kp_id in prev_keypoints:
                    curr_pos = curr_keypoints[kp_id]
                    prev_pos = prev_keypoints[kp_id]
                    shift_info = self._measure_keypoint_shift(curr_pos, prev_pos)

                    if shift_info['is_changed']:
                        changed_joints.append({
                            'name': joint_name,
                            **shift_info,
                        })

            # 方向向量差異（只有在位移沒有觸發時才懶計算）
            hip_name, knee_name, ankle_name = list(leg_ids.keys())
            curr_leg_kps = self._get_leg_keypoints(curr_keypoints, leg_ids)
            prev_leg_kps = self._get_leg_keypoints(prev_keypoints, leg_ids) if prev_keypoints else {}

            direction_diffs = None
            direction_threshold_exceeded = False

            if include_direction and not changed_joints:
                # 當前與前一幀的三個關節都存在時，才可計算方向差
                if (hip_name in curr_leg_kps and knee_name in curr_leg_kps and ankle_name in curr_leg_kps
                        and hip_name in prev_leg_kps and knee_name in prev_leg_kps and ankle_name in prev_leg_kps):
                    curr_units = self._bone_unit_vectors_from_points(
                        curr_leg_kps[hip_name], curr_leg_kps[knee_name], curr_leg_kps[ankle_name]
                    )
                    prev_units = self._bone_unit_vectors_from_points(
                        prev_leg_kps[hip_name], prev_leg_kps[knee_name], prev_leg_kps[ankle_name]
                    )

                    diffs = [None, None]
                    keys = ['hip_knee', 'knee_ankle']
                    for i, k in enumerate(keys):
                        u_c = curr_units.get(k)
                        u_p = prev_units.get(k)
                        if u_c is not None and u_p is not None:
                            diffs[i] = float(np.linalg.norm(u_c - u_p))

                    valid = [d for d in diffs if d is not None]
                    max_diff = max(valid) if valid else None
                    direction_diffs = {'hip_knee': diffs[0], 'knee_ankle': diffs[1], 'max': max_diff}
                    if max_diff is not None:
                        direction_threshold_exceeded = max_diff >= self.dir_threshold

            # 如果任一關節變化或方向差異超過閾值，則視為該腿有變動
            if changed_joints or direction_threshold_exceeded:
                changed_legs[leg_name] = {
                    'changed_joints': [joint['name'] for joint in changed_joints],
                    'joints': self._get_leg_keypoints(curr_keypoints, leg_ids),
                    'changes': changed_joints,
                    'direction_change': direction_threshold_exceeded,
                    'direction_diffs': direction_diffs,
                }

        return changed_legs

    def _build_acceleration_release_changed_legs(self, curr_keypoints):
        """高加速度解除時，將兩條腿都視為變動幀，讓後續模型可重算。"""
        changed_legs = {}
        for leg_name, leg_ids in (("left_leg", self.LEFT_LEG_IDS), ("right_leg", self.RIGHT_LEG_IDS)):
            changed_legs[leg_name] = {
                'changed_joints': list(leg_ids.keys()),
                'joints': self._get_leg_keypoints(curr_keypoints, leg_ids),
                'changes': [],
                'acceleration_release': True,
            }
        return changed_legs

    def _get_leg_keypoints(self, keypoints_dict, leg_ids):
        """
        取得單一腿部的三個關節座標
        :param keypoints_dict: dict {kp_id: {'x': x, 'y': y}, ...}
        :param leg_ids: dict, e.g. {'left_hip': 11, 'left_knee': 13, 'left_ankle': 15}
        :return: dict {joint_name: {'x': x, 'y': y}, ...}
        """
        leg_keypoints = {}
        for joint_name, kp_id in leg_ids.items():
            if kp_id in keypoints_dict:
                leg_keypoints[joint_name] = keypoints_dict[kp_id]
        return leg_keypoints

    def _build_model_input_samples(self, changed_legs):
        """
        將變化腿部資料轉成可直接送進模型的 3 點輸入格式
        每一腿對應一筆 sample；若雙腿都變，會有兩筆 sample。
        :param changed_legs: dict
        :return: list of dict, e.g. [{'leg': 'left_leg', 'points': [[x,y],[x,y],[x,y]]}, ...]
        """
        samples = []
        point_order = {
            'left_leg': ['left_hip', 'left_knee', 'left_ankle'],
            'right_leg': ['right_hip', 'right_knee', 'right_ankle']
        }

        for leg_name in ('left_leg', 'right_leg'):
            if leg_name not in changed_legs:
                continue

            joints = changed_legs[leg_name].get('joints', {})
            ordered_points = []
            for joint_name in point_order[leg_name]:
                if joint_name not in joints:
                    ordered_points = []
                    break
                ordered_points.append([joints[joint_name]['x'], joints[joint_name]['y']])

            if len(ordered_points) == 3:
                samples.append({
                    'leg': leg_name,
                    'points': ordered_points
                })

        return samples

    def _compute_theta_from_2d(self, curr_keypoints):
        """計算左右髖形成的 2D 骨盆軸角度，不需要額外校正常數。"""
        try:
            hip_left = curr_keypoints.get(self.LEFT_LEG_IDS['left_hip'])
            hip_right = curr_keypoints.get(self.RIGHT_LEG_IDS['right_hip'])
        except Exception:
            return None

        if not (hip_left and hip_right):
            return None

        dx = float(hip_right['x'] - hip_left['x'])
        dy = float(hip_right['y'] - hip_left['y'])
        if dx == 0.0 and dy == 0.0:
            return None

        return math.degrees(math.atan2(dy, dx)) % 360.0

    def _extract_keypoints_dict(self, keypoints_data):
        """
        從關鍵點列表中提取左腿/右腿的座標
        :param keypoints_data: list of dict, e.g., [{'id':..., 'x':..., 'y':..., 'score':...}, ...]
        :return: dict {kp_id: {'x': x, 'y': y}, ...}
        """
        keypoints_dict = {}
        lower_body_ids = set(self.LEFT_LEG_IDS.values()) | set(self.RIGHT_LEG_IDS.values())
        
        for kp in keypoints_data:
            kp_id = kp.get('id')
            if kp_id in lower_body_ids:
                x = kp.get('x', None)
                y = kp.get('y', None)
                if x is not None and y is not None:
                    keypoints_dict[kp_id] = {'x': x, 'y': y}
        
        return keypoints_dict

    def update(self, keypoints_data):
        """
        傳入當前幀的關鍵點數據，並與上一幀比較 (如果有的話)
        :param keypoints_data: dict, e.g., {'id': 0, 'keypoints': [...]}
        :return: dict (包含分析結果)
        """
        curr_keypoints = keypoints_data.get('keypoints', [])
        curr_keypoints_dict = self._extract_keypoints_dict(curr_keypoints)

        # 保留最近三幀做加速度判斷
        self.frame_history.append(curr_keypoints_dict.copy())
        if len(self.frame_history) > 3:
            self.frame_history.pop(0)
        
        # 初始化結果
        result = {
            "current_keypoints": curr_keypoints_dict,
            "status": "Target Detected",
            "details": "First Frame - No history to compare",
            "changed_legs": {},
            "model_input_samples": [],
            "baseline_updated": False,
            "acceleration": None,
            "acceleration_high": False,
            "acceleration_released": False,
        }
        
        # 如果沒有提取到關鍵點，返回錯誤
        if not curr_keypoints_dict:
            result["status"] = "Error"
            result["details"] = "No lower body keypoints detected"
            return result
        
        # 第一幀只用來建立 baseline，不做比較
        if not self.prev_keypoints:
            self.prev_keypoints = curr_keypoints_dict.copy()
            # 同步嘗試更新 prev_theta（若可計算）
            theta = self._compute_theta_from_2d(curr_keypoints_dict)
            if theta is not None:
                self.prev_theta = theta
            result["baseline_updated"] = True
            return result

        # 1) 加速度優先：只有在有三幀資料時才計算
        acceleration_info = None
        if len(self.frame_history) >= 3:
            acceleration_info = self._compute_frame_acceleration(
                self.frame_history[-1],
                self.frame_history[-2],
                self.frame_history[-3],
            )
            result["acceleration"] = acceleration_info

        # 高加速度狀態：只觀察，不做後續判斷
        if acceleration_info is not None:
            if self.high_acceleration_active:
                if acceleration_info["total"] <= self.acc_threshold:
                    # 第一次降回閾值以下：這一幀直接視為變動幀並更新 baseline
                    self.high_acceleration_active = False
                    result["acceleration_released"] = True
                    result["status"] = "STATUS: Changed"
                    result["details"] = "TYPE: Acceleration Released - Force Update"

                    changed_legs = self._build_acceleration_release_changed_legs(curr_keypoints_dict)
                    result["changed_legs"] = changed_legs
                    result["model_input_samples"] = self._build_model_input_samples(changed_legs)

                    self.prev_keypoints = curr_keypoints_dict.copy()
                    # 同步嘗試更新 prev_theta（若可計算）
                    theta = self._compute_theta_from_2d(curr_keypoints_dict)
                    if theta is not None:
                        self.prev_theta = theta
                    result["baseline_updated"] = True
                    return result

                result["acceleration_high"] = True
                result["status"] = "STATUS: High Acceleration"
                result["details"] = "TYPE: Monitoring Only"
                return result

            if acceleration_info["total"] > self.acc_threshold:
                self.high_acceleration_active = True
                result["acceleration_high"] = True
                result["status"] = "STATUS: High Acceleration"
                result["details"] = "TYPE: Monitoring Only"
                return result

        # 2) 骨盆軸方向（依照論文順序移到距離與方向判定之前）
        theta = self._compute_theta_from_2d(curr_keypoints_dict)
        if theta is not None:
            # 若先前已有 theta，計算差值
            if self.prev_theta is not None:
                # 計算最短旋轉路徑，避免穿越 -180/180 邊界造成大跳動
                t1 = theta % 360.0
                t0 = self.prev_theta % 360.0
                raw_diff = abs(t1 - t0)
                delta_theta = min(raw_diff, 360.0 - raw_diff)
                if delta_theta > self.theta_threshold:
                    # 當作變動幀，標記並更新 baseline
                    changed_legs = self._build_acceleration_release_changed_legs(curr_keypoints_dict)
                    # 標註為 orientation 變動
                    for v in changed_legs.values():
                        v['orientation_change'] = True
                        v['theta'] = theta
                        v['delta_theta'] = delta_theta

                    result["changed_legs"] = changed_legs
                    result["model_input_samples"] = self._build_model_input_samples(changed_legs)
                    result["status"] = "STATUS: Changed"
                    result["details"] = "TYPE: Orientation Change"
                    self.prev_keypoints = curr_keypoints_dict.copy()
                    result["baseline_updated"] = True
                    self.prev_theta = theta
                    return result
            # 若未觸發變動，仍更新 prev_theta 供下幀比較
            self.prev_theta = theta

        # 3) f 與 f-1 的關節移動距離檢查
        prev_frame_keypoints = self.frame_history[-2] if len(self.frame_history) >= 2 else {}
        changed_legs = self._detect_changed_legs(
            curr_keypoints_dict,
            prev_keypoints=prev_frame_keypoints,
            include_direction=False,
        )

        # 4) 速度方向長度（保留原本的方向向量判斷公式）
        if not changed_legs:
            changed_legs = self._detect_changed_legs(
                curr_keypoints_dict,
                prev_keypoints=self.prev_keypoints,
                include_direction=True,
            )

        result["changed_legs"] = changed_legs
        result["model_input_samples"] = self._build_model_input_samples(changed_legs)

        if not changed_legs:
            result["status"] = "STATUS: No Changed"
            result["details"] = "TYPE: No Significant Changes Detected"
        else:
            left_changed = 'left_leg' in changed_legs
            right_changed = 'right_leg' in changed_legs

            if left_changed and right_changed:
                result["status"] = "STATUS: All Changed"
                result["details"] = "TYPE: Both Legs Changed"
            elif left_changed:
                result["status"] = "STATUS: Changed"
                result["details"] = "TYPE: Left Leg Changed"
            else:
                result["status"] = "STATUS: Changed"
                result["details"] = "TYPE: Right Leg Changed"

            # 只有偵測到變動時，才把當前幀推進成新的 baseline
            self.prev_keypoints = curr_keypoints_dict.copy()
            # 同步嘗試更新 prev_theta（若可計算）
            theta = self._compute_theta_from_2d(curr_keypoints_dict)
            if theta is not None:
                self.prev_theta = theta
            result["baseline_updated"] = True

        return result


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
L2D_MODEL_PATH = os.path.join(
    PROJECT_ROOT, 'checkpoint_hybrid', 'L2D-J_best.pth'
)
H36M_2D_NPZ_PATH = os.path.join(
    PROJECT_ROOT, 'data', 'data_2d_h36m_cpn_ft_h36m_dbb.npz'
)
H36M_3D_NPZ_PATH = os.path.join(
    PROJECT_ROOT, 'data', 'data_3d_h36m.npz'
)
IGA_WINDOW = 1
PRFK_H36M_TO_COCO_IDS = [
    (11, 4), (13, 5), (15, 6),
    (12, 1), (14, 2), (16, 3),
]
LOWER_BODY_JOINT_ORDER = [
    'left_hip', 'left_knee', 'left_ankle',
    'right_hip', 'right_knee', 'right_ankle',
]
LOWER_BODY_JOINT_TO_INDEX = {
    name: index for index, name in enumerate(LOWER_BODY_JOINT_ORDER)
}
L2D_INPUT_H36M_INDICES = [1, 2, 3, 4, 5, 6]



class LowerBodyIGANet(nn.Module):
    """Six-joint IGANet architecture used by L2D-J_best.pth."""

    def __init__(self, depth=3, channel=512):
        super().__init__()
        from model.graph_frames import Graph
        from model.model_IGANet import IGANet, encoder

        full_adj = Graph('hm36_gt', 'spatial', pad=1).A
        lower_adj = full_adj[
            :,
            L2D_INPUT_H36M_INDICES,
            :,
        ][
            :,
            :,
            L2D_INPUT_H36M_INDICES,
        ].copy()
        column_mass = lower_adj.sum(axis=(0, 1), keepdims=True)
        lower_adj = np.divide(
            lower_adj,
            column_mass,
            out=np.zeros_like(lower_adj),
            where=column_mass > 0,
        )

        self.A = nn.Parameter(
            torch.tensor(lower_adj, dtype=torch.float32),
            requires_grad=False,
        )
        self.encoder = encoder(2, channel // 2, channel)
        self.IGANet = IGANet(
            depth=depth,
            embed_dim=channel,
            adj=self.A,
            length=len(L2D_INPUT_H36M_INDICES),
        )
        self.fcn = nn.Linear(channel, 3)
        self.input_h36m_indices = tuple(L2D_INPUT_H36M_INDICES)
        self.is_lower_body_model = True

    def forward(self, x):
        x = rearrange(x, 'b f j c -> (b f) j c').contiguous()
        x = self.encoder(x)
        x = self.IGANet(x)
        x = self.fcn(x)
        return rearrange(x, 'b j c -> b 1 j c').contiguous()


def load_h36m_data_helpers():
    """使用專案內的 common/ 載入及正規化 Human3.6M。"""
    from common.camera import normalize_screen_coordinates, world_to_camera
    from common.h36m_dataset import Human36mDataset

    def create_2d_data(data_path, dataset):
        positions_2d = np.load(
            data_path, allow_pickle=True
        )['positions_2d'].item()
        for subject in positions_2d:
            for action in positions_2d[subject]:
                for camera_index, keypoints in enumerate(
                        positions_2d[subject][action]):
                    camera = dataset.cameras()[subject][camera_index]
                    keypoints = keypoints.copy()
                    keypoints[..., :2] = normalize_screen_coordinates(
                        keypoints[..., :2],
                        w=camera['res_w'],
                        h=camera['res_h'],
                    )
                    positions_2d[subject][action][camera_index] = keypoints
        return positions_2d

    def read_3d_data(dataset):
        for subject in dataset.subjects():
            for action in dataset[subject]:
                animation = dataset[subject][action]
                positions_3d = []
                for camera in animation['cameras']:
                    position = world_to_camera(
                        animation['positions'],
                        R=camera['orientation'],
                        t=camera['translation'],
                    )
                    position[:, :] -= position[:, :1]
                    positions_3d.append(position)
                animation['positions_3d'] = positions_3d
        return dataset

    return create_2d_data, read_3d_data, Human36mDataset



def h36m_frame_to_prfk_payload(frame_2d, frame_index=None):
    """Convert one H36M frame into PRFK payload format."""
    keypoints = []
    for kp_id, h36m_idx in PRFK_H36M_TO_COCO_IDS:
        x, y = frame_2d[h36m_idx]
        keypoints.append({
            'id': kp_id,
            'x': float(x),
            'y': float(y),
            'score': 1.0,
        })

    payload = {'keypoints': keypoints}
    if frame_index is not None:
        payload['frame_index'] = int(frame_index)
    return payload


def build_prfk_update_plan(
        pts_2d_h36m,
        dist_threshold,
        dir_threshold=0.0,
        acc_threshold=0.0,
        theta_threshold=0.0):
    """Build a per-frame PRFK plan with combined lower-body joint changes."""
    # 四個閾值同時為0代表PRFK不篩除任何幀，所有輸入都送入3D模型。
    if all(abs(float(value)) < 1e-12 for value in (
            dist_threshold, dir_threshold, acc_threshold, theta_threshold)):
        return [
            {
                'left_update': True,
                'right_update': True,
                'left_changed_joints': LOWER_BODY_JOINT_ORDER[:3],
                'right_changed_joints': LOWER_BODY_JOINT_ORDER[3:],
                'changed_joints': LOWER_BODY_JOINT_ORDER.copy(),
                'changed_joint_indices': list(range(6)),
                'any_update': True,
            }
            for _ in range(len(pts_2d_h36m))
        ]

    monitor = PRFKKeypointMonitor(
        dist_threshold=dist_threshold,
        dir_threshold=dir_threshold,
        acc_threshold=acc_threshold,
        theta_threshold=theta_threshold,
    )
    frame_plan = []

    for frame_idx, frame_2d in enumerate(pts_2d_h36m):
        result = monitor.update(h36m_frame_to_prfk_payload(frame_2d, frame_idx))
        changed_legs = result.get('changed_legs', {})

        left_changed_joints = changed_legs.get('left_leg', {}).get('changed_joints', [])
        right_changed_joints = changed_legs.get('right_leg', {}).get('changed_joints', [])
        changed_joints = [
            joint_name for joint_name in LOWER_BODY_JOINT_ORDER
            if joint_name in left_changed_joints or joint_name in right_changed_joints
        ]

        frame_plan.append({
            'left_update': frame_idx == 0 or 'left_leg' in changed_legs,
            'right_update': frame_idx == 0 or 'right_leg' in changed_legs,
            'left_changed_joints': left_changed_joints,
            'right_changed_joints': right_changed_joints,
            'changed_joints': changed_joints,
            'changed_joint_indices': [LOWER_BODY_JOINT_TO_INDEX[name] for name in changed_joints],
            'any_update': frame_idx == 0 or bool(changed_legs),
        })

    return frame_plan


def compute_iga_flops_per_frame(model):
    """Compute IGA FLOPs for a single frame (IGA is single-frame model)."""
    import copy
    try:
        if isinstance(model, (torch.jit.ScriptModule, torch.jit.RecursiveScriptModule)):
            raise TypeError('thop does not support TorchScript models')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inner = model.module if hasattr(model, 'module') else model
        input_indices = getattr(inner, 'input_h36m_indices', None)
        joint_count = len(input_indices) if input_indices is not None else 17
        dummy = torch.randn(
            1,
            1,
            joint_count,
            2,
            device=device,
        )
        inner_copy = copy.deepcopy(inner).to(device)
        flops, _ = thop_profile(inner_copy, inputs=(dummy,), verbose=False)
        del inner_copy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return flops
    except Exception as e:
        print(f"[WARNING] Failed to compute IGA FLOPs dynamically: {e}")
        print("[WARNING] Using fallback value 0.0 GFLOPs")
        return 0.0


def _select_output(pred):
    """Extract 3D output from model (single-frame model outputs already (B, 1, J, 3))."""
    if pred.ndim == 4 and pred.shape[1] == 1:
        return pred  # Already (B, 1, J, 3)
    if pred.ndim == 3:
        return pred.unsqueeze(1)  # (B, J, 3) -> (B, 1, J, 3)
    raise ValueError(f"Unexpected prediction shape: {tuple(pred.shape)}")


def select_model_2d_input(points_2d, model):
    """只選出實際送入六關節模型的2D點，並檢查輸入形狀。"""
    if points_2d.ndim != 3 or points_2d.shape[-1] != 2:
        raise ValueError(
            f'Expected 2D input shape (frames, joints, 2), got '
            f'{tuple(points_2d.shape)}'
        )

    input_indices = getattr(model, 'input_h36m_indices', None)
    expected_joints = len(input_indices) if input_indices is not None else 17
    source_joint_count = points_2d.shape[1]
    if source_joint_count == expected_joints:
        # 自訂輸入本來就是6點，不需要再次切片。
        selected = points_2d
        selection_text = 'input already has the expected joints'
    elif input_indices is not None and max(input_indices) < source_joint_count:
        # Human3.6M NPZ原本有17點；這裡只取index 1～6。
        # 切片後才會轉成PyTorch tensor並送入IGANet。
        selected = points_2d[:, list(input_indices), :]
        selection_text = f'selected H36M indices {list(input_indices)}'
    else:
        raise ValueError(
            f'Model requires {expected_joints} joints, but input has '
            f'{source_joint_count}'
        )

    # 若不是6點就立即停止，避免不小心把17點送進L2D模型。
    assert selected.shape[1] == expected_joints
    if not getattr(model, '_input_shape_logged', False):
        print(
            '[INPUT CHECK] '
            f'original={tuple(points_2d.shape)} | {selection_text} | '
            f'model_input={tuple(selected.shape)}',
            flush=True,
        )
        model._input_shape_logged = True
    return np.ascontiguousarray(selected)


def run_iga_inference(pts_2d_norm, frame_plan, iga_model, use_prfk=True, pred_cache=None):
    """Run IGA inference frame-by-frame (single-frame model)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    predicted_frames = []
    last_pred = None
    num_updates = 0
    num_skips = 0
    t_3d_start = time.time()

    with torch.no_grad():
        for i, info in enumerate(frame_plan):
            should_update = True if not use_prfk else bool(info['any_update'])

            if not should_update and last_pred is not None:
                num_skips += 1
                predicted_frames.append(last_pred)
                continue

            if pred_cache is not None and i in pred_cache:
                pred_3d = pred_cache[i]
            else:
                frame_np = pts_2d_norm[i:i+1]
                frame_np = select_model_2d_input(frame_np, iga_model)
                frame_2d = (
                    torch.from_numpy(frame_np)
                    .float()
                    .unsqueeze(1)
                    .to(device)
                )
                pred = iga_model(frame_2d)
                pred_3d = _select_output(pred)  # (B, 1, J, 3)
                
                if pred_cache is not None:
                    pred_cache[i] = pred_3d

            last_pred = pred_3d
            predicted_frames.append(pred_3d)
            num_updates += 1

    predicted_3d = torch.cat(predicted_frames, dim=1)  # (1, T=num_frames, J, 3)
    
    t_3d = time.time() - t_3d_start

    return predicted_3d, t_3d, num_updates, num_skips


def run_iga_inference_batched(
        pts_2d_norm,
        frame_plan,
        iga_model,
        use_prfk=True,
        batch_size=2048):
    """Run update frames in batches, then apply PRFK hold-last routing."""
    device = next(iga_model.parameters()).device
    num_frames = len(pts_2d_norm)

    update_indices = []
    has_prediction = False
    for index, info in enumerate(frame_plan):
        should_update = True if not use_prfk else bool(info['any_update'])
        if should_update or not has_prediction:
            update_indices.append(index)
            has_prediction = True

    selected = select_model_2d_input(
        pts_2d_norm[update_indices],
        iga_model,
    )

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    t_3d_start = time.perf_counter()
    predicted_updates = []
    with torch.no_grad():
        for start in range(0, len(selected), batch_size):
            batch = torch.from_numpy(
                np.ascontiguousarray(selected[start:start + batch_size])
            ).float().unsqueeze(1).to(device)
            prediction = _select_output(iga_model(batch))[:, 0]
            predicted_updates.append(prediction.float().cpu().numpy())

    predicted_updates = np.concatenate(predicted_updates, axis=0)
    predicted_all = np.empty(
        (num_frames, predicted_updates.shape[1], 3),
        dtype=np.float32,
    )
    update_cursor = 0
    current_prediction = None
    update_set = set(update_indices)
    for index in range(num_frames):
        if index in update_set:
            current_prediction = predicted_updates[update_cursor]
            update_cursor += 1
        predicted_all[index] = current_prediction

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t_3d_start
    num_updates = len(update_indices)
    return (
        torch.from_numpy(predicted_all).unsqueeze(0),
        elapsed,
        num_updates,
        num_frames - num_updates,
    )


def build_iga_model(checkpoint_path, num_frame=IGA_WINDOW):
    """Load either the original 17-joint IGANet or six-joint L2D model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    is_l2d = (
        isinstance(checkpoint, dict)
        and 'model_pos' in checkpoint
        and (
            checkpoint.get('input_shape') == [1, 1, 6, 2]
            or checkpoint.get('lower_body_indices') == [1, 2, 3, 4, 5, 6]
        )
    )

    if is_l2d:
        model = LowerBodyIGANet()
        state_dict = checkpoint['model_pos']
        print('[INFO] Building six-joint L2D IGANet')
    else:
        try:
            from model.model_IGANet import Model as IGANetModel
            print("[INFO] Building original 17-joint IGANet")
        except Exception as e:
            print(f"[ERROR] Failed to import IGANet model: {e}")
            raise

        class Args:
            layers = 3
            channel = 512
            n_joints = 17

        model = IGANetModel(Args())
        model.input_h36m_indices = None
        model.is_lower_body_model = False

        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model_pos' in checkpoint:
            state_dict = checkpoint['model_pos']
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint and not is_l2d:
        state_dict = checkpoint['state_dict']

    # Remove 'module.' prefix if present (from DataParallel)
    if (
            isinstance(state_dict, dict)
            and state_dict
            and list(state_dict.keys())[0].startswith('module.')):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Load to device and model
    model = model.to(device)
    load_result = model.load_state_dict(state_dict, strict=is_l2d)
    print(
        f"[INFO] IGANet checkpoint loaded successfully: {checkpoint_path}"
    )
    if not is_l2d and (load_result.missing_keys or load_result.unexpected_keys):
        print(f'[WARNING] Missing keys: {load_result.missing_keys}')
        print(f'[WARNING] Unexpected keys: {load_result.unexpected_keys}')

    model.eval()
    return model


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_path', default=H36M_2D_NPZ_PATH)
    parser.add_argument('--npz_3d_path', default=H36M_3D_NPZ_PATH)
    parser.add_argument('--subject', default='S9,S11')
    parser.add_argument('--camera_idx', type=int, default=-1)
    parser.add_argument('--checkpoint_hybrid', default=L2D_MODEL_PATH)
    parser.add_argument('--prfk_acc_threshold', type=float, default=0.0)
    parser.add_argument('--prfk_theta_threshold', type=float, default=0.0)
    parser.add_argument('--prfk_dist_threshold', type=float, default=0.0)
    parser.add_argument('--prfk_dir_threshold', type=float, default=0.0)
    args = parser.parse_args()

    # --------------------------------------------------------------
    # 1. 讀取 Human3.6M CPN 2D NPZ。
    #
    # positions_2d 的資料階層為：
    # positions_2d[subject][action][camera]
    # 每段陣列形狀為 (frames, 17, 2)，座標仍是影像像素。
    # --------------------------------------------------------------
    raw_positions_2d = np.load(
        args.npz_path,
        allow_pickle=True,
    )['positions_2d'].item()
    subjects = [
        value.strip()
        for value in args.subject.split(',')
        if value.strip()
    ]
    missing_subjects = [
        subject
        for subject in subjects
        if subject not in raw_positions_2d
    ]
    if missing_subjects:
        raise KeyError(
            f'Subjects {missing_subjects!r} are not in {args.npz_path}'
        )

    # --------------------------------------------------------------
    # 2. 讀取 Human3.6M 3D NPZ並套用官方資料前處理。
    #
    # read_3d_data：
    #   世界座標 -> 各攝影機座標，並扣除joint 0（骨盆）位置，
    #   因此3D答案是骨盆置中、單位為公尺。
    #
    # create_2d_data：
    #   依每台攝影機的解析度，把CPN像素座標正規化成
    #   IGANet訓練時使用的座標範圍。
    # --------------------------------------------------------------
    create_2d_data, read_3d_data, Human36mDataset = (
        load_h36m_data_helpers()
    )
    dataset = read_3d_data(Human36mDataset(args.npz_3d_path))
    normalized_positions_2d = create_2d_data(
        args.npz_path,
        dataset,
    )

    # 載入六關節 L2D-J_best.pth。模型本身的節點數就是6，
    # 不是先跑17點之後才把上半身輸出丟掉。
    print('[NPZ inference mode] Loading IGA model...')
    iga_model = build_iga_model(args.checkpoint_hybrid, num_frame=1)
    iga_flops = compute_iga_flops_per_frame(iga_model)

    num_frames = 0
    num_updates = 0
    sequence_count = 0
    joint_error_sum = torch.zeros(6, dtype=torch.float64)
    total_3d_seconds = 0.0

    # --------------------------------------------------------------
    # 3. 逐一配對 subject -> action -> camera。
    #
    # --camera_idx -1 代表四台攝影機全部驗證。
    # 每個動作／攝影機都視為獨立序列，不能跨序列沿用PRFK狀態。
    # --------------------------------------------------------------
    for subject in subjects:
        for action_name in sorted(raw_positions_2d[subject]):
            raw_cameras = raw_positions_2d[subject][action_name]
            normalized_cameras = normalized_positions_2d[
                subject
            ][action_name]
            gt_cameras = dataset[subject][action_name]['positions_3d']
            if args.camera_idx == -1:
                camera_indices = range(len(raw_cameras))
            elif 0 <= args.camera_idx < len(raw_cameras):
                camera_indices = [args.camera_idx]
            else:
                raise IndexError(
                    f'Camera {args.camera_idx} is unavailable for '
                    f'{subject}/{action_name}'
                )

            for camera_idx in camera_indices:
                raw_sequence = raw_cameras[camera_idx]
                normalized_sequence = normalized_cameras[camera_idx]
                gt_sequence = gt_cameras[camera_idx]
                # CPN 2D與3D GT有些序列尾端可能相差一幀，
                # 因此只取兩者共同長度，避免2D與3D答案錯位。
                sequence_frames = min(
                    len(raw_sequence),
                    len(normalized_sequence),
                    len(gt_sequence),
                )
                raw_sequence = raw_sequence[:sequence_frames]
                normalized_sequence = normalized_sequence[
                    :sequence_frames
                ]
                # --------------------------------------------------
                # 4. 只從17點3D答案取出下半身六點。
                #
                # Human3.6M index：
                # 1=右髖、2=右膝、3=右腳踝、
                # 4=左髖、5=左膝、6=左腳踝。
                #
                # gt_lower形狀：(frames, 6, 3)
                # --------------------------------------------------
                gt_lower = gt_sequence[
                    :sequence_frames,
                    L2D_INPUT_H36M_INDICES,
                    :,
                ].astype(np.float32)

                sequence_count += 1
                num_frames += sequence_frames

                # Reset PRFK at every subject/action/camera boundary.
                frame_plan = build_prfk_update_plan(
                    raw_sequence,
                    dist_threshold=args.prfk_dist_threshold,
                    dir_threshold=args.prfk_dir_threshold,
                    acc_threshold=args.prfk_acc_threshold,
                    theta_threshold=args.prfk_theta_threshold,
                )

                # --------------------------------------------------
                # 5. 執行六關節2D -> 六關節3D推論。
                #
                # normalized_sequence原本是(frames, 17, 2)；
                # run_iga_inference_batched內會呼叫
                # select_model_2d_input，只選[1,2,3,4,5,6]，
                # 真正送入模型的tensor為(batch, 1, 6, 2)。
                # prediction輸出為(1, frames, 6, 3)。
                # 終端機的[INPUT CHECK]會印出實際輸入形狀。
                # --------------------------------------------------
                prediction, elapsed, updates, _ = (
                    run_iga_inference_batched(
                        normalized_sequence,
                        frame_plan,
                        iga_model,
                        use_prfk=True,
                    )
                )
                prediction = prediction[0]
                gt_tensor = torch.from_numpy(gt_lower)
                # --------------------------------------------------
                # 6. 計算每幀、每關節的Protocol #1 MPJPE。
                #
                # ||預測3D - 3D答案||_2，先保留六個關節各自誤差，
                # 最後再對所有S9+S11幀數做加權平均並轉成毫米。
                # --------------------------------------------------
                joint_errors = torch.linalg.vector_norm(
                    prediction - gt_tensor,
                    dim=-1,
                )
                joint_error_sum += joint_errors.double().sum(dim=0)
                num_updates += updates
                total_3d_seconds += elapsed

    baseline_total = num_frames * iga_flops
    prfk_total = num_updates * iga_flops
    arithmetic_flops_total = 2.0 * prfk_total
    baseline_per_frame = baseline_total / num_frames
    prfk_per_frame = prfk_total / num_frames
    # 每個關節的總誤差 / 全部有效幀數，再由公尺轉成毫米。
    per_joint_mpjpe = joint_error_sum / num_frames * 1000.0
    # 六個關節MPJPE的平均就是最後的下半身六點MPJPE。
    combined_mpjpe = per_joint_mpjpe.mean().item()

    print('=' * 70)
    print(f'2D input: {args.npz_path}')
    print(f'3D ground truth: {args.npz_3d_path}')
    print(
        f'Subjects: {",".join(subjects)} | '
        f'camera index: {args.camera_idx} '
        f'| sequences: {sequence_count}'
    )
    print(
        'PRFK thresholds: '
        f'distance={args.prfk_dist_threshold:g}, '
        f'direction={args.prfk_dir_threshold:g}, '
        f'acceleration={args.prfk_acc_threshold:g}, '
        f'theta={args.prfk_theta_threshold:g}'
    )
    print('Model input per frame: (1, 1, 6, 2)')
    print('Model output per frame: (1, 1, 6, 3)')
    print(
        'Joint order: '
        '[R-Hip, R-Knee, R-Ankle, L-Hip, L-Knee, L-Ankle]'
    )
    print(f'Total executed frames: {num_frames}')
    print(f'IGA updates: {num_updates}; skips: {num_frames - num_updates}')
    print(f'IGA per update: {iga_flops / 1e9:.6f} GMac')
    print(f'Baseline total: {baseline_total / 1e12:.6f} TMac')
    print(
        f'Baseline total / total frames: '
        f'{baseline_per_frame / 1e9:.6f} GMac/frame'
    )
    print(f'PRFK-gated total: {prfk_total / 1e12:.6f} TMac')
    print(
        f'Total arithmetic FLOPs (1 MAC = 2 FLOPs): '
        f'{arithmetic_flops_total / 1e12:.6f} TFLOPs'
    )
    print(
        f'PRFK-gated total / total frames: '
        f'{prfk_per_frame / 1e6:.6f} MMac/frame'
    )
    print(
        f'Arithmetic FLOPs per frame (1 MAC = 2 FLOPs): '
        f'{2.0 * prfk_per_frame / 1e6:.6f} MFLOPs/frame'
    )
    joint_names = [
        'R-Hip',
        'R-Knee',
        'R-Ankle',
        'L-Hip',
        'L-Knee',
        'L-Ankle',
    ]
    print('=' * 70)
    print('FINAL SIX-JOINT VALIDATION MPJPE (S9+S11)')
    print('=' * 70)
    for joint_name, error in zip(joint_names, per_joint_mpjpe):
        print(f'{joint_name:<8}: {error.item():.6f} mm')
    print(f'Combined lower-body MPJPE: {combined_mpjpe:.6f} mm')
    print(f'Actual 3D inference time: {total_3d_seconds:.3f} s')
    print(
        f'Average 3D inference time: '
        f'{total_3d_seconds / num_frames * 1000.0:.6f} ms/frame'
    )
    return



if __name__ == '__main__':
    evaluate()
