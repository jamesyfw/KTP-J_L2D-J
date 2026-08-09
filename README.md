# KTP-J 與 L2D-J

本專案包含 **KTP-J** 與 **L2D-J** 兩種下半身六關節的 2D-to-3D 姿態估測方法，並使用 PRFK 判斷是否需要更新3D關節結果。

## 基本執行方式

請先在本專案的根目錄開啟終端機。

### KTP-J

使用目前程式內的預設 PRFK 設定執行：

```bash
python "3DKTP_from_images(1-6).py"
```

### L2D-J

使用 Human3.6M 的 S9、S11 與全部4台攝影機進行驗證：

```bash
python "3DIGA_from_images(1-6).py" --npz_path "data/data_2d_h36m_cpn_ft_h36m_dbb.npz" --npz_3d_path "data/data_3d_h36m.npz" --subject "S9,S11" --camera_idx -1
```

> 目前 KTP-J 的 NPZ 驗證流程尚未接入 PRFK，因此 KTP-J 的上述基本指令使用圖片輸入流程；L2D-J 則可以直接使用 NPZ 驗證 PRFK。
