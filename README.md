# KTP-J 與 L2D-J

本專案包含 **KTP-J** 與 **L2D-J** 兩種下半身六關節的 2D-to-3D 姿態估測方法，並使用 PRFK 判斷是否需要更新3D關節結果。

## 基本執行方式

請先在本專案的根目錄開啟終端機。

### KTP-J

使用 Human3.6M 的 S9、S11 與全部4台攝影機進行驗證：

```bash
python "3DKTP_from_images(1-6).py"
```

### L2D-J

使用 Human3.6M 的 S9、S11 與全部4台攝影機進行驗證：

```bash
python "3DL2D_from_images(1-6).py"
```

## 調整 PRFK 閾值

四個閾值依序為：加速度、骨盆軸角度、移動距離、方向。

```text
--prfk_acc_threshold       加速度閾值
--prfk_theta_threshold     骨盆軸角度閾值
--prfk_dist_threshold      移動距離閾值
--prfk_dir_threshold       方向閾值
```

### KTP-J

```bash
python "3DKTP_from_images(1-6).py" --prfk_acc_threshold 20 --prfk_theta_threshold 10 --prfk_dist_threshold 10 --prfk_dir_threshold 0.3
```

### L2D-J

```bash
python "3DL2D_from_images(1-6).py" --prfk_acc_threshold 20 --prfk_theta_threshold 10 --prfk_dist_threshold 10 --prfk_dir_threshold 0.3
```

未指定參數時，四個閾值預設皆為 `0`，代表所有幀都送入3D模型。

## 指定模型路徑

KTP-J 與 L2D-J 統一使用 `--checkpoint_hybrid` 指定模型。

### KTP-J

```bash
python "3DKTP_from_images(1-6).py" --checkpoint_hybrid "checkpoint_hybrid/KTP-J_best.bin"
```

### L2D-J

```bash
python "3DL2D_from_images(1-6).py" --checkpoint_hybrid "checkpoint_hybrid/L2D-J_best.pth"
```
