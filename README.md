# KTP-J 與 L2D-J

### KTP-J

使用 Human3.6M 的 S9、S11 與全部4台攝影機進行驗證：

```bash
python "3DKTP_from_images(1-6).py"
```

### L2D-J

使用 Human3.6M 的 S9、S11 與全部4台攝影機進行驗證：

```bash
python "3DIGA_from_images(1-6).py"
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
python "3DKTP_from_images(1-6).py" --prfk_acc_threshold 0 --prfk_theta_threshold 0 --prfk_dist_threshold 0 --prfk_dir_threshold 0
```

### L2D-J

```bash
python "3DIGA_from_images(1-6).py" --prfk_acc_threshold 0 --prfk_theta_threshold 0 --prfk_dist_threshold 0 --prfk_dir_threshold 0
```

未指定參數時，四個閾值預設皆為 `0`，代表所有幀都送入3D模型。
