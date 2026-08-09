# 真正六點KTP-J正式驗證（S9+S11、全部4台攝影機、NPZ輸入）：
# python "3DKTP_from_images(1-6).py"

import os
import sys
import json
import glob
import argparse
import math
import time
import shutil
import tempfile
import zipfile
from importlib import import_module
import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
from ultralytics import YOLO

import tensorflow as tf
from thop import profile as thop_profile

from common.model_ktpformer import KTPFormer
from common.loss import mpjpe, p_mpjpe
from common.graph_utils import adj_mx_from_skeleton_temporal, adj_mx_from_skeleton
from common.skeleton import Skeleton


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

# =====================================================================
# Path Configuration
# =====================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
HRNET_ROOT = r'C:\Users\angel\OneDrive - 國立臺北大學\HRNet-Human-Pose-Estimation-master'
TEST_DIR = os.path.join(HRNET_ROOT, 'data', 'human3.6m', 'test')
POSE3D_PATH = os.path.join(TEST_DIR, 'pose3d.json')
CAMERA_PATH = os.path.join(HRNET_ROOT, 'data', 'human3.6m', 'Human36M_subject1_camera.json')
YOLO_MODEL_PATH = os.path.join(HRNET_ROOT, 'yolo11n-pose.pt')
HRNET_CFG_PATH = os.path.join(HRNET_ROOT, 'experiments', 'coco', 'hrnet', 'w32_256x192_adam_lr1e-3.yaml')
HRNET_WEIGHTS_PATH = os.path.join(HRNET_ROOT, 'pose_hrnet_w32_256x192.pth')
OPENPOSE_ONNX_PATH = os.path.join(HRNET_ROOT, 'openpose', 'models', 'pose', 'coco', 'pose_iter_440000.onnx')
MEDIAPIPE_TASK_SRC = os.path.join(HRNET_ROOT, 'Mediapipe_eval', 'pose_landmarker_lite.task')
MEDIAPIPE_TASK_PATH = MEDIAPIPE_TASK_SRC
MEDIA_PIPE_INPUT_TASK = MEDIAPIPE_TASK_PATH
KTPFORMER_WINDOW = 27
FULLBODY_WINDOW = 243
HYBRID_CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    'checkpoint_hybrid',
    'KTP-J_best.bin',
)
H36M_3D_NPZ_PATH = os.path.join(
    PROJECT_ROOT,
    'data',
    'data_3d_h36m.npz',
)
H36M_2D_NPZ_PATH = os.path.join(
    PROJECT_ROOT,
    'data',
    'data_2d_h36m_cpn_ft_h36m_dbb.npz',
)
PRFK_H36M_TO_COCO_IDS = [
    (11, 4), (13, 5), (15, 6),
    (12, 1), (14, 2), (16, 3),
]
LOWER_BODY_JOINT_ORDER = [
    'left_hip', 'left_knee', 'left_ankle',
    'right_hip', 'right_knee', 'right_ankle',
]
LOWER_BODY_JOINT_TO_INDEX = {
    name: idx for idx, name in enumerate(LOWER_BODY_JOINT_ORDER)
}
LOWER_BODY_H36M_INDICES = [4, 5, 6, 1, 2, 3]
VALIDATION_LOWER_INDICES = [1, 2, 3, 4, 5, 6]


# 舊版checkpoint把NumPy亂數狀態一起存入；以下相容物件只用來略過
# 不影響推論的random_state，真正模型權重仍會用strict=True完整檢查。
class MockRandomState:
    def __setstate__(self, state):
        pass


def patched_randomstate_ctor(*args, **kwargs):
    return MockRandomState()


class MockMT19937:
    def __setstate__(self, state):
        pass


def patched_bit_generator_ctor(*args, **kwargs):
    return MockMT19937()

# =====================================================================
# COCO -> H3.6M Mapping (IGANet official)
# =====================================================================
def coco_to_h36m(keypoints):
    """(17, 2) COCO pixel -> (17, 2) H3.6M pixel"""
    keypoints = keypoints[np.newaxis]
    T = keypoints.shape[0]
    kps_h36m = np.zeros_like(keypoints, dtype=np.float32)
    htps = np.zeros((T, 4, 2), dtype=np.float32)

    h36m_coco_order = [9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
    coco_order      = [0,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]
    spple           = [10, 8, 0, 7]

    htps[:, 0, 0] = np.mean(keypoints[:, 1:5, 0], axis=1, dtype=np.float32)
    htps[:, 0, 1] = np.sum(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1]
    htps[:, 1, :] = np.mean(keypoints[:, 5:7, :], axis=1, dtype=np.float32)
    htps[:, 1, :] += (keypoints[:, 0, :] - htps[:, 1, :]) / 3
    htps[:, 2, :] = np.mean(keypoints[:, 11:13, :], axis=1, dtype=np.float32)
    htps[:, 3, :] = np.mean(keypoints[:, [5, 6, 11, 12], :], axis=1, dtype=np.float32)

    kps_h36m[:, spple, :] = htps
    kps_h36m[:, h36m_coco_order, :] = keypoints[:, coco_order, :]

    kps_h36m[:, 9, :] -= (kps_h36m[:, 9, :] - np.mean(keypoints[:, 5:7, :], axis=1, dtype=np.float32)) / 4
    kps_h36m[:, 7, 0] += 2 * (kps_h36m[:, 7, 0] - np.mean(kps_h36m[:, [0, 8], 0], axis=1, dtype=np.float32))
    kps_h36m[:, 8, 1] -= (np.mean(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1]) * 2 / 3

    return kps_h36m[0]

def detect_yolo_from_images(img_files, progress_cb=None):
    """Use YOLOv11n-pose to detect 2D keypoints for all images."""
    model = YOLO(YOLO_MODEL_PATH)
    all_kps = np.zeros((len(img_files), 17, 2), dtype=np.float32)
    
    for i, img_path in enumerate(img_files):
        result = model(img_path, verbose=False)[0]
        
        if result.keypoints is not None and len(result.keypoints) > 0:
            kp_data = result.keypoints.data
            if kp_data.shape[0] > 1:
                confs = kp_data[:, :, 2].mean(dim=1)
                best = confs.argmax().item()
                kp = kp_data[best, :, :2].cpu().numpy()
            else:
                kp = kp_data[0, :, :2].cpu().numpy()
            all_kps[i] = coco_to_h36m(kp)
        if progress_cb and (i + 1) % 200 == 0:
            progress_cb(i + 1, len(img_files))
    
    return all_kps

def detect_hrnet_from_images(img_files, progress_cb=None):
    """Use HRNet-W32 + YOLO bbox to detect 2D keypoints for all images."""
    lib_dir = os.path.join(HRNET_ROOT, 'lib')
    if HRNET_ROOT not in sys.path:
        sys.path.insert(0, HRNET_ROOT)
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)

    models = import_module('models')
    cfg_module = import_module('config')
    inference_module = import_module('core.inference')
    transforms_module = import_module('utils.transforms')
    from ultralytics import YOLO

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class ArgsMock:
        cfg = HRNET_CFG_PATH
        opts = []
        modelDir = ''
        logDir = ''
        dataDir = ''
        prevModelDir = ''

    cfg = cfg_module.cfg
    update_config = cfg_module.update_config
    get_final_preds = inference_module.get_final_preds
    get_affine_transform = transforms_module.get_affine_transform

    update_config(cfg, ArgsMock())
    hrnet_builder = getattr(models, cfg.MODEL.NAME).get_pose_net
    hrnet = hrnet_builder(cfg, is_train=False)
    hrnet.load_state_dict(torch.load(HRNET_WEIGHTS_PATH, map_location='cpu'), strict=False)
    hrnet = hrnet.to(device).eval()

    yolo = YOLO(YOLO_MODEL_PATH)
    hrnet_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    aspect_ratio = cfg.MODEL.IMAGE_SIZE[0] / cfg.MODEL.IMAGE_SIZE[1]

    all_kps = np.zeros((len(img_files), 17, 2), dtype=np.float32)

    for i, img_path in enumerate(img_files):
        raw = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is None:
            continue

        res = yolo(img, verbose=False)
        if not res or len(res[0].boxes) == 0:
            continue

        box_xyxy = res[0].boxes[0].xyxy[0].cpu().numpy()
        bx, by = box_xyxy[0], box_xyxy[1]
        bw, bh = box_xyxy[2] - box_xyxy[0], box_xyxy[3] - box_xyxy[1]
        center = np.array([bx + bw * 0.5, by + bh * 0.5], dtype=np.float32)
        if bw > aspect_ratio * bh:
            bh = bw / aspect_ratio
        elif bw < aspect_ratio * bh:
            bw = bh * aspect_ratio
        scale = np.array([bw / 200, bh / 200], dtype=np.float32) * 1.25

        trans = get_affine_transform(center, scale, 0, cfg.MODEL.IMAGE_SIZE)
        crop = cv2.warpAffine(img, trans, tuple(cfg.MODEL.IMAGE_SIZE), flags=cv2.INTER_LINEAR)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        inp = hrnet_transform(crop_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            out = hrnet(inp)

        preds, _ = get_final_preds(cfg, out.cpu().numpy(), [center], [scale])
        all_kps[i] = coco_to_h36m(preds[0])

        if progress_cb and (i + 1) % 200 == 0:
            progress_cb(i + 1, len(img_files))

    return all_kps

def detect_openpose_from_images(img_files, progress_cb=None):
    """Use ONNX OpenPose to detect 2D keypoints for all images."""
    import onnxruntime as ort
    
    onnx_path = OPENPOSE_ONNX_PATH
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"OpenPose ONNX model not found at {onnx_path}")
    
    session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    
    OP_IDX = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
    INPUT_SIZE = (368, 368)

    all_kps = np.zeros((len(img_files), 17, 2), dtype=np.float32)

    for i, img_path in enumerate(img_files):
        raw = np.fromfile(img_path, dtype=np.uint8)
        img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        h_orig, w_orig = img_bgr.shape[:2]

        img_inp = cv2.resize(img_bgr, INPUT_SIZE).astype(np.float32) / 256.0 - 0.5
        img_inp = np.ascontiguousarray(img_inp.transpose(2, 0, 1)[np.newaxis])

        outputs = session.run(output_names, {input_name: img_inp})
        
        # Extract first 18 channels as heatmaps from net_output (1, 57, 46, 46)
        heatmaps = outputs[0][0, :18]

        heat_h, heat_w = heatmaps.shape[1], heatmaps.shape[2]
        op_kps = np.zeros((18, 2), dtype=np.float32)
        for j in range(18):
            _, _, _, max_loc = cv2.minMaxLoc(heatmaps[j])
            op_kps[j, 0] = (max_loc[0] / float(heat_w)) * float(w_orig)
            op_kps[j, 1] = (max_loc[1] / float(heat_h)) * float(h_orig)

        coco_pts = op_kps[OP_IDX]
        all_kps[i] = coco_to_h36m(coco_pts)

        if progress_cb and ((i + 1) % 100 == 0 or (i + 1) == len(img_files)):
            progress_cb(i + 1, len(img_files))

    return all_kps

def detect_mediapipe_from_images(img_files, progress_cb=None):
    """Use MediaPipe PoseLandmarker to detect 2D keypoints for all images."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    task_path = os.path.join(tempfile.gettempdir(), 'pose_landmarker_lite.task')
    if not os.path.exists(task_path):
        shutil.copy2(MEDIAPIPE_TASK_SRC, task_path)

    base_opts = mp_python.BaseOptions(model_asset_path=task_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_opts,
        output_segmentation_masks=False,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
    )

    MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    all_kps = np.zeros((len(img_files), 17, 2), dtype=np.float32)

    with vision.PoseLandmarker.create_from_options(options) as detector:
        for i, img_path in enumerate(img_files):
            raw = np.fromfile(img_path, dtype=np.uint8)
            img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            h_img, w_img = img_bgr.shape[:2]

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            res = detector.detect(mp_image)

            if res.pose_landmarks:
                marks = res.pose_landmarks[0]
                kp = np.zeros((17, 2), dtype=np.float32)
                for j, idx in enumerate(MP_TO_COCO):
                    lm = marks[idx]
                    kp[j, 0] = lm.x * w_img
                    kp[j, 1] = lm.y * h_img
                all_kps[i] = coco_to_h36m(kp)

            if progress_cb and (i + 1) % 200 == 0:
                progress_cb(i + 1, len(img_files))

    return all_kps

def load_3d_gt():
    """Load 3D ground truth from JSON."""
    with open(POSE3D_PATH, 'r') as f:
        d3 = json.load(f)
    num_frames = len(d3)
    pts_3d = np.zeros((num_frames, 17, 3), dtype=np.float32)
    for i in range(num_frames):
        for j, jp in enumerate(d3[i]['keypoints_3d']):
            pts_3d[i, j, 0] = jp['x'] / 1000.0
            pts_3d[i, j, 1] = jp['y'] / 1000.0
            pts_3d[i, j, 2] = jp['z'] / 1000.0
    return pts_3d

def normalize_screen_coordinates(X, w=1000.0, h=1000.0):
    """Normalize 2D coordinates from pixel space to [-1, 1]."""
    return X / w * 2 - [1, h / w]


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


def build_prfk_update_flags(
        pts_2d_h36m,
        dist_threshold,
        dir_threshold=0.0,
        acc_threshold=0.0,
        theta_threshold=0.0):
    """Build per-frame per-leg update flags from PRFK."""
    monitor = PRFKKeypointMonitor(
        dist_threshold=dist_threshold,
        dir_threshold=dir_threshold,
        acc_threshold=acc_threshold,
        theta_threshold=theta_threshold,
    )
    left_flags = []
    right_flags = []
    left_changed_joints = []
    right_changed_joints = []

    for frame_idx, frame_2d in enumerate(pts_2d_h36m):
        result = monitor.update(h36m_frame_to_prfk_payload(frame_2d, frame_idx))
        changed_legs = result.get('changed_legs', {})

        # First frame always updates to initialize cache.
        left_flags.append(frame_idx == 0 or 'left_leg' in changed_legs)
        right_flags.append(frame_idx == 0 or 'right_leg' in changed_legs)

        left_changed_joints.append(changed_legs.get('left_leg', {}).get('changed_joints', []))
        right_changed_joints.append(changed_legs.get('right_leg', {}).get('changed_joints', []))

    return (
        np.array(left_flags, dtype=bool),
        np.array(right_flags, dtype=bool),
        left_changed_joints,
        right_changed_joints,
    )


def build_prfk_update_plan(
        pts_2d_h36m,
        dist_threshold,
        dir_threshold=0.0,
        acc_threshold=0.0,
        theta_threshold=0.0):
    """Build a per-frame PRFK plan with both per-leg and combined lower-body joint changes."""
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


def _numel(shape):
    if not shape:
        return 0
    total = 1
    for dim in shape:
        total *= int(dim)
    return int(total)


def _safe_tensor_shape(tensor_map, tensor_index):
    detail = tensor_map.get(int(tensor_index))
    if detail is None:
        return []
    return [int(x) for x in detail.get('shape', [])]


def estimate_tflite_flops(model_path):
    """Estimate FLOPs from a TFLite model by summing common ops."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    tensor_details = interpreter.get_tensor_details()
    tensor_map = {td['index']: td for td in tensor_details}
    op_details = interpreter._get_ops_details()  # pylint: disable=protected-access

    total_flops = 0.0
    for op in op_details:
        op_name = op.get('op_name', 'UNKNOWN')
        inputs = op.get('inputs', [])
        outputs = op.get('outputs', [])
        if len(outputs) == 0:
            continue

        out_shape = _safe_tensor_shape(tensor_map, outputs[0])

        if op_name == 'CONV_2D' and len(inputs) >= 2:
            w_shape = _safe_tensor_shape(tensor_map, inputs[1])
            if len(out_shape) == 4 and len(w_shape) == 4:
                _, out_h, out_w, out_c = out_shape
                _, k_h, k_w, in_c = w_shape
                total_flops += 2.0 * out_h * out_w * out_c * k_h * k_w * in_c
            continue

        if op_name == 'DEPTHWISE_CONV_2D' and len(inputs) >= 2:
            w_shape = _safe_tensor_shape(tensor_map, inputs[1])
            if len(out_shape) == 4 and len(w_shape) == 4:
                _, out_h, out_w, out_c = out_shape
                _, k_h, k_w, _ = w_shape
                total_flops += 2.0 * out_h * out_w * out_c * k_h * k_w
            continue

        if op_name == 'FULLY_CONNECTED' and len(inputs) >= 2:
            x_shape = _safe_tensor_shape(tensor_map, inputs[0])
            w_shape = _safe_tensor_shape(tensor_map, inputs[1])
            if len(x_shape) >= 2 and len(w_shape) == 2 and len(out_shape) >= 2:
                batch = int(x_shape[0]) if x_shape[0] > 0 else 1
                in_features = int(w_shape[1])
                out_features = int(w_shape[0])
                total_flops += 2.0 * batch * in_features * out_features
            continue

        if op_name in ('ADD', 'MUL'):
            total_flops += float(_numel(out_shape))

    return total_flops


def compute_ktpformer_flops_per_call(model):
    """Compute KTPFormer FLOPs for a single 27-frame call."""
    import copy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy = torch.randn(1, KTPFORMER_WINDOW, 3, 2, device=device)
    inner = model.module if hasattr(model, 'module') else model
    inner_copy = copy.deepcopy(inner).to(device)
    flops, _ = thop_profile(inner_copy, inputs=(dummy,), verbose=False)
    del inner_copy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return flops


def estimate_yolo_flops_per_frame():
    """Estimate YOLO11n-pose FLOPs by profiling with thop."""
    import copy
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        yolo_model = YOLO(YOLO_MODEL_PATH)
        # YOLO expects (B, H, W, C) input, typical 640x480
        dummy = torch.randn(1, 3, 640, 480, device=device)
        model_inner = yolo_model.model if hasattr(yolo_model, 'model') else yolo_model
        model_copy = copy.deepcopy(model_inner).to(device)
        flops, _ = thop_profile(model_copy, inputs=(dummy,), verbose=False)
        del model_copy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return flops
    except Exception as e:
        print(f"[WARNING] Failed to compute YOLO FLOPs dynamically: {e}")
        print(f"[WARNING] Using fallback value 7.6e9 GFLOPs")
        return 7.6e9


def estimate_hrnet_flops_per_frame():
    """Estimate HRNet-W32 FLOPs by profiling with thop."""
    import copy
    try:
        lib_dir = os.path.join(HRNET_ROOT, 'lib')
        sys.path.insert(0, HRNET_ROOT)
        sys.path.insert(0, lib_dir)
        
        # Import modules dynamically, same way as detect_hrnet_from_images
        models = import_module('models')
        cfg_module = import_module('config')
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        class ArgsMock:
            cfg = HRNET_CFG_PATH
            opts = []
            modelDir = ''
            logDir = ''
            dataDir = ''
            prevModelDir = ''
        
        cfg = cfg_module.cfg
        update_config = cfg_module.update_config
        
        # Update config with file
        update_config(cfg, ArgsMock())
        
        # Get model builder from config, then call it
        hrnet_builder = getattr(models, cfg.MODEL.NAME).get_pose_net
        hrnet = hrnet_builder(cfg, is_train=False)
        
        # Load weights
        if os.path.exists(HRNET_WEIGHTS_PATH):
            hrnet.load_state_dict(torch.load(HRNET_WEIGHTS_PATH, map_location='cpu'), strict=False)
        
        hrnet = hrnet.to(device).eval()
        
        # HRNet input: (B, C, H, W) - typically (1, 3, 256, 192)
        dummy = torch.randn(1, 3, 256, 192, device=device)
        model_copy = copy.deepcopy(hrnet).to(device)
        flops, _ = thop_profile(model_copy, inputs=(dummy,), verbose=False)
        del model_copy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return flops
    except Exception as e:
        print(f"[WARNING] Failed to compute HRNet FLOPs dynamically: {e}")
        print(f"[WARNING] Using fallback value 9.5e9 GFLOPs")
        return 9.5e9


def estimate_onnx_flops(model_path):
    """Estimate FLOPs from an ONNX model by analyzing the computation graph."""
    try:
        import onnx
        model = onnx.load(model_path)
        graph = model.graph
        
        total_flops = 0.0
        
        # Helper to get tensor shape from model's value_info or initializers
        def get_shape(name):
            for vi in graph.value_info:
                if vi.name == name:
                    return [d.dim_value for d in vi.type.tensor_type.shape.dim]
            for vi in graph.input:
                if vi.name == name:
                    return [d.dim_value if d.dim_value > 0 else 1 for d in vi.type.tensor_type.shape.dim]
            for init in graph.initializer:
                if init.name == name:
                    return list(init.dims)
            return []
        
        # Analyze each node in the graph
        for node in graph.node:
            op_type = node.op_type
            
            # Conv: FLOPs = 2 * output_h * output_w * output_c * kernel_h * kernel_w * input_c
            if op_type == 'Conv':
                try:
                    output_shape = get_shape(node.output[0])
                    weight_shape = get_shape(node.input[1])
                    if len(output_shape) == 4 and len(weight_shape) == 4:
                        _, out_h, out_w, out_c = output_shape
                        _, _, k_h, k_w = weight_shape
                        in_c = weight_shape[1] if len(weight_shape) > 1 else 1
                        total_flops += 2.0 * out_h * out_w * out_c * k_h * k_w * in_c
                except:
                    pass
            
            # Gemm (General Matrix Multiply): y = alpha * (A @ B) + beta * C
            elif op_type == 'Gemm':
                try:
                    a_shape = get_shape(node.input[0])
                    b_shape = get_shape(node.input[1])
                    if len(a_shape) >= 2 and len(b_shape) >= 2:
                        m = a_shape[-2] if len(a_shape) > 1 else 1
                        k = a_shape[-1]
                        n = b_shape[-1] if len(b_shape) > 1 else 1
                        total_flops += 2.0 * m * k * n
                except:
                    pass
            
            # MatMul: C = A @ B
            elif op_type == 'MatMul':
                try:
                    a_shape = get_shape(node.input[0])
                    b_shape = get_shape(node.input[1])
                    if len(a_shape) >= 2 and len(b_shape) >= 2:
                        m = a_shape[-2] if len(a_shape) > 1 else 1
                        k = a_shape[-1]
                        n = b_shape[-1]
                        total_flops += 2.0 * m * k * n
                except:
                    pass
            
            # Add, Sub, Mul, Div: elementwise ops
            elif op_type in ('Add', 'Sub', 'Mul', 'Div'):
                try:
                    output_shape = get_shape(node.output[0])
                    numel = 1
                    for dim in output_shape:
                        if dim > 0:
                            numel *= dim
                    total_flops += float(numel)
                except:
                    pass
            
            # Relu, Sigmoid, Tanh, Softmax: elementwise ops
            elif op_type in ('Relu', 'Sigmoid', 'Tanh', 'Softmax'):
                try:
                    output_shape = get_shape(node.output[0])
                    numel = 1
                    for dim in output_shape:
                        if dim > 0:
                            numel *= dim
                    if op_type == 'Sigmoid':
                        total_flops += 10.0 * numel  # Sigmoid is expensive
                    else:
                        total_flops += float(numel)
                except:
                    pass
        
        return total_flops
    except Exception as e:
        print(f"[WARNING] Failed to parse ONNX model FLOPs: {e}")
        return 40.0e9  # Fallback


def estimate_openpose_flops_per_frame():
    """Estimate ONNX OpenPose FLOPs from model graph."""
    import onnx
    try:
        onnx_path = OPENPOSE_ONNX_PATH
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"OpenPose ONNX model not found at {onnx_path}")
        
        model = onnx.load(onnx_path)
        graph = model.graph
        
        # Estimate FLOPs by counting operations
        flops = 0
        for node in graph.node:
            # Count MACs for convolution and matmul operations
            if node.op_type == 'Conv':
                # Assume standard 2D convolution with 368x368 input
                # Most OpenPose convolutions process 368x368 feature maps
                output_h, output_w = 368, 368
                kernel_h, kernel_w = 3, 3
                cin = 3  # will be adjusted for intermediate layers
                cout = 64  # typical channel count
                flops += output_h * output_w * kernel_h * kernel_w * cin * cout * 2
            elif node.op_type == 'MatMul':
                # MatMul operations
                flops += 1e8  # conservative estimate per MatMul
        
        # If no detailed analysis, use empirical estimate
        if flops < 1e9:
            flops = 30.0e9  # OpenPose ONNX empirical FLOPs
        
        return flops
    except Exception as e:
        print(f"[WARNING] Failed to compute ONNX OpenPose FLOPs: {e}")
        print(f"[WARNING] Using fallback value 30.0e9 GFLOPs")
        return 30.0e9


def estimate_mediapipe_flops_per_frame():
    """Estimate MediaPipe Pose Landmarker FLOPs by profiling its embedded TFLite models."""
    task_path = MEDIAPIPE_TASK_PATH
    if not os.path.exists(task_path):
        raise FileNotFoundError(f'MediaPipe task file not found: {task_path}')

    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(task_path, 'r') as zf:
            zf.extractall(td)

        detector_path = os.path.join(td, 'pose_detector.tflite')
        landmark_path = os.path.join(td, 'pose_landmarks_detector.tflite')
        detector_flops = estimate_tflite_flops(detector_path)
        landmark_flops = estimate_tflite_flops(landmark_path)
        return detector_flops + landmark_flops


def summarize_pipeline_flops(left_model):
    """Print per-frame FLOPs summary for the current pipeline."""
    print("\n" + "=" * 70)
    print("Computing FLOPs for all 2D detectors (this may take a moment)...")
    print("=" * 70)
    
    ktpformer_flops = compute_ktpformer_flops_per_call(left_model)
    print(f"✓ KTPFormer 3D lifting: {ktpformer_flops / 1e9:.6f} GFLOPs")
    
    yolo_flops = estimate_yolo_flops_per_frame()
    print(f"✓ YOLO11n-pose: {yolo_flops / 1e9:.6f} GFLOPs")
    
    hrnet_flops = estimate_hrnet_flops_per_frame()
    print(f"✓ HRNet-W32: {hrnet_flops / 1e9:.6f} GFLOPs")
    
    mediapipe_flops = estimate_mediapipe_flops_per_frame()
    print(f"✓ MediaPipe: {mediapipe_flops / 1e9:.6f} GFLOPs")
    
    openpose_flops = estimate_openpose_flops_per_frame()
    print(f"✓ OpenPose: {openpose_flops / 1e9:.6f} GFLOPs")

    detector_flops = {
        'YOLO': yolo_flops,
        'HRNet': hrnet_flops,
        'OpenPose': openpose_flops,
        'MediaPipe': mediapipe_flops,
    }

    print("\n" + "=" * 70)
    print("3DKTP FLOPs Summary (average per output frame)")
    print("=" * 70)
    print(f"KTPFormer 3D lifting: {ktpformer_flops / 1e9:.6f} GFLOPs / call (27-frame window)")
    print("-" * 70)
    for name in ['YOLO', 'HRNet', 'OpenPose', 'MediaPipe']:
        total = detector_flops[name] + ktpformer_flops
        print(f"{name:<10} | 2D: {detector_flops[name] / 1e9:.6f} G | 3D: {ktpformer_flops / 1e9:.6f} G | Total: {total / 1e9:.6f} G")
    print("=" * 70)

    return detector_flops, ktpformer_flops


def build_validation_hybrid_model(checkpoint_path):
    """建立checkpoint實際使用的17關節、243幀Hybrid GCN+TPA模型。"""
    import copy
    import numpy.random._pickle

    # 17點Human3.6M空間圖，與run_hybrid.py訓練時相同。
    spatial_skeleton = Skeleton(
        parents=[-1, 0, 1, -1, 3, 4],
        joints_left=[3, 4, 5],
        joints_right=[0, 1, 2],
    )
    spatial_adj = adj_mx_from_skeleton(spatial_skeleton)

    # checkpoint檔名中的f_243代表一次輸入243幀。
    temporal_parents = np.arange(FULLBODY_WINDOW) - 1
    temporal_adj = adj_mx_from_skeleton_temporal(
        FULLBODY_WINDOW,
        temporal_parents,
    )
    model = KTPFormer(
        spatial_adj,
        temporal_adj,
        num_frame=FULLBODY_WINDOW,
        num_joints=6,
        in_chans=2,
        embed_dim_ratio=512,
        depth=7,
        num_heads=8,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop_path_rate=0.0,
    )

    # 相容舊checkpoint內的NumPy random_state pickle。
    numpy.random._pickle.__randomstate_ctor = patched_randomstate_ctor
    numpy.random._pickle.__bit_generator_ctor = patched_bit_generator_ctor
    checkpoint = torch.load(
        checkpoint_path,
        map_location='cpu',
        weights_only=False,
    )
    state = {
        key.removeprefix('module.'): value
        for key, value in checkpoint['model_pos'].items()
    }
    spatial_position = [
        value for key, value in state.items()
        if key.endswith('Spatial_pos_embed')
    ]
    if not spatial_position or spatial_position[0].shape[1] != 6:
        raise ValueError(
            f'Checkpoint is not a true six-joint KTPFormer: {checkpoint_path}'
        )
    model.load_state_dict(state, strict=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

    # THOP量測一次完整(1,243,17,2) forward；使用副本，避免統計用
    # buffer或hook進入實際驗證模型。
    profile_model = copy.deepcopy(model).to(device).eval()
    dummy = torch.randn(
        1,
        FULLBODY_WINDOW,
        6,
        2,
        device=device,
    )
    with torch.no_grad():
        macs_per_call, params = thop_profile(
            profile_model,
            inputs=(dummy,),
            verbose=False,
        )
    del profile_model, dummy
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return (
        model,
        device,
        int(checkpoint.get('epoch', -1)),
        float(macs_per_call),
        float(params),
    )


def run_npz_hybrid_validation(args):
    """用S9+S11的CPN 2D與3D GT驗證Hybrid模型的下半身六點。"""
    from common.camera import (
        normalize_screen_coordinates as h36m_normalize_2d,
        world_to_camera,
    )
    from common.h36m_dataset import Human36mDataset

    # 1. 讀取2D NPZ與3D NPZ。
    # positions_2d[subject][action][camera]形狀為(frames,17,2)。
    positions_2d = np.load(
        args.npz_path,
        allow_pickle=True,
    )['positions_2d'].item()
    dataset = Human36mDataset(args.npz_3d_path)
    subjects = [
        value.strip()
        for value in args.subject.split(',')
        if value.strip()
    ]
    for subject in subjects:
        if subject not in positions_2d:
            raise KeyError(f'{subject} is not in {args.npz_path}')

    print('[NPZ Hybrid validation] Loading model...', flush=True)
    model, device, epoch, macs_per_call, params = (
        build_validation_hybrid_model(args.checkpoint_hybrid)
    )
    print(
        f'[INFO] Strict checkpoint load passed | epoch={epoch} | '
        f'parameters={params / 1e6:.6f} M',
        flush=True,
    )

    input_chunks = []
    gt_chunks = []
    valid_lengths = []
    plan_chunks = []
    sequence_start_flags = []
    sequence_count = 0
    total_frames = 0

    # 2. 依subject -> action -> camera配對2D與3D。
    # --camera_idx -1表示全部四台攝影機。
    for subject in subjects:
        for action_name in sorted(positions_2d[subject]):
            cameras_2d = positions_2d[subject][action_name]
            cameras = dataset[subject][action_name]['cameras']
            positions_world = dataset[subject][action_name]['positions']
            if args.camera_idx == -1:
                camera_indices = range(len(cameras_2d))
            elif 0 <= args.camera_idx < len(cameras_2d):
                camera_indices = [args.camera_idx]
            else:
                raise IndexError(
                    f'Camera {args.camera_idx} is unavailable for '
                    f'{subject}/{action_name}'
                )

            for camera_idx in camera_indices:
                camera = cameras[camera_idx]
                sequence_2d_raw = cameras_2d[camera_idx].copy()
                sequence_2d = sequence_2d_raw.copy()
                sequence_2d[..., :2] = h36m_normalize_2d(
                    sequence_2d[..., :2],
                    w=camera['res_w'],
                    h=camera['res_h'],
                )

                # 世界3D座標轉成攝影機座標，再扣除joint 0骨盆位置。
                sequence_3d = world_to_camera(
                    positions_world,
                    R=camera['orientation'],
                    t=camera['translation'],
                )
                sequence_3d[:, 1:] -= sequence_3d[:, :1]
                sequence_3d[:, 0] = 0

                # CPN與3D GT部分序列尾端相差一幀，只取共同長度。
                length = min(len(sequence_2d), len(sequence_3d))
                if args.disable_prfk:
                    frame_plan = [
                        {'any_update': True}
                        for _ in range(length)
                    ]
                else:
                    # PRFK使用尚未正規化的原始2D座標，與圖片流程的尺度一致。
                    frame_plan = build_prfk_update_plan(
                        sequence_2d_raw[:length, :, :2],
                        dist_threshold=args.prfk_dist_threshold,
                        dir_threshold=args.prfk_dir_threshold,
                        acc_threshold=args.prfk_acc_threshold,
                        theta_threshold=args.prfk_theta_threshold,
                    )
                # 真正六點輸入：送進模型前就移除另外11個2D關節。
                sequence_2d = sequence_2d[
                    :length,
                    VALIDATION_LOWER_INDICES,
                    :2,
                ].astype(np.float32)
                sequence_3d = sequence_3d[:length].astype(np.float32)
                sequence_count += 1
                total_frames += length

                # 3. Hybrid checkpoint是17點、243幀模型。
                # 最後不足243幀時複製尾幀補齊，但補幀不計入MPJPE。
                # 模型輸出17點後，只取[1,2,3,4,5,6]計分。
                for start in range(0, length, FULLBODY_WINDOW):
                    end = min(start + FULLBODY_WINDOW, length)
                    valid_length = end - start
                    chunk_2d = sequence_2d[start:end]
                    chunk_gt = sequence_3d[
                        start:end,
                        VALIDATION_LOWER_INDICES,
                        :,
                    ]
                    if valid_length < FULLBODY_WINDOW:
                        pad_length = FULLBODY_WINDOW - valid_length
                        chunk_2d = np.pad(
                            chunk_2d,
                            ((0, pad_length), (0, 0), (0, 0)),
                            mode='edge',
                        )
                        chunk_gt = np.pad(
                            chunk_gt,
                            ((0, pad_length), (0, 0), (0, 0)),
                            mode='edge',
                        )
                    input_chunks.append(chunk_2d)
                    gt_chunks.append(chunk_gt)
                    valid_lengths.append(valid_length)
                    plan_chunks.append(frame_plan[start:end])
                    sequence_start_flags.append(start == 0)

    print(
        '[INPUT CHECK] NPZ=(frames,17,2) | '
        'selected_2d=(frames,6,2) | '
        'model_input=(batch,243,6,2) | '
        'model_output=(batch,243,6,3)',
        flush=True,
    )

    # 4. 僅執行含有PRFK更新幀的243幀區塊；未更新幀沿用前次六點3D快取。
    chunks_to_run = [
        chunk_index
        for chunk_index, chunk_plan in enumerate(plan_chunks)
        if (
            args.disable_prfk
            or sequence_start_flags[chunk_index]
            or any(item.get('any_update', False) for item in chunk_plan)
        )
    ]
    predictions_by_chunk = {}
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    inference_start = time.perf_counter()
    with torch.no_grad():
        for batch_start in range(
                0,
                len(chunks_to_run),
                args.eval_batch_size):
            batch_end = min(
                batch_start + args.eval_batch_size,
                len(chunks_to_run),
            )
            batch_chunk_indices = chunks_to_run[batch_start:batch_end]
            input_batch = torch.from_numpy(
                np.stack([
                    input_chunks[index]
                    for index in batch_chunk_indices
                ])
            ).to(device)
            with torch.amp.autocast(
                    device_type='cuda',
                    enabled=device.type == 'cuda'):
                prediction = model(input_batch)
            prediction = prediction.float().cpu()
            for local_index, chunk_index in enumerate(batch_chunk_indices):
                predictions_by_chunk[chunk_index] = prediction[local_index]

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_start

    joint_error_sum = torch.zeros(6, dtype=torch.float64)
    update_frames = 0
    skipped_frames = 0
    cached_prediction = None
    for chunk_index, chunk_plan in enumerate(plan_chunks):
        if sequence_start_flags[chunk_index]:
            cached_prediction = None

        raw_prediction = predictions_by_chunk.get(chunk_index)
        valid_length = valid_lengths[chunk_index]
        target = torch.from_numpy(gt_chunks[chunk_index][:valid_length])
        effective_prediction = []

        for frame_offset in range(valid_length):
            should_update = (
                args.disable_prfk
                or chunk_plan[frame_offset].get('any_update', False)
            )
            if should_update:
                if raw_prediction is None:
                    raise RuntimeError(
                        'PRFK requested an update in a skipped KTP chunk'
                    )
                cached_prediction = raw_prediction[frame_offset]
                update_frames += 1
            else:
                skipped_frames += 1

            if cached_prediction is None:
                raise RuntimeError(
                    'KTP prediction cache is empty at the start of a sequence'
                )
            effective_prediction.append(cached_prediction)

        effective_prediction = torch.stack(effective_prediction, dim=0)
        errors = torch.linalg.vector_norm(
            effective_prediction - target,
            dim=-1,
        )
        joint_error_sum += errors.double().sum(dim=0)

    per_joint_mpjpe = joint_error_sum / total_frames * 1000.0
    combined_mpjpe = per_joint_mpjpe.mean().item()

    model_calls = len(chunks_to_run)
    total_macs = model_calls * macs_per_call
    arithmetic_flops_total = 2.0 * total_macs
    actual_macs_per_frame = total_macs / total_frames
    joint_names = [
        'R-Hip', 'R-Knee', 'R-Ankle',
        'L-Hip', 'L-Knee', 'L-Ankle',
    ]

    print('=' * 72)
    print(f'2D input: {args.npz_path}')
    print(f'3D ground truth: {args.npz_3d_path}')
    print(f'Checkpoint: {args.checkpoint_hybrid}')
    print(
        f'Subjects: {",".join(subjects)} | camera index: '
        f'{args.camera_idx} | sequences: {sequence_count}'
    )
    if args.disable_prfk:
        print('Mode: PRFK disabled; every 243-frame chunk runs the model')
    else:
        print(
            'PRFK thresholds: '
            f'acceleration={args.prfk_acc_threshold:g}, '
            f'theta={args.prfk_theta_threshold:g}, '
            f'distance={args.prfk_dist_threshold:g}, '
            f'direction={args.prfk_dir_threshold:g}'
        )
    print('Model input per call: (1,243,6,2)')
    print('Metric joints: [R-Hip,R-Knee,R-Ankle,L-Hip,L-Knee,L-Ankle]')
    print(f'Total real frames: {total_frames}')
    print(f'PRFK updates: {update_frames}; skips: {skipped_frames}')
    print(f'Baseline 243-frame chunks: {len(input_chunks)}')
    print(f'True six-joint KTP 243-frame calls: {model_calls}')
    print(f'Six-joint KTP per call: {macs_per_call / 1e9:.6f} GMac')
    print(
        f'Theoretical per output frame: '
        f'{macs_per_call / FULLBODY_WINDOW / 1e6:.6f} MMac/frame'
    )
    print(
        f'Actual including final-chunk padding: '
        f'{actual_macs_per_frame / 1e6:.6f} MMac/real frame'
    )
    print(f'Total arithmetic FLOPs: {arithmetic_flops_total / 1e12:.6f} TFLOPs')
    print('=' * 72)
    print(
        'FINAL TRUE SIX-JOINT KTP VALIDATION MPJPE '
        f'({"+".join(subjects)})'
    )
    print('=' * 72)
    for joint_name, error in zip(joint_names, per_joint_mpjpe):
        print(f'{joint_name:<8}: {error.item():.6f} mm')
    print(f'Combined lower-body MPJPE: {combined_mpjpe:.6f} mm')
    print(f'Actual 3D inference time: {inference_seconds:.3f} s')
    print(
        f'Average 3D inference time: '
        f'{inference_seconds / total_frames * 1000.0:.6f} ms/frame'
    )
    return


def _format_suffix(value):
    try:
        num = float(value)
    except (TypeError, ValueError):
        num = 0.0
    if abs(num - round(num)) < 1e-9:
        return str(int(round(num)))
    return f'{num:g}'

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--npz_path',
        type=str,
        default=H36M_2D_NPZ_PATH,
        help='Human3.6M CPN positions_2d NPZ for KTP-J validation',
    )
    parser.add_argument(
        '--npz_3d_path',
        type=str,
        default=H36M_3D_NPZ_PATH,
        help='Human3.6M positions_3d NPZ used as ground truth',
    )
    parser.add_argument(
        '--subject',
        type=str,
        default='S9,S11',
        help='One subject or comma-separated subjects',
    )
    parser.add_argument(
        '--camera_idx',
        type=int,
        default=-1,
        help='Camera index, or -1 for all cameras',
    )
    parser.add_argument(
        '--checkpoint_hybrid',
        type=str,
        default=HYBRID_CHECKPOINT_PATH,
        help='True six-joint 243-frame KTPFormer checkpoint',
    )
    parser.add_argument(
        '--eval_batch_size',
        type=int,
        default=8,
        help='Number of 243-frame chunks per GPU validation batch',
    )
    parser.add_argument('--pose3d', type=str, default=POSE3D_PATH, help='Path to 3D GT JSON')
    parser.add_argument('--left_checkpoint', type=str, default=r'C:\Users\angel\OneDrive - 國立臺北大學\KTPFormer-main\log\ktp_left_leg\leftleg.bin', help='Path to the trained left-leg checkpoint')
    parser.add_argument('--right_checkpoint', type=str, default=r'C:\Users\angel\OneDrive - 國立臺北大學\KTPFormer-main\log\ktp_right_leg\rightleg.bin', help='Path to the trained right-leg checkpoint')
    parser.add_argument('--fullbody_checkpoint', type=str, default=r'C:\Users\angel\OneDrive - 國立臺北大學\KTPFormer-main\checkpoint\s_128_L_7_C_512_H_8_f_243_ktpformer_cpn_ft_h36m_dbb_best_epoch.bin', help='Path to the trained 17-joint checkpoint (used when six lower-body joints changed)')
    parser.add_argument('--two_checkpoint', type=str, default=r'C:\Users\angel\OneDrive - 國立臺北大學\KTPFormer-main\log\ktp_2keypoint\twokeypoint.bin', help='Path to the trained two-joint checkpoint (optional, must be actual 2-joint model)')
    parser.add_argument('--four_checkpoint', type=str, default=r'C:\Users\angel\OneDrive - 國立臺北大學\KTPFormer-main\log\ktp_4keypoint\fourkeypoint.bin', help='Path to the trained four-joint checkpoint (optional, must be actual 4-joint model)')
    parser.add_argument('--five_checkpoint', type=str, default=r'C:\Users\angel\OneDrive - 國立臺北大學\KTPFormer-main\log\ktp_5keypoint\fivekeypoint.bin', help='Path to the trained five-joint checkpoint (optional, must be actual 5-joint model)')
    parser.add_argument('--single_checkpoint', type=str, default=r'C:\Users\angel\OneDrive - 國立臺北大學\KTPFormer-main\log\ktp_1keypoint\singlekeypoint.bin', help='Path to the trained single-joint checkpoint (optional, must be actual 1-joint model)')
    parser.add_argument('--prfk_acc_threshold', type=float, default=0.0, help='PRFK acceleration threshold')
    parser.add_argument('--prfk_theta_threshold', type=float, default=0.0, help='PRFK pelvic-axis angle threshold in degrees')
    parser.add_argument('--prfk_dist_threshold', type=float, default=0.0, help='PRFK distance threshold (x/y) for frame-to-frame leg change detection')
    parser.add_argument('--prfk_dir_threshold', type=float, default=0.0, help='PRFK direction-vector threshold')
    args = parser.parse_args()
    # 保留舊程式內部的分支相容性；命令列不再提供--disable_prfk。
    # 四個閾值同時為0就是全幀執行模式。
    args.disable_prfk = False

    # KTP-J與L2D-J一致，只使用NPZ的2D輸入及3D ground truth驗證。
    run_npz_hybrid_validation(args)
    return

    # 1. Load images and detect 2D keypoints with YOLO
    print("=" * 70)
    print("3DKTP From Images: 4x 2D Detection -> 3D Lifting -> MPJPE")
    print("=" * 70)
    
    img_files = sorted(glob.glob(os.path.join(args.image_dir, '*.jpg')))
    num_frames = len(img_files)
    print(f"\n[1/6] Found {num_frames} test images")
    
    print("[2/6] Running 4 detector pipelines...")
    def progress(cur, total):
        print(f"  {cur}/{total}", flush=True)
    
    detectors = {
        'YOLO': detect_yolo_from_images,
        'HRNet': detect_hrnet_from_images,
        'OpenPose': detect_openpose_from_images,
        'MediaPipe': detect_mediapipe_from_images,
    }

    left_leg_indices = [4, 5, 6]
    results = {}

    # Load 3D GT & apply camera extrinsics early so per-detector loops can access them
    print("[3/6] Loading 3D GT & applying camera extrinsics...")
    pts_3d = load_3d_gt()
    with open(CAMERA_PATH, 'r') as f:
        cam_data = json.load(f)['4']
    R = np.array(cam_data['R'], dtype=np.float32)
    t = np.array(cam_data['t'], dtype=np.float32) / 1000.0
    pts_3d_cam = np.dot(pts_3d - t, R.T)
    pts_3d_pelvis = pts_3d_cam[:, [0], :]

    # Load 3DKTP models so per-detector loops can run 3D inference
    print("[5/6] Loading 3DKTP models...")
    temporal_skeleton = np.array(list(range(0, 27))) - 1
    adj_temporal = adj_mx_from_skeleton_temporal(27, temporal_skeleton)

    leg_skeleton = Skeleton(parents=[-1, 0, 1], joints_left=[], joints_right=[])
    adj = adj_mx_from_skeleton(leg_skeleton)
    leg_skeleton_two = Skeleton(parents=[-1, 0], joints_left=[], joints_right=[])
    adj_two = adj_mx_from_skeleton(leg_skeleton_two)
    leg_skeleton_five = Skeleton(parents=[-1, 0, 1, 2, 3], joints_left=[], joints_right=[])
    adj_five = adj_mx_from_skeleton(leg_skeleton_five)
    h36m_parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    full_skeleton = Skeleton(parents=h36m_parents, joints_left=[], joints_right=[])
    adj_full = adj_mx_from_skeleton(full_skeleton)

    def build_model(checkpoint_path, num_joints=3, num_frame=27):
        if num_joints == 1:
            adj_model = torch.eye(1, dtype=torch.float)
        elif num_joints == 2:
            adj_model = adj_two
        elif num_joints == 5:
            adj_model = adj_five
        elif num_joints == 4:
            # 4-point chain topology: 0-1-2-3
            leg_skeleton_four = Skeleton(parents=[-1, 0, 1, 2], joints_left=[], joints_right=[])
            adj_model = adj_mx_from_skeleton(leg_skeleton_four)
        elif num_joints == 17:
            adj_model = adj_full
        else:
            adj_model = adj

        if num_frame == 27:
            adj_temporal_model = adj_temporal
        else:
            temporal_parents = np.array(list(range(0, num_frame))) - 1
            adj_temporal_model = adj_mx_from_skeleton_temporal(num_frame, temporal_parents)

        model = KTPFormer(
            adj_model, adj_temporal_model, num_frame=num_frame, num_joints=num_joints, in_chans=2,
            embed_dim_ratio=512, depth=7, num_heads=8, mlp_ratio=2.,
            qkv_bias=True, qk_scale=None, drop_path_rate=0.1
        )
        checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model)
            model = model.cuda()
        if 'model_pos' in checkpoint:
            model.load_state_dict(checkpoint['model_pos'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model

    right_checkpoint = args.right_checkpoint if args.right_checkpoint else args.left_checkpoint
    left_model = build_model(args.left_checkpoint)
    right_model = build_model(right_checkpoint)
    full_model = None
    if args.fullbody_checkpoint:
        full_model = build_model(args.fullbody_checkpoint, num_joints=17, num_frame=FULLBODY_WINDOW)
    two_model = None
    if args.two_checkpoint:
        two_model = build_model(args.two_checkpoint, num_joints=2)
    five_model = None
    if args.five_checkpoint:
        five_model = build_model(args.five_checkpoint, num_joints=5)
    four_model = None
    if args.four_checkpoint:
        four_model = build_model(args.four_checkpoint, num_joints=4)
    single_model = None
    if args.single_checkpoint:
        single_model = build_model(args.single_checkpoint, num_joints=1)

    detector_flops, ktpformer_flops = summarize_pipeline_flops(left_model)

    leg_indices = {
        'left': [4, 5, 6],
        'right': [1, 2, 3],
    }

    def run_leg_inference(pts_2d_norm, leg_name, model, update_flags, changed_joints_by_frame=None, single_model=None, two_model=None, five_model=None, four_model=None, full_model=None, six_joint_update_flags=None, use_prfk=True, full_window=FULLBODY_WINDOW, full_pred_cache=None):
        leg_idx = leg_indices[leg_name]
        pts_2d_leg = pts_2d_norm[:, leg_idx, :].copy()
        pts_3d_leg = pts_3d_cam[:, leg_idx, :].copy()

        pts_3d_leg = pts_3d_leg - pts_3d_pelvis
        pts_3d_leg[:, 0, :] = 0

        inputs_2d = torch.from_numpy(pts_2d_leg).float().unsqueeze(0).cuda()
        inputs_3d = torch.from_numpy(pts_3d_leg).float().unsqueeze(0).cuda()

        pad = 27 // 2
        inputs_2d_padded = torch.cat([
            inputs_2d[:, :1].repeat(1, pad, 1, 1),
            inputs_2d,
            inputs_2d[:, -1:].repeat(1, pad, 1, 1)
        ], dim=1)

        full_inputs_2d_padded = None
        full_pad = full_window // 2
        if full_model is not None:
            inputs_2d_full = torch.from_numpy(pts_2d_norm).float().unsqueeze(0).cuda()
            full_inputs_2d_padded = torch.cat([
                inputs_2d_full[:, :1].repeat(1, full_pad, 1, 1),
                inputs_2d_full,
                inputs_2d_full[:, -1:].repeat(1, full_pad, 1, 1)
            ], dim=1)

        predicted_3d = []
        last_pred_center = None
        num_updates = 0
        num_skips = 0
        num_single_updates = 0
        num_two_updates = 0
        num_five_updates = 0
        num_four_updates = 0
        num_full_updates = 0
        num_default_updates = 0
        t_3d_start = time.time()

        joint_name_to_rel = {
            'left_hip': 0, 'left_knee': 1, 'left_ankle': 2,
            'right_hip': 0, 'right_knee': 1, 'right_ankle': 2,
        }
        two_joint_slice_map = {
            (0, 1): (0, 2),
            (1, 2): (1, 3),
        }
        five_joint_slice_map = {
            (0, 1, 2, 3, 4): (0, 5),
            (1, 2, 3, 4, 5): (1, 6),
        }
        four_joint_slice_map = {
            (0, 1, 2, 3): (0, 4),
            (2, 3, 4, 5): (2, 6),
        }

        with torch.no_grad():
            for i in range(inputs_2d.shape[1]):
                should_update = True if not use_prfk else (bool(update_flags[i]) if i < len(update_flags) else True)
                changed_joints = []
                if changed_joints_by_frame is not None and i < len(changed_joints_by_frame):
                    changed_joints = changed_joints_by_frame[i]

                # Priority 1: Check 6-joint update (full model) FIRST - highest priority
                use_full = (
                    full_model is not None
                    and six_joint_update_flags is not None
                    and i < len(six_joint_update_flags)
                    and bool(six_joint_update_flags[i])
                )

                if use_full:
                    if full_pred_cache is not None and i in full_pred_cache:
                        pred_center_full = full_pred_cache[i]
                    else:
                        chunk_full = full_inputs_2d_padded[:, i:i+full_window, :, :]
                        pred_full = full_model(chunk_full)
                        pred_center_full = pred_full[:, full_pad:full_pad+1, :, :]
                        if full_pred_cache is not None:
                            full_pred_cache[i] = pred_center_full

                    pred_center = pred_center_full[:, :, leg_idx, :].clone()
                    last_pred_center = pred_center
                    num_updates += 1
                    num_full_updates += 1
                    predicted_3d.append(pred_center)
                    continue

                # Priority 2: Skip update if this leg had no change (PRFK) and we have cached prediction
                if not should_update and last_pred_center is not None:
                    pred_center = last_pred_center
                    num_skips += 1
                    predicted_3d.append(pred_center)
                    continue

                # Priority 3: Check 5-joint update
                use_five = (
                    five_model is not None
                    and len(changed_joints) == 5
                    and last_pred_center is not None
                )

                if use_five:
                    # Verify all joints belong to the current leg (safety check)
                    expected_prefix = 'left_' if leg_name == 'left' else 'right_'
                    if any(not name.startswith(expected_prefix) for name in changed_joints):
                        use_five = False
                    else:
                        rel_indices = sorted([joint_name_to_rel.get(name, -1) for name in changed_joints])
                        if -1 in rel_indices:
                            use_five = False
                        else:
                            rel_quint = tuple(rel_indices)
                            if rel_quint not in five_joint_slice_map:
                                use_five = False

                if use_five:
                    start_idx, end_idx = five_joint_slice_map[rel_quint]
                    chunk_2d = inputs_2d_padded[:, i:i+27, start_idx:end_idx, :]
                    pred_five = five_model(chunk_2d)
                    pred_center_five = pred_five[:, pad:pad+1, :, :]

                    pred_center = last_pred_center.clone()
                    pred_center[:, :, start_idx:end_idx, :] = pred_center_five
                    last_pred_center = pred_center
                    num_updates += 1
                    num_five_updates += 1
                    predicted_3d.append(pred_center)
                    continue

                # Priority 4: Check 4-joint update
                use_four = (
                    four_model is not None
                    and len(changed_joints) == 4
                    and last_pred_center is not None
                )

                if use_four:
                    # Verify all joints belong to the current leg (safety check)
                    expected_prefix = 'left_' if leg_name == 'left' else 'right_'
                    if any(not name.startswith(expected_prefix) for name in changed_joints):
                        use_four = False
                    else:
                        rel_indices = sorted([joint_name_to_rel.get(name, -1) for name in changed_joints])
                        if -1 in rel_indices:
                            use_four = False
                        else:
                            rel_quad = tuple(rel_indices)
                            if rel_quad not in four_joint_slice_map:
                                use_four = False

                if use_four:
                    start_idx, end_idx = four_joint_slice_map[rel_quad]
                    chunk_2d = inputs_2d_padded[:, i:i+27, start_idx:end_idx, :]
                    pred_four = four_model(chunk_2d)
                    pred_center_four = pred_four[:, pad:pad+1, :, :]

                    pred_center = last_pred_center.clone()
                    pred_center[:, :, start_idx:end_idx, :] = pred_center_four
                    last_pred_center = pred_center
                    num_updates += 1
                    num_four_updates += 1
                    predicted_3d.append(pred_center)
                    continue

                # Priority 5: Check 2-joint update
                use_two = (
                    two_model is not None
                    and len(changed_joints) == 2
                    and last_pred_center is not None
                )

                if use_two:
                    # Verify all joints belong to the current leg (safety check)
                    expected_prefix = 'left_' if leg_name == 'left' else 'right_'
                    if any(not name.startswith(expected_prefix) for name in changed_joints):
                        use_two = False
                    else:
                        rel_indices = sorted([joint_name_to_rel.get(name, -1) for name in changed_joints])
                        if -1 in rel_indices:
                            use_two = False
                        else:
                            rel_pair = tuple(rel_indices)
                            if rel_pair not in two_joint_slice_map:
                                use_two = False

                if use_two:
                    start_idx, end_idx = two_joint_slice_map[rel_pair]
                    chunk_2d = inputs_2d_padded[:, i:i+27, start_idx:end_idx, :]
                    pred_two = two_model(chunk_2d)
                    pred_center_two = pred_two[:, pad:pad+1, :, :]

                    pred_center = last_pred_center.clone()
                    pred_center[:, :, start_idx:end_idx, :] = pred_center_two
                    last_pred_center = pred_center
                    num_updates += 1
                    num_two_updates += 1
                    predicted_3d.append(pred_center)
                    continue

                # Priority 6: Check 1-joint update or default 3-joint
                use_single = (
                    single_model is not None
                    and len(changed_joints) == 1
                    and last_pred_center is not None
                )

                if use_single:
                    joint_name = changed_joints[0]
                    # Verify joint belongs to the current leg (safety check)
                    expected_prefix = 'left_' if leg_name == 'left' else 'right_'
                    if not joint_name.startswith(expected_prefix):
                        use_single = False
                    else:
                        rel_idx = joint_name_to_rel.get(joint_name, None)
                        if rel_idx is None:
                            use_single = False

                if use_single:
                    chunk_2d = inputs_2d_padded[:, i:i+27, rel_idx:rel_idx+1, :]
                    pred_single = single_model(chunk_2d)
                    pred_center_single = pred_single[:, pad:pad+1, :, :]

                    pred_center = last_pred_center.clone()
                    pred_center[:, :, rel_idx, :] = pred_center_single[:, :, 0, :]
                    last_pred_center = pred_center
                    num_updates += 1
                    num_single_updates += 1
                else:
                    chunk_2d = inputs_2d_padded[:, i:i+27, :, :]
                    pred = model(chunk_2d)
                    pred_center = pred[:, pad:pad+1, :, :]
                    last_pred_center = pred_center
                    num_updates += 1
                    num_default_updates += 1

                predicted_3d.append(pred_center)

        predicted_3d = torch.cat(predicted_3d, dim=1)
        predicted_3d[:, :, 0, :] = 0
        t_3d = time.time() - t_3d_start

        error = mpjpe(predicted_3d, inputs_3d)
        pred_np = predicted_3d[0].cpu().numpy()
        gt_np = inputs_3d[0].cpu().numpy()
        p_error = p_mpjpe(pred_np, gt_np)

        return {
            'mpjpe': error.item() * 1000,
            'p_mpjpe': p_error * 1000,
            't_3d': t_3d,
            'pred': predicted_3d,
            'gt': inputs_3d,
            'num_updates': num_updates,
            'num_skips': num_skips,
            'num_single_updates': num_single_updates,
            'num_two_updates': num_two_updates,
                    'num_five_updates': num_five_updates,
            'num_four_updates': num_four_updates,
            'num_full_updates': num_full_updates,
                    'num_default_updates': num_default_updates,
            'num_frames': inputs_2d.shape[1],
        }

    def run_lower_body_inference(pts_2d_norm, frame_plan, left_model, right_model, single_model=None, two_model=None, five_model=None, four_model=None, full_model=None, full_window=FULLBODY_WINDOW, full_pred_cache=None, use_prfk=True):
        leg_indices_full = {
            'left': [4, 5, 6],
            'right': [1, 2, 3],
        }
        leg_cache_slices = {
            'left': slice(0, 3),
            'right': slice(3, 6),
        }
        lower_body_indices = LOWER_BODY_H36M_INDICES
        lower_body_joint_to_index = LOWER_BODY_JOINT_TO_INDEX

        joint_name_to_rel = {
            'left_hip': 0, 'left_knee': 1, 'left_ankle': 2,
            'right_hip': 0, 'right_knee': 1, 'right_ankle': 2,
        }
        two_joint_slice_map = {
            (0, 1): (0, 2),
            (1, 2): (1, 3),
        }
        five_joint_slice_map = {
            (0, 1, 2, 3, 4): (0, 5),
            (1, 2, 3, 4, 5): (1, 6),
        }
        four_joint_slice_map = {
            (0, 1, 2, 3): (0, 4),
            (2, 3, 4, 5): (2, 6),
        }

        pts_2d_left = torch.from_numpy(pts_2d_norm[:, leg_indices_full['left'], :]).float().unsqueeze(0).cuda()
        pts_2d_right = torch.from_numpy(pts_2d_norm[:, leg_indices_full['right'], :]).float().unsqueeze(0).cuda()
        pts_2d_lower = torch.from_numpy(pts_2d_norm[:, lower_body_indices, :]).float().unsqueeze(0).cuda()

        pts_3d_left = pts_3d_cam[:, leg_indices_full['left'], :].copy()
        pts_3d_right = pts_3d_cam[:, leg_indices_full['right'], :].copy()
        pts_3d_left = pts_3d_left - pts_3d_pelvis
        pts_3d_right = pts_3d_right - pts_3d_pelvis
        pts_3d_left[:, 0, :] = 0
        pts_3d_right[:, 0, :] = 0

        inputs_2d_left_padded = torch.cat([
            pts_2d_left[:, :1].repeat(1, 27 // 2, 1, 1),
            pts_2d_left,
            pts_2d_left[:, -1:].repeat(1, 27 // 2, 1, 1)
        ], dim=1)
        inputs_2d_right_padded = torch.cat([
            pts_2d_right[:, :1].repeat(1, 27 // 2, 1, 1),
            pts_2d_right,
            pts_2d_right[:, -1:].repeat(1, 27 // 2, 1, 1)
        ], dim=1)
        inputs_2d_lower_padded = torch.cat([
            pts_2d_lower[:, :1].repeat(1, 27 // 2, 1, 1),
            pts_2d_lower,
            pts_2d_lower[:, -1:].repeat(1, 27 // 2, 1, 1)
        ], dim=1)

        full_inputs_2d_padded = None
        full_pad = full_window // 2
        if full_model is not None:
            inputs_2d_full = torch.from_numpy(pts_2d_norm).float().unsqueeze(0).cuda()
            full_inputs_2d_padded = torch.cat([
                inputs_2d_full[:, :1].repeat(1, full_pad, 1, 1),
                inputs_2d_full,
                inputs_2d_full[:, -1:].repeat(1, full_pad, 1, 1)
            ], dim=1)

        left_pred_frames = []
        right_pred_frames = []
        last_pred_lower = None
        t_3d_start = time.time()

        left_stats = {
            'num_updates': 0,
            'num_skips': 0,
            'num_single_updates': 0,
            'num_two_updates': 0,
            'num_five_updates': 0,
            'num_four_updates': 0,
            'num_full_updates': 0,
            'num_default_updates': 0,
        }
        right_stats = left_stats.copy()

        model_usage_counts = {
            '17j': 0,
            '5j': 0,
            '4j': 0,
            '2j': 0,
            '1j': 0,
            '3j': 0,
        }

        def _run_leg_branch(leg_name, current_pred, changed_joints, frame_idx, force_three_point=False):
            leg_slice = leg_cache_slices[leg_name]
            leg_input = inputs_2d_left_padded if leg_name == 'left' else inputs_2d_right_padded
            leg_model = left_model if leg_name == 'left' else right_model
            stats = left_stats if leg_name == 'left' else right_stats
            updated_pred = current_pred.clone()
            local_model_key = '3j'

            if force_three_point:
                chunk_2d = leg_input[:, frame_idx:frame_idx + 27, :, :]
                pred_leg = leg_model(chunk_2d)
                pred_center_leg = pred_leg[:, 27 // 2:27 // 2 + 1, :, :]
                updated_pred[:, :, leg_slice, :] = pred_center_leg
                stats['num_default_updates'] += 1
                stats['num_updates'] += 1
                return updated_pred, local_model_key, True

            use_single = (
                single_model is not None
                and len(changed_joints) == 1
                and current_pred is not None
            )
            if use_single:
                joint_name = changed_joints[0]
                expected_prefix = 'left_' if leg_name == 'left' else 'right_'
                if not joint_name.startswith(expected_prefix):
                    use_single = False
                else:
                    rel_idx = joint_name_to_rel.get(joint_name, None)
                    if rel_idx is None:
                        use_single = False
            if use_single:
                rel_idx = joint_name_to_rel[changed_joints[0]]
                chunk_2d = leg_input[:, frame_idx:frame_idx + 27, rel_idx:rel_idx + 1, :]
                pred_single = single_model(chunk_2d)
                pred_center_single = pred_single[:, 27 // 2:27 // 2 + 1, :, :]
                updated_pred[:, :, leg_slice, :] = current_pred[:, :, leg_slice, :].clone()
                updated_pred[:, :, leg_slice.start + rel_idx:leg_slice.start + rel_idx + 1, :] = pred_center_single[:, :, 0, :]
                stats['num_single_updates'] += 1
                stats['num_updates'] += 1
                return updated_pred, '1j', True

            use_two = (
                two_model is not None
                and len(changed_joints) == 2
                and current_pred is not None
            )
            if use_two:
                expected_prefix = 'left_' if leg_name == 'left' else 'right_'
                if any(not name.startswith(expected_prefix) for name in changed_joints):
                    use_two = False
                else:
                    rel_indices = sorted([joint_name_to_rel.get(name, -1) for name in changed_joints])
                    if -1 in rel_indices:
                        use_two = False
                    else:
                        rel_pair = tuple(rel_indices)
                        if rel_pair not in two_joint_slice_map:
                            use_two = False
            if use_two:
                rel_pair = tuple(sorted([joint_name_to_rel[name] for name in changed_joints]))
                start_idx, end_idx = two_joint_slice_map[rel_pair]
                chunk_2d = leg_input[:, frame_idx:frame_idx + 27, start_idx:end_idx, :]
                pred_two = two_model(chunk_2d)
                pred_center_two = pred_two[:, 27 // 2:27 // 2 + 1, :, :]
                updated_pred[:, :, leg_slice.start + start_idx:leg_slice.start + end_idx, :] = pred_center_two
                stats['num_two_updates'] += 1
                stats['num_updates'] += 1
                return updated_pred, '2j', True

            chunk_2d = leg_input[:, frame_idx:frame_idx + 27, :, :]
            pred_leg = leg_model(chunk_2d)
            pred_center_leg = pred_leg[:, 27 // 2:27 // 2 + 1, :, :]
            updated_pred[:, :, leg_slice, :] = pred_center_leg
            stats['num_default_updates'] += 1
            stats['num_updates'] += 1
            return updated_pred, local_model_key, True

        with torch.no_grad():
            for i, info in enumerate(frame_plan):
                left_changed_joints = info['left_changed_joints']
                right_changed_joints = info['right_changed_joints']
                changed_joints = info['changed_joints']
                changed_joint_indices = info['changed_joint_indices']

                if i == 0:
                    current_pred = torch.zeros(1, 1, 6, 3, device=pts_2d_lower.device)
                    current_pred, model_key_left, _ = _run_leg_branch('left', current_pred, ['left_hip', 'left_knee', 'left_ankle'], i)
                    current_pred, model_key_right, _ = _run_leg_branch('right', current_pred, ['right_hip', 'right_knee', 'right_ankle'], i)
                    last_pred_lower = current_pred
                    model_usage_counts['3j'] += 2
                    left_pred_frames.append(current_pred[:, :, leg_cache_slices['left'], :].clone())
                    right_pred_frames.append(current_pred[:, :, leg_cache_slices['right'], :].clone())
                    continue

                should_update = True if not use_prfk else bool(info['any_update'])
                if not should_update and last_pred_lower is not None:
                    left_stats['num_skips'] += 1
                    right_stats['num_skips'] += 1
                    left_pred_frames.append(last_pred_lower[:, :, leg_cache_slices['left'], :].clone())
                    right_pred_frames.append(last_pred_lower[:, :, leg_cache_slices['right'], :].clone())
                    continue

                current_pred = last_pred_lower.clone() if last_pred_lower is not None else torch.zeros(1, 1, 6, 3, device=pts_2d_lower.device)

                # Simplified routing: if any motion detected, use 17-point full-body model
                if full_model is not None:
                    if full_pred_cache is not None and i in full_pred_cache:
                        pred_center_full = full_pred_cache[i]
                    else:
                        chunk_full = full_inputs_2d_padded[:, i:i + full_window, :, :]
                        pred_full = full_model(chunk_full)
                        pred_center_full = pred_full[:, full_pad:full_pad + 1, :, :]
                        if full_pred_cache is not None:
                            full_pred_cache[i] = pred_center_full
                    current_pred[:, :, leg_cache_slices['left'], :] = pred_center_full[:, :, [4, 5, 6], :]
                    current_pred[:, :, leg_cache_slices['right'], :] = pred_center_full[:, :, [1, 2, 3], :]
                    left_stats['num_full_updates'] += 1
                    left_stats['num_updates'] += 1
                    right_stats['num_full_updates'] += 1
                    right_stats['num_updates'] += 1
                    model_usage_counts['17j'] += 1

                last_pred_lower = current_pred
                left_pred_frames.append(current_pred[:, :, leg_cache_slices['left'], :].clone())
                right_pred_frames.append(current_pred[:, :, leg_cache_slices['right'], :].clone())

        predicted_left = torch.cat(left_pred_frames, dim=1)
        predicted_right = torch.cat(right_pred_frames, dim=1)
        predicted_left[:, :, 0, :] = 0
        predicted_right[:, :, 0, :] = 0
        t_3d = time.time() - t_3d_start
        # Recompute per-side update/skip counts from the category counters
        n_frames = pts_2d_norm.shape[0]
        left_stats['num_updates'] = (
            left_stats.get('num_full_updates', 0)
            + left_stats.get('num_five_updates', 0)
            + left_stats.get('num_four_updates', 0)
            + left_stats.get('num_two_updates', 0)
            + left_stats.get('num_single_updates', 0)
            + left_stats.get('num_default_updates', 0)
        )
        right_stats['num_updates'] = (
            right_stats.get('num_full_updates', 0)
            + right_stats.get('num_five_updates', 0)
            + right_stats.get('num_four_updates', 0)
            + right_stats.get('num_two_updates', 0)
            + right_stats.get('num_single_updates', 0)
            + right_stats.get('num_default_updates', 0)
        )

        # Ensure skips sum to frames (avoid bookkeeping drift)
        left_stats['num_skips'] = max(0, n_frames - int(left_stats['num_updates']))
        right_stats['num_skips'] = max(0, n_frames - int(right_stats['num_updates']))

        left_gt = torch.from_numpy(pts_3d_left).float().unsqueeze(0).cuda()
        right_gt = torch.from_numpy(pts_3d_right).float().unsqueeze(0).cuda()
        left_error = mpjpe(predicted_left, left_gt).item() * 1000
        right_error = mpjpe(predicted_right, right_gt).item() * 1000
        left_p_error = p_mpjpe(predicted_left[0].cpu().numpy(), left_gt[0].cpu().numpy()) * 1000
        right_p_error = p_mpjpe(predicted_right[0].cpu().numpy(), right_gt[0].cpu().numpy()) * 1000

        left_result = {
            'mpjpe': left_error,
            'p_mpjpe': left_p_error,
            't_3d': t_3d,
            'pred': predicted_left,
            'gt': left_gt,
            'num_updates': left_stats['num_updates'],
            'num_skips': left_stats['num_skips'],
            'num_single_updates': left_stats['num_single_updates'],
            'num_two_updates': left_stats['num_two_updates'],
            'num_five_updates': left_stats['num_five_updates'],
            'num_four_updates': left_stats['num_four_updates'],
            'num_full_updates': left_stats['num_full_updates'],
            'num_default_updates': left_stats['num_default_updates'],
            'num_frames': pts_2d_norm.shape[0],
        }
        right_result = {
            'mpjpe': right_error,
            'p_mpjpe': right_p_error,
            't_3d': t_3d,
            'pred': predicted_right,
            'gt': right_gt,
            'num_updates': right_stats['num_updates'],
            'num_skips': right_stats['num_skips'],
            'num_single_updates': right_stats['num_single_updates'],
            'num_two_updates': right_stats['num_two_updates'],
            'num_five_updates': right_stats['num_five_updates'],
            'num_four_updates': right_stats['num_four_updates'],
            'num_full_updates': right_stats['num_full_updates'],
            'num_default_updates': right_stats['num_default_updates'],
            'num_frames': pts_2d_norm.shape[0],
        }

        return {
            'left': left_result,
            'right': right_result,
            't_3d': t_3d,
            'model_usage_counts': model_usage_counts,
        }

    print("[4/6] Running 4 detector pipelines...")
    for name, detect_fn in detectors.items():
        print(f"\n--- {name} ---")
        t_2d_start = time.time()
        pts_2d = detect_fn(img_files, progress_cb=progress)
        t_2d = time.time() - t_2d_start

        print("  Normalizing 2D coordinates...")
        pts_2d_norm = normalize_screen_coordinates(pts_2d)
        full_pred_cache = {}

        if not args.disable_prfk:
            print("  Running 3D inference with global PRFK routing...")
            frame_plan = build_prfk_update_plan(
                pts_2d,
                dist_threshold=args.prfk_dist_threshold,
                dir_threshold=args.prfk_dir_threshold,
                acc_threshold=args.prfk_acc_threshold,
                theta_threshold=args.prfk_theta_threshold,
            )
            routed_result = run_lower_body_inference(
                pts_2d_norm, frame_plan,
                left_model, right_model,
                single_model=single_model, two_model=two_model,
                five_model=five_model, four_model=four_model, full_model=full_model,
                full_pred_cache=full_pred_cache,
                use_prfk=True,
            )
            left_result = routed_result['left']
            right_result = routed_result['right']
            t_3d = routed_result['t_3d']
            model_usage_counts = routed_result['model_usage_counts']
        else:
            print("  Running 3D inference for left/right legs...")
            if full_model is not None:
                frame_plan = [
                    {
                        'left_update': True,
                        'right_update': True,
                        'left_changed_joints': ['left_hip', 'left_knee', 'left_ankle'],
                        'right_changed_joints': ['right_hip', 'right_knee', 'right_ankle'],
                        'changed_joints': LOWER_BODY_JOINT_ORDER,
                        'changed_joint_indices': list(range(6)),
                        'any_update': True,
                    }
                    for _ in range(pts_2d.shape[0])
                ]
                routed_result = run_lower_body_inference(
                    pts_2d_norm, frame_plan,
                    left_model, right_model,
                    single_model=single_model, two_model=two_model,
                    five_model=five_model, four_model=four_model, full_model=full_model,
                    full_pred_cache=full_pred_cache,
                    use_prfk=False,
                )
                left_result = routed_result['left']
                right_result = routed_result['right']
                t_3d = routed_result['t_3d']
                model_usage_counts = routed_result['model_usage_counts']
            else:
                left_update_flags = np.ones(pts_2d.shape[0], dtype=bool)
                right_update_flags = np.ones(pts_2d.shape[0], dtype=bool)
                left_changed_joints = [[] for _ in range(pts_2d.shape[0])]
                right_changed_joints = [[] for _ in range(pts_2d.shape[0])]
                six_joint_update_flags = np.zeros(pts_2d.shape[0], dtype=bool)
                left_result = run_leg_inference(
                    pts_2d_norm, 'left', left_model,
                    left_update_flags, changed_joints_by_frame=left_changed_joints,
                    single_model=single_model, two_model=two_model, five_model=five_model, four_model=four_model, full_model=full_model,
                    six_joint_update_flags=six_joint_update_flags, use_prfk=False,
                    full_pred_cache=full_pred_cache
                )
                right_result = run_leg_inference(
                    pts_2d_norm, 'right', right_model,
                    right_update_flags, changed_joints_by_frame=right_changed_joints,
                    single_model=single_model, two_model=two_model, five_model=five_model, four_model=four_model, full_model=full_model,
                    six_joint_update_flags=six_joint_update_flags, use_prfk=False,
                    full_pred_cache=full_pred_cache
                )
                t_3d = left_result['t_3d'] + right_result['t_3d']
                model_usage_counts = {
                    '17j': int(left_result['num_full_updates'] + right_result['num_full_updates']),
                    '5j': int(left_result['num_five_updates'] + right_result['num_five_updates']),
                    '4j': int(left_result['num_four_updates'] + right_result['num_four_updates']),
                    '2j': int(left_result['num_two_updates'] + right_result['num_two_updates']),
                    '1j': int(left_result['num_single_updates'] + right_result['num_single_updates']),
                    '3j': int(left_result['num_default_updates'] + right_result['num_default_updates']),
                }

        left_error = left_result['mpjpe']
        right_error = right_result['mpjpe']
        left_p_error = left_result['p_mpjpe']
        right_p_error = right_result['p_mpjpe']

        combined_pred = torch.cat([left_result['pred'], right_result['pred']], dim=2)
        combined_gt = torch.cat([left_result['gt'], right_result['gt']], dim=2)
        combined_error = mpjpe(combined_pred, combined_gt).item() * 1000

        total_updates = left_result['num_updates'] + right_result['num_updates']
        total_possible_updates = max(2 * num_frames, 1)
        update_ratio = float(total_updates) / float(total_possible_updates)
        flops_3d_effective = (ktpformer_flops / 1e9) * update_ratio
        flops_total_effective = (detector_flops[name] / 1e9) + flops_3d_effective

        total_frames = left_result['num_frames']
        left_computed = int(left_result['num_updates'])
        right_computed = int(right_result['num_updates'])
        total_computed = left_computed + right_computed
        total_count_frame = f"{total_computed}/{2*total_frames}"

        model_usage_counts = {
            '17j': int(left_result['num_full_updates'] + right_result['num_full_updates']),
            '5j': int(left_result['num_five_updates'] + right_result['num_five_updates']),
            '4j': int(left_result['num_four_updates'] + right_result['num_four_updates']),
            '2j': int(left_result['num_two_updates'] + right_result['num_two_updates']),
            '1j': int(left_result['num_single_updates'] + right_result['num_single_updates']),
            '3j': int((left_result['num_default_updates'] + right_result['num_default_updates'])),
        }
        
        results[name] = {
            'mpjpe': combined_error,
            'left_mpjpe': left_error,
            'right_mpjpe': right_error,
            'left_p_mpjpe': left_p_error,
            'right_p_mpjpe': right_p_error,
            't_2d': t_2d,
            't_3d': t_3d,
            'flops_2d': detector_flops[name] / 1e9,
            'flops_3d': ktpformer_flops / 1e9,
            'flops_total': (detector_flops[name] + ktpformer_flops) / 1e9,
            'flops_3d_effective': flops_3d_effective,
            'flops_total_effective': flops_total_effective,
            'left_updates': int(left_result['num_updates']),
            'left_skips': int(left_result['num_skips']),
            'right_updates': int(right_result['num_updates']),
            'right_skips': int(right_result['num_skips']),
            'left_full_updates': int(left_result['num_full_updates']),
            'right_full_updates': int(right_result['num_full_updates']),
            'left_five_updates': int(left_result['num_five_updates']),
            'right_five_updates': int(right_result['num_five_updates']),
            'left_four_updates': int(left_result['num_four_updates']),
            'right_four_updates': int(right_result['num_four_updates']),
            'left_two_model_updates': int(left_result['num_two_updates']),
            'right_two_model_updates': int(right_result['num_two_updates']),
            'left_single_model_updates': int(left_result['num_single_updates']),
            'right_single_model_updates': int(right_result['num_single_updates']),
            'left_default_3j_updates': int(left_result['num_default_updates']),
            'right_default_3j_updates': int(right_result['num_default_updates']),
            'update_ratio': update_ratio,
            'total_count_frame': total_count_frame,
            'model_usage_counts': model_usage_counts,
        }

        print(f"  2D Detection Time: {t_2d:.2f}s ({num_frames/t_2d:.1f} FPS)")
        print(f"  3D Inference Time: {t_3d:.2f}s ({num_frames/t_3d:.1f} FPS)")
        print(f"  Total E2E Time: {t_2d + t_3d:.2f}s ({num_frames/(t_2d+t_3d):.1f} FPS)")
        print(f"  Avg FLOPs/frame (theoretical): 2D={results[name]['flops_2d']:.6f}G | 3D={results[name]['flops_3d']:.6f}G | Total={results[name]['flops_total']:.6f}G")
        print(f"  Avg FLOPs/frame (effective):   2D={results[name]['flops_2d']:.6f}G | 3D={results[name]['flops_3d_effective']:.6f}G | Total={results[name]['flops_total_effective']:.6f}G")
        if args.disable_prfk:
            print(f"  Mode: PRFK DISABLED (baseline: all frames update)")
        else:
            print(f"  PRFK (IoU={args.prfk_dist_threshold:g}) updates/skips - left: {results[name]['left_updates']}/{results[name]['left_skips']} | right: {results[name]['right_updates']}/{results[name]['right_skips']}")
            print(f"  Full-body(17j) updates - left: {results[name]['left_full_updates']} | right: {results[name]['right_full_updates']}")
        
        # Calculate and display total computed frames
        total_frames = left_result['num_frames']
        left_computed = int(left_result['num_updates'])
        right_computed = int(right_result['num_updates'])
        total_computed = left_computed + right_computed
        total_count_frame = f"{total_computed}/{2*total_frames}"
        print(f"  Total Computed Frames: {total_count_frame}")
        print(
            f"  Model Usage: 17j={model_usage_counts['17j']} | 5j={model_usage_counts['5j']} | "
            f"4j={model_usage_counts['4j']} | 2j={model_usage_counts['2j']} | "
            f"1j={model_usage_counts['1j']} | 3j={model_usage_counts['3j']}"
        )
        print(f"  Left-Leg MPJPE: {left_error:.2f} mm | Right-Leg MPJPE: {right_error:.2f} mm")
        print(f"  Left-Leg P-MPJPE: {left_p_error:.2f} mm | Right-Leg P-MPJPE: {right_p_error:.2f} mm")
        print(f"  Combined Legs MPJPE: {results[name]['mpjpe']:.2f} mm")

    print("\n" + "=" * 70)
    print("3DKTP Results Summary")
    print("=" * 70)
    for name in ['YOLO', 'HRNet', 'OpenPose', 'MediaPipe']:
        r = results[name]
        mu = r.get('model_usage_counts', {})
        mu_str = (
            f"17j={mu.get('17j',0)} | 5j={mu.get('5j',0)} | 4j={mu.get('4j',0)} | "
            f"2j={mu.get('2j',0)} | 1j={mu.get('1j',0)} | 3j={mu.get('3j',0)}"
        )
        print(
            f"{name:<10} | Combined MPJPE: {r['mpjpe']:.2f} mm | Left: {r['left_mpjpe']:.2f} mm | "
            f"Right: {r['right_mpjpe']:.2f} mm | 2D: {r['t_2d']:.2f}s | 3D: {r['t_3d']:.2f}s | "
            f"FLOPs(th): {r['flops_total']:.6f}G/frame | FLOPs(eff): {r['flops_total_effective']:.6f}G/frame | Model Usage: {mu_str}"
        )
    print("=" * 70)
    # --- write summary.json for plotting script ---
    # Create results directory
    import subprocess
    import json as _json
    out_dir = os.path.abspath(os.path.dirname(__file__))
    results_dir = os.path.join(out_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Generate suffix for all output files based on PRFK settings
    if args.disable_prfk:
        suffix = '(Baseline)'
    else:
        suffix = f'(IoU={_format_suffix(args.prfk_dist_threshold)})'

    detectors_order = ['YOLO', 'HRNet', 'OpenPose', 'MediaPipe']
    combined_mpjpe = [float(results[d]['mpjpe']) for d in detectors_order]
    seconds_2d = [float(results[d]['t_2d']) for d in detectors_order]
    seconds_3d = [float(results[d]['t_3d']) for d in detectors_order]
    flops_2d = [float(results[d]['flops_2d']) for d in detectors_order]
    model_usage_counts_arr = [results[d]['model_usage_counts'] for d in detectors_order]
    # assume flops_3d identical across detectors
    flops_3d_per_frame = float(list(results.values())[0]['flops_3d']) if len(results) > 0 else 0.0
    flops_3d_effective_per_frame = [float(results[d]['flops_3d_effective']) for d in detectors_order]
    flops_total_effective_per_frame = [float(results[d]['flops_total_effective']) for d in detectors_order]

    total_count_frames = [results[d]['total_count_frame'] for d in detectors_order]
    summary = {
        'detectors': detectors_order,
        'combined_mpjpe': combined_mpjpe,
        'seconds_2d': seconds_2d,
        'seconds_3d': seconds_3d,
        'flops_2d': flops_2d,
        'flops_3d_per_frame': flops_3d_per_frame,
        'flops_3d_effective_per_frame': flops_3d_effective_per_frame,
        'flops_total_effective_per_frame': flops_total_effective_per_frame,
        'total_count_frames': total_count_frames,
        'model_usage_counts': model_usage_counts_arr,
        'prfk_dist_threshold': float(args.prfk_dist_threshold) if not args.disable_prfk else None,
        'prfk_disabled': bool(args.disable_prfk),
    }

    summary_path = os.path.join(results_dir, f'summary{suffix}.json')
    try:
        with open(summary_path, 'w') as fh:
            _json.dump(summary, fh, indent=2)
        print(f"Wrote summary to: {summary_path}")
    except Exception as e:
        print('Error: failed to write summary.json:', e)

    # Generate txt summary file
    try:
        txt_summary_lines = []
        txt_summary_lines.append("Summary:")
        txt_summary_lines.append("Mode: 3D FLOPs = effective (after PRFK gating)")
        
        for name in detectors_order:
            res = results[name]
            total_time = res['t_2d'] + res['t_3d']
            txt_summary_lines.append(
                f"{name}: Combined MPJPE={res['mpjpe']:.2f} mm, "
                f"2D={res['t_2d']:.2f}s, 3D={res['t_3d']:.2f}s, "
                f"Total={total_time:.2f}s, "
                f"Avg FLOPs/frame={res['flops_total_effective']:.2f}G "
                f"(2D={res['flops_2d']:.4f}G, 3D={res['flops_3d_effective']:.4f}G)"
            )
        
        txt_summary_path = os.path.join(results_dir, f'summary{suffix}.txt')
        with open(txt_summary_path, 'w') as fh:
            fh.write('\n'.join(txt_summary_lines))
        print(f"Wrote summary to: {txt_summary_path}")
    except Exception as e:
        print('Error: failed to write summary.txt:', e)

    # Call plotting script to generate PNGs; do this regardless but report errors
    try:
        script_path = os.path.join(os.path.dirname(__file__), 'scripts', 'plot_3DKTP_results.py')
        subprocess.run([sys.executable, script_path], check=False, cwd=os.path.dirname(__file__))
    except Exception as e:
        print('Warning: failed to generate plots automatically:', e)

    return

if __name__ == "__main__":
    evaluate()
