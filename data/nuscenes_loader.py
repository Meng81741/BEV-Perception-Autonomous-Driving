"""
NuScenes 零依赖加载器
=====================
直接解析 nuScenes JSON 标注文件，无需 nuscenes-devkit。
支持 v1.0-mini 和 v1.0-trainval。

用法:
    from nuscenes_loader import NuScenesLoader
    loader = NuScenesLoader('path/to/nuscenes', version='v1.0-mini')
    for sample in loader:
        images = sample['images']        # [6, H, W, 3] numpy
        boxes_3d = sample['boxes_3d']    # [N, 9]
        labels = sample['labels']        # [N]
"""

import os
import json
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Iterator
from collections import defaultdict


# ============================================================================
# NuScenes 类别映射
# ============================================================================

NUSCENES_CLASS_NAMES = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier',
]

NUSCENES_NAME_TO_ID = {
    'vehicle.car': 0,
    'vehicle.truck': 1,
    'vehicle.bus.rigid': 2,
    'vehicle.bus.bendy': 2,
    'vehicle.trailer': 3,
    'vehicle.construction': 4,
    'human.pedestrian.adult': 5,
    'human.pedestrian.child': 5,
    'human.pedestrian.construction_worker': 5,
    'human.pedestrian.personal_mobility': 5,
    'human.pedestrian.police_officer': 5,
    'human.pedestrian.stroller': 5,
    'human.pedestrian.wheelchair': 5,
    'vehicle.motorcycle': 6,
    'vehicle.bicycle': 7,
    'movable_object.trafficcone': 8,
    'movable_object.barrier': 9,
}

# nuScenes 6 相机
CAM_NAMES = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
]


class NuScenesLoader:
    """
    零依赖 nuScenes 数据加载器

    直接读取 JSON → 构建样本索引 → 按需加载图像和标注。
    """

    def __init__(
        self,
        data_root: str,
        version: str = 'v1.0-mini',
        image_size: Tuple[int, int] = (450, 800),
        num_cams: int = 6,
    ):
        self.data_root = data_root
        self.version = version
        self.image_size = image_size
        self.num_cams = num_cams

        # 加载 JSON 表
        self._tables = {}
        self._load_tables()

        # 构建样本列表
        self._samples = []
        self._build_sample_list()

        print(f"[NuScenes] {len(self._samples)} samples loaded from {version}")

    # ---- JSON 加载 ----

    def _load_tables(self):
        """加载所有标注 JSON 文件"""
        ann_dir = os.path.join(self.data_root, self.version)
        table_names = [
            'sample', 'sample_data', 'sample_annotation',
            'calibrated_sensor', 'ego_pose', 'sensor',
            'scene', 'category', 'attribute', 'instance',
        ]
        for name in table_names:
            path = os.path.join(ann_dir, f'{name}.json')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    self._tables[name] = json.load(f)
            else:
                self._tables[name] = []

        # 建立 token → record 索引
        self._idx = {}
        for name, records in self._tables.items():
            self._idx[name] = {r['token']: r for r in records}

    def _get(self, table: str, token: str) -> dict:
        return self._idx.get(table, {}).get(token, {})

    # ---- 样本列表构建 ----

    def _build_sample_list(self):
        """遍历所有 scene，收集 sample tokens，并建立 sample → camera data 映射"""
        # 建立 sample_token → [camera_data_tokens] 的映射
        self._sample_cam_data = defaultdict(dict)
        for rec in self._tables.get('sample_data', []):
            if not rec.get('is_key_frame', False):
                continue
            filename = rec.get('filename', '')
            # 提取相机名称: "samples/CAM_FRONT/xxx.jpg" → "CAM_FRONT"
            parts = filename.replace('\\', '/').split('/')
            if len(parts) >= 2 and parts[0] == 'samples':
                cam_name = parts[1]
                if cam_name in CAM_NAMES:
                    self._sample_cam_data[rec['sample_token']][cam_name] = rec['token']

        # 遍历 scene 收集样本（只保留有完整 6 相机数据的）
        for scene in self._tables.get('scene', []):
            token = scene.get('first_sample_token', '')
            count = 0
            while token and count < 1000:
                # 检查是否有完整的 6 相机数据
                cam_data = self._sample_cam_data.get(token, {})
                if len(cam_data) == len(CAM_NAMES):
                    self._samples.append(token)
                token_rec = self._get('sample', token)
                token = token_rec.get('next', '')
                count += 1

        # 建立 sample_token → annotations 索引
        self._sample_anns = defaultdict(list)
        for ann in self._tables.get('sample_annotation', []):
            self._sample_anns[ann['sample_token']].append(ann)

    def __len__(self) -> int:
        return len(self._samples)

    # ---- 数据提取 ----

    def get_images(self, sample_token: str) -> np.ndarray:
        """获取 6 路相机图像 [6, H, W, 3] RGB"""
        cam_data_map = self._sample_cam_data.get(sample_token, {})
        images = []

        for cam_name in CAM_NAMES:
            data_token = cam_data_map.get(cam_name, '')
            data_rec = self._get('sample_data', data_token)
            filename = data_rec.get('filename', '')

            full_path = os.path.normpath(os.path.join(self.data_root, filename))
            if os.path.exists(full_path):
                img = cv2.imread(full_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img.shape[:2] != self.image_size:
                    img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
            else:
                img = np.zeros((*self.image_size, 3), dtype=np.uint8)

            images.append(img)

        return np.stack(images, axis=0)  # [6, H, W, 3]

    def get_camera_params(self, sample_token: str) -> np.ndarray:
        """获取相机参数 [6, 16]"""
        cam_data_map = self._sample_cam_data.get(sample_token, {})
        params = []

        for cam_name in CAM_NAMES:
            data_token = cam_data_map.get(cam_name, '')
            data_rec = self._get('sample_data', data_token)

            calib_token = data_rec.get('calibrated_sensor_token', '')
            calib = self._get('calibrated_sensor', calib_token)

            ego_token = data_rec.get('ego_pose_token', '')
            ego = self._get('ego_pose', ego_token)

            # 内参
            intrinsic = np.array(calib.get('camera_intrinsic', np.eye(3)))
            fx, fy = intrinsic[0, 0], intrinsic[1, 1]
            cx, cy = intrinsic[0, 2], intrinsic[1, 2]

            # 外参
            trans_ego2cam = np.array(calib.get('translation', [0, 0, 0]))
            rot_ego2cam = np.array(calib.get('rotation', [1, 0, 0, 0]))

            trans_global2ego = np.array(ego.get('translation', [0, 0, 0]))
            rot_global2ego = np.array(ego.get('rotation', [1, 0, 0, 0]))

            # 拼接为 16 维
            p = np.concatenate([
                [fx, fy, cx, cy],
                rot_ego2cam, trans_ego2cam,
                rot_global2ego, trans_global2ego,
            ])
            p = np.pad(p, (0, max(0, 16 - len(p))))[:16]
            params.append(p)

        return np.stack(params, axis=0)  # [6, 16]

    def get_boxes(self, sample_token: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取 3D 标注框

        Returns:
            boxes_3d: [N, 9] (cx, cy, cz, l, w, h, yaw, vx, vy)
            labels: [N] int
        """
        boxes = []
        labels = []

        for ann in self._sample_anns.get(sample_token, []):
            # 获取类别名: annotation → instance → category
            instance = self._get('instance', ann.get('instance_token', ''))
            cat_token = instance.get('category_token', '')
            category = self._get('category', cat_token)
            cat_name = category.get('name', '')

            if cat_name not in NUSCENES_NAME_TO_ID:
                continue
            cls_id = NUSCENES_NAME_TO_ID[cat_name]

            cx, cy, cz = ann.get('translation', [0, 0, 0])
            l, w, h = ann.get('size', [0, 0, 0])

            # 四元数 → yaw
            qw, qx, qy, qz = ann.get('rotation', [1, 0, 0, 0])
            yaw = np.arctan2(2.0 * (qw * qz + qx * qy),
                             1.0 - 2.0 * (qy * qy + qz * qz))

            # 速度
            vel = ann.get('velocity', None)
            if vel is not None and len(vel) >= 2:
                vx, vy = float(vel[0]), float(vel[1])
            else:
                vx, vy = 0.0, 0.0

            boxes.append([cx, cy, cz, l, w, h, yaw, vx, vy])
            labels.append(cls_id)

        if len(boxes) == 0:
            return np.zeros((0, 9), dtype=np.float32), np.zeros(0, dtype=np.int64)

        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def __getitem__(self, idx: int) -> Dict:
        """获取单个样本"""
        token = self._samples[idx]

        images = self.get_images(token)
        cam_params = self.get_camera_params(token)
        boxes_3d, labels = self.get_boxes(token)

        return {
            'images': images,           # [6, H, W, 3]
            'camera_params': cam_params, # [6, 16]
            'boxes_3d': boxes_3d,        # [N, 9]
            'labels': labels,            # [N]
            'token': token,
        }

    def __iter__(self) -> Iterator[Dict]:
        for i in range(len(self)):
            yield self[i]


# ============================================================================
# 快速测试
# ============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default=r'D:\workDir\driver_detection\datasat\nuscenes')
    parser.add_argument('--version', type=str, default='v1.0-mini')
    args = parser.parse_args()

    print("=" * 50)
    print("NuScenes 零依赖加载器测试")
    print("=" * 50)

    loader = NuScenesLoader(args.data_root, args.version)

    # 测试前 3 个样本
    for i in range(min(3, len(loader))):
        sample = loader[i]
        print(f"\nSample {i}: {sample['token'][:12]}...")
        print(f"  Images: {sample['images'].shape}  ({sample['images'].dtype})")
        print(f"  Cam params: {sample['camera_params'].shape}")
        print(f"  Boxes: {sample['boxes_3d'].shape}")
        print(f"  Labels: {sample['labels'].shape}")
        if len(sample['labels']) > 0:
            unique, counts = np.unique(sample['labels'], return_counts=True)
            for cls_id, cnt in zip(unique, counts):
                name = NUSCENES_CLASS_NAMES[cls_id] if cls_id < len(NUSCENES_CLASS_NAMES) else '?'
                print(f"    {name}: {cnt}")

    print("\n✅ 加载器正常工作!")
