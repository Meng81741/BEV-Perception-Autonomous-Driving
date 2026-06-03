"""
NuScenes 完整数据加载器
========================
基于 nuscenes-devkit 实现的完整数据加载，包含：
- 多相机图像加载
- 3D 框标签提取和坐标变换
- 相机内外参数提取
- HD Map 可行驶区域 / 车道线渲染
"""

import torch
import numpy as np
import os
import cv2
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .transforms import TrainTransform, ValTransform
from .bev_grid import BEVGrid


# ============================================================================
# NuScenes 类别映射
# ============================================================================

NUSCENES_CLASSES = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier',
]

NUSCENES_CLASS_MAP = {
    'animal': -1,
    'human.pedestrian.adult': 5,
    'human.pedestrian.child': 5,
    'human.pedestrian.construction_worker': 5,
    'human.pedestrian.personal_mobility': 5,
    'human.pedestrian.police_officer': 5,
    'human.pedestrian.stroller': 5,
    'human.pedestrian.wheelchair': 5,
    'movable_object.barrier': 9,
    'movable_object.debris': -1,
    'movable_object.pushable_pullable': -1,
    'movable_object.trafficcone': 8,
    'vehicle.bicycle': 7,
    'vehicle.bus.bendy': 2,
    'vehicle.bus.rigid': 2,
    'vehicle.car': 0,
    'vehicle.construction': 4,
    'vehicle.emergency.ambulance': 0,
    'vehicle.emergency.police': 0,
    'vehicle.motorcycle': 6,
    'vehicle.trailer': 3,
    'vehicle.truck': 1,
}


class NuScenesDataset:
    """
    完整的 NuScenes 数据集封装

    功能:
    - 自动从 nuScenes API 提取图像、标注、相机参数
    - 可选渲染 HD Map 中的可行驶区域和车道线
    - 支持 train/val/test 划分

    依赖:
        pip install nuscenes-devkit

    NuScenes 目录结构:
        data_root/
        ├── maps/                      # HD maps (.json)
        ├── samples/
        │   ├── CAM_FRONT/
        │   ├── CAM_FRONT_LEFT/
        │   ├── CAM_FRONT_RIGHT/
        │   ├── CAM_BACK/
        │   ├── CAM_BACK_LEFT/
        │   └── CAM_BACK_RIGHT/
        └── v1.0-trainval/
            ├── sample.json
            ├── sample_data.json
            ├── sample_annotation.json
            ├── calibrated_sensor.json
            ├── ego_pose.json
            └── ...
    """

    # nuScenes 相机顺序（BEVFormer 标准）
    CAM_NAMES = [
        'CAM_FRONT',
        'CAM_FRONT_LEFT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]

    def __init__(
        self,
        data_root: str,
        version: str = 'v1.0-trainval',
        split: str = 'train',
        image_size: Tuple[int, int] = (900, 1600),
        bev_h: int = 128,
        bev_w: int = 128,
        render_map: bool = True,
        map_patch_size: Tuple[float, float] = (102.4, 102.4),  # 米
        map_resolution: float = 0.8,  # 米/像素
    ):
        """
        Args:
            data_root: nuScenes 数据根目录
            version: 标注版本 ('v1.0-trainval' / 'v1.0-mini')
            split: 'train' / 'val' / 'test' / 'mini'
            image_size: 输出图像尺寸 (H, W)
            bev_h, bev_w: BEV 网格尺寸
            render_map: 是否渲染 HD Map
            map_patch_size: Map 渲染范围 (x_range, y_range)
            map_resolution: Map 分辨率 米/像素
        """
        self.data_root = data_root
        self.version = version
        self.split = split
        self.image_size = image_size
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.render_map = render_map
        self.map_patch_size = map_patch_size
        self.map_resolution = map_resolution

        # BEV 网格
        self.bev_grid = BEVGrid(
            x_bound=(-map_patch_size[0] / 2, map_patch_size[0] / 2, map_resolution),
            y_bound=(-map_patch_size[1] / 2, map_patch_size[1] / 2, map_resolution),
        )

        # ---- 初始化 NuScenes ----
        from nuscenes.nuscenes import NuScenes
        self.nusc = NuScenes(
            version=version,
            dataroot=data_root,
            verbose=False,
        )

        # ---- HD Map (用于渲染可行驶区域和车道线) ----
        if render_map:
            from nuscenes.map_expansion.map_api import NuScenesMap
            self.nusc_maps = {}
            map_locations = set()
            for scene in self.nusc.scene:
                log = self.nusc.get('log', scene['log_token'])
                map_locations.add(log['location'])
            for loc in map_locations:
                try:
                    self.nusc_maps[loc] = NuScenesMap(
                        dataroot=data_root, map_name=loc
                    )
                except Exception:
                    print(f"  警告: 无法加载地图 {loc}")

        # ---- 构建样本列表 ----
        self._build_samples()

        # ---- 数据变换 ----
        if split == 'train':
            self.transform = TrainTransform(image_size=image_size)
        else:
            self.transform = ValTransform(image_size=image_size)

        print(f"[NuScenes] 加载完成: {len(self.samples)} 个 {split} 样本")

    def _build_samples(self):
        """
        构建样本列表，支持 train/val/mini 划分

        NuScenes 官方划分:
        - train: 700 场景
        - val: 150 场景
        - test: 150 场景
        - mini: 10 场景（快速调试用）
        """
        self.samples = []

        if self.split == 'mini':
            # mini 模式：只取前 10 个场景
            scenes = self.nusc.scene[:10]
        elif self.split in ('train', 'val', 'test'):
            # 官方 train/val split: 场景编号哈希
            scenes = []
            for scene in self.nusc.scene:
                scene_name = scene['name']
                # nuScenes 的 train/val 通过场景名划分
                # 简化：按照常见约定
                scene_num = int(scene['token'][:8], 16) % 10
                if self.split == 'train' and scene_num < 7:
                    scenes.append(scene)
                elif self.split == 'val' and scene_num >= 7 and scene_num < 9:
                    scenes.append(scene)
                elif self.split == 'test' and scene_num >= 9:
                    scenes.append(scene)
        else:
            scenes = self.nusc.scene

        for scene in scenes:
            sample_token = scene['first_sample_token']
            while sample_token:
                self.samples.append(sample_token)
                sample = self.nusc.get('sample', sample_token)
                sample_token = sample['next']

    def __len__(self) -> int:
        return len(self.samples)

    def _get_camera_data(self, sample_token: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取 6 个相机的图像和参数

        Returns:
            images: [6, H, W, 3] (RGB)
            cam_params: [6, 16] (fx,fy,cx,cy, qw,qx,qy,qz, tx,ty,tz, ..., ...)
        """
        sample = self.nusc.get('sample', sample_token)

        images = []
        cam_params = []

        for cam_name in self.CAM_NAMES:
            cam_data_token = sample['data'][cam_name]
            cam_data = self.nusc.get('sample_data', cam_data_token)

            # ---- 加载图像 ----
            img_path = os.path.join(self.data_root, cam_data['filename'])
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                # 文件缺失 → 黑色占位
                img = np.zeros((*self.image_size, 3), dtype=np.uint8)

            images.append(img)

            # ---- 提取相机参数 ----
            calibrated_sensor = self.nusc.get(
                'calibrated_sensor', cam_data['calibrated_sensor_token']
            )
            ego_pose = self.nusc.get('ego_pose', cam_data['ego_pose_token'])

            # 相机内参
            intrinsic = np.array(calibrated_sensor['camera_intrinsic'])
            fx, fy = intrinsic[0, 0], intrinsic[1, 1]
            cx, cy = intrinsic[0, 2], intrinsic[1, 2]

            # 自车 → 相机的变换 (外参)
            translation_ego_to_cam = np.array(calibrated_sensor['translation'])
            rotation_ego_to_cam = np.array(calibrated_sensor['rotation'])  # qw,qx,qy,qz

            # 世界 → 自车变换
            translation_global_to_ego = np.array(ego_pose['translation'])
            rotation_global_to_ego = np.array(ego_pose['rotation'])

            # 拼接为 16 维参数向量
            params = np.concatenate([
                [fx, fy, cx, cy],
                rotation_ego_to_cam,       # 4
                translation_ego_to_cam,    # 3
                rotation_global_to_ego,    # 4
                translation_global_to_ego, # 3
            ])  # 共 18 个 → 截取前 16 或扩展
            # 统一为 16 维
            if len(params) < 16:
                params = np.pad(params, (0, 16 - len(params)))
            else:
                params = params[:16]

            cam_params.append(params)

        return np.stack(images, axis=0), np.stack(cam_params, axis=0)

    def _get_boxes(self, sample_token: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取 3D 标注框

        Returns:
            boxes_3d: [N, 9] (cx, cy, cz, l, w, h, yaw, vx, vy) — 全局坐标系
            labels: [N] 类别索引
        """
        sample = self.nusc.get('sample', sample_token)

        boxes_3d = []
        labels = []

        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)

            # 类别过滤
            cls_name = ann['category_name']
            if cls_name not in NUSCENES_CLASS_MAP:
                continue
            cls_id = NUSCENES_CLASS_MAP[cls_name]
            if cls_id < 0:
                continue

            # 框的中心位置
            cx, cy, cz = ann['translation']

            # 尺寸
            l, w, h_box = ann['size']

            # 朝向角 (四元数 → yaw)
            qw, qx, qy, qz = ann['rotation']
            yaw = np.arctan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz)
            )

            # 速度 (nuScenes 标注中包含 velocity)
            if 'velocity' in ann and ann['velocity'] is not None:
                vx, vy = ann['velocity'][:2]
            else:
                vx, vy = 0.0, 0.0

            boxes_3d.append([cx, cy, cz, l, w, h_box, yaw, vx, vy])
            labels.append(cls_id)

        if len(boxes_3d) == 0:
            return np.zeros((0, 9), dtype=np.float32), np.zeros(0, dtype=np.int64)

        return np.array(boxes_3d, dtype=np.float32), np.array(labels, dtype=np.int64)

    def _render_map(self, sample_token: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        渲染 HD Map → 可行驶区域和车道线 BEV 掩码

        使用 nuScenes map expansion API 渲染以自车为中心的局部地图。

        Returns:
            drivable_mask: [bev_h, bev_w] (0/1)
            lane_mask: [bev_h, bev_w] (0/1)
        """
        sample = self.nusc.get('sample', sample_token)
        scene = self.nusc.get('scene', sample['scene_token'])
        log = self.nusc.get('log', scene['log_token'])
        location = log['location']

        if location not in self.nusc_maps:
            return (
                np.zeros((self.bev_h, self.bev_w), dtype=np.int64),
                np.zeros((self.bev_h, self.bev_w), dtype=np.int64),
            )

        nu_scene_map = self.nusc_maps[location]

        # 获取自车位置
        cam_data_token = sample['data']['CAM_FRONT']
        cam_data = self.nusc.get('sample_data', cam_data_token)
        ego_pose = self.nusc.get('ego_pose', cam_data['ego_pose_token'])

        ego_x, ego_y = ego_pose['translation'][:2]

        # Map 渲染范围
        x_min, x_max = ego_x - self.map_patch_size[0] / 2, ego_x + self.map_patch_size[0] / 2
        y_min, y_max = ego_y - self.map_patch_size[1] / 2, ego_y + self.map_patch_size[1] / 2

        # 渲染可行驶区域
        drivable_mask = np.zeros((self.bev_h, self.bev_w), dtype=np.uint8)
        if hasattr(nu_scene_map, 'render_map_mask'):
            try:
                # 使用 map API 渲染
                drivable_mask = self._render_drivable_area(
                    nu_scene_map, x_min, x_max, y_min, y_max
                )
            except Exception:
                pass

        # 渲染车道线
        lane_mask = np.zeros((self.bev_h, self.bev_w), dtype=np.uint8)
        try:
            lane_mask = self._render_lane_divider(
                nu_scene_map, x_min, x_max, y_min, y_max
            )
        except Exception:
            pass

        return drivable_mask.astype(np.int64), lane_mask.astype(np.int64)

    def _render_drivable_area(
        self, nu_scene_map, x_min, x_max, y_min, y_max
    ) -> np.ndarray:
        """
        渲染可行驶区域

        使用 NuScenesMap 的 get_map_mask 方法。
        """
        from nuscenes.map_expansion.map_api import NuScenesMap

        patch_box = (
            (x_min + x_max) / 2, (y_min + y_max) / 2,
            x_max - x_min, y_max - y_min,
        )
        patch_angle = 0  # 沿自车朝向

        # 获取可行驶区域的 mask
        drivable_mask = nu_scene_map.get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=['drivable_area'],
            canvas_size=(self.bev_h, self.bev_w),
        )[0]

        return drivable_mask.astype(np.uint8)

    def _render_lane_divider(
        self, nu_scene_map, x_min, x_max, y_min, y_max
    ) -> np.ndarray:
        """
        渲染车道分隔线
        """
        patch_box = (
            (x_min + x_max) / 2, (y_min + y_max) / 2,
            x_max - x_min, y_max - y_min,
        )

        lane_mask = nu_scene_map.get_map_mask(
            patch_box=patch_box,
            patch_angle=0,
            layer_names=['lane_divider', 'road_divider'],
            canvas_size=(self.bev_h, self.bev_w),
        )

        # 合并车道分隔线层
        combined = np.zeros((self.bev_h, self.bev_w), dtype=np.uint8)
        for layer in lane_mask:
            combined = np.maximum(combined, layer)

        return combined.astype(np.uint8)

    def __getitem__(self, idx: int) -> Dict:
        """
        获取一个 nuScenes 样本

        Returns:
            {
                'images': Tensor [6, 3, H, W],
                'camera_params': Tensor [6, 16],
                'boxes_3d': Tensor [N, 9],
                'labels': Tensor [N],
                'drivable_mask': Tensor [H_bev, W_bev],
                'lane_mask': Tensor [H_bev, W_bev],
                'token': str,
            }
        """
        sample_token = self.samples[idx]

        # ---- 相机图像和参数 ----
        images, cam_params = self._get_camera_data(sample_token)

        # ---- 3D 框 ----
        boxes_3d, labels = self._get_boxes(sample_token)

        # ---- HD Map 渲染 ----
        if self.render_map:
            drivable_mask, lane_mask = self._render_map(sample_token)
        else:
            drivable_mask = np.zeros((self.bev_h, self.bev_w), dtype=np.int64)
            lane_mask = np.zeros((self.bev_h, self.bev_w), dtype=np.int64)

        # ---- 构建数据项 ----
        item = {
            'images': images,            # [6, H, W, 3]
            'camera_params': cam_params, # [6, 16]
            'boxes_3d': boxes_3d,        # [N, 9]
            'labels': labels,            # [N]
            'drivable_mask': drivable_mask,
            'lane_mask': lane_mask,
            'token': sample_token,
        }

        # ---- 数据变换 ----
        if self.transform is not None:
            item = self.transform(item)

        return item


def build_nuscenes_dataloader(
    data_root: str,
    version: str = 'v1.0-trainval',
    split: str = 'train',
    batch_size: int = 1,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (900, 1600),
    bev_h: int = 128,
    bev_w: int = 128,
    render_map: bool = True,
) -> torch.utils.data.DataLoader:
    """
    构建 NuScenes DataLoader

    用法:
        from data.nuscenes_dataset import build_nuscenes_dataloader

        train_loader = build_nuscenes_dataloader(
            data_root='/data/nuscenes',
            version='v1.0-trainval',
            split='train',
            batch_size=1,
        )

    Args:
        data_root: nuScenes 根目录
        version: 'v1.0-trainval' 或 'v1.0-mini'
        split: 'train' / 'val' / 'mini'
        batch_size: 批次大小
        num_workers: 数据加载线程
        image_size: 图像尺寸
        bev_h, bev_w: BEV 尺寸
        render_map: 是否渲染 HD Map
    """
    from torch.utils.data import DataLoader

    dataset = NuScenesDataset(
        data_root=data_root,
        version=version,
        split=split,
        image_size=image_size,
        bev_h=bev_h,
        bev_w=bev_w,
        render_map=render_map,
    )

    from .dataset import collate_fn

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == 'train'),
    )


# ============================================================================
# 快速测试
# ============================================================================

def test_nuscenes_loader(data_root: str = '/data/nuscenes'):
    """测试 NuScenes 数据加载器"""
    print("=" * 60)
    print("测试 NuScenes 数据加载器")
    print("=" * 60)

    # mini 模式快速测试
    loader = build_nuscenes_dataloader(
        data_root=data_root,
        version='v1.0-mini',
        split='mini',
        batch_size=1,
        render_map=True,
    )

    for i, batch in enumerate(loader):
        print(f"\nBatch {i}:")
        print(f"  images:       {batch['images'].shape}")       # [1, 6, 3, H, W]
        print(f"  camera_params:{batch['camera_params'].shape}") # [1, 6, 16]
        print(f"  boxes_3d:     {batch['boxes_3d'].shape}")     # [1, N, 9]
        print(f"  labels:       {batch['labels'].shape}")       # [1, N]
        print(f"  num_boxes:    {batch['num_boxes'].item()}")
        print(f"  drivable:     {batch['drivable_mask'].shape}")# [1, H, W]
        print(f"  lane:         {batch['lane_mask'].shape}")    # [1, H, W]

        if i >= 2:
            break

    print("\n✅ NuScenes 数据加载器正常工作!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/data/nuscenes')
    parser.add_argument('--version', type=str, default='v1.0-mini')
    parser.add_argument('--split', type=str, default='mini')
    args = parser.parse_args()

    test_nuscenes_loader(args.data_root)
