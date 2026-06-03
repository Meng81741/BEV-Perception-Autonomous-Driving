"""
数据集定义
==========
定义 BEV 多任务感知数据集接口，支持 NuScenes 等自动驾驶数据集。

数据集结构:
  每个样本包含:
  - images: [N_cam, H, W, 3] 环视相机图像
  - camera_params: [N_cam, 16] 相机内外参数
  - boxes_3d: [N, 9] 3D 标注框 (cx, cy, cz, l, w, h, yaw, vx, vy)
  - labels: [N] 类别标签
  - drivable_mask: [H_bev, W_bev] 可行驶区域掩码
  - lane_mask: [H_bev, W_bev] 车道线掩码
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json

from .transforms import TrainTransform, ValTransform
from .bev_grid import BEVGrid


class BEVPerceptionDataset(Dataset):
    """
    BEV 多任务感知数据集基类

    支持的数据集格式:
    - NuScenes (通过 nuscenes-devkit)
    - 自定义格式（通过 pickle / json 标注文件）
    """

    CLASS_NAMES = [
        'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
        'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier',
    ]

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (900, 1600),
        bev_h: int = 128,
        bev_w: int = 128,
        num_cams: int = 6,
        use_nuscenes: bool = False,
        ann_file: Optional[str] = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_cams = num_cams
        self.use_nuscenes = use_nuscenes

        # BEV 网格
        self.bev_grid = BEVGrid(
            x_bound=(-51.2, 51.2, 102.4 / bev_h),
            y_bound=(-51.2, 51.2, 102.4 / bev_w),
        )

        # 数据变换
        if split == 'train':
            self.transform = TrainTransform(image_size=image_size)
        else:
            self.transform = ValTransform(image_size=image_size)

        # ---- 加载数据索引 ----
        if use_nuscenes:
            self._init_nuscenes()
        else:
            self._init_custom(ann_file)

    def _init_nuscenes(self):
        """
        初始化 NuScenes 数据集

        需要安装 nuscenes-devkit:
          pip install nuscenes-devkit
        """
        try:
            from nuscenes.nuscenes import NuScenes
            self.nusc = NuScenes(
                version='v1.0-trainval',
                dataroot=self.data_root,
                verbose=False,
            )
        except ImportError:
            raise ImportError(
                "请安装 nuscenes-devkit: pip install nuscenes-devkit"
            )

        # 获取场景-样本映射
        self.samples = []
        for scene in self.nusc.scene:
            sample_token = scene['first_sample_token']
            while sample_token:
                sample = self.nusc.get('sample', sample_token)
                self.samples.append(sample)
                sample_token = sample['next']

        print(f"[NuScenes] 加载 {len(self.samples)} 个样本 ({self.split})")

    def _init_custom(self, ann_file: Optional[str] = None):
        """
        初始化自定义数据集

        自定义标注文件格式 (JSON):
        {
            "samples": [
                {
                    "token": "sample_001",
                    "images": ["cam_front.jpg", "cam_front_left.jpg", ...],
                    "camera_params": [[...], ...],
                    "boxes_3d": [[cx,cy,cz,l,w,h,yaw,vx,vy], ...],
                    "labels": [0, 1, ...],
                    "drivable_mask": "drivable_001.png",
                    "lane_mask": "lane_001.png",
                },
                ...
            ]
        }
        """
        if ann_file is None:
            # 搜索常见标注文件
            candidates = ['annotations.json', 'labels.json', 'samples.json']
            for cand in candidates:
                path = os.path.join(self.data_root, cand)
                if os.path.exists(path):
                    ann_file = path
                    break

        if ann_file and os.path.exists(ann_file):
            with open(ann_file, 'r') as f:
                data = json.load(f)
            self.samples = data['samples']
            print(f"[Custom] 加载 {len(self.samples)} 个样本来自 {ann_file}")
        else:
            # 演示模式：生成虚拟样本
            print("[Custom] 未找到标注文件，使用演示模式")
            self.samples = self._generate_demo_samples()

    def _generate_demo_samples(self, num_samples: int = 100) -> List[Dict]:
        """生成演示用虚拟样本数据"""
        samples = []
        for i in range(num_samples):
            samples.append({
                'token': f'demo_{i:04d}',
                'images': [f'cam_{c}_{i:04d}.jpg' for c in range(self.num_cams)],
                'camera_params': np.random.randn(self.num_cams, 16).tolist(),
                'boxes_3d': np.random.randn(np.random.randint(0, 20), 9).tolist(),
                'labels': np.random.randint(0, 10, np.random.randint(0, 20)).tolist(),
                'drivable_mask': f'drivable_{i:04d}.png',
                'lane_mask': f'lane_{i:04d}.png',
            })
        return samples

    def _load_image(self, img_path: str) -> np.ndarray:
        """加载单张图像"""
        import cv2
        full_path = os.path.join(self.data_root, img_path)

        if not os.path.exists(full_path):
            # 演示模式：返回随机图像
            return np.random.randint(0, 256, (*self.image_size, 3), dtype=np.uint8)

        img = cv2.imread(full_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_mask(self, mask_path: str, h: int, w: int) -> np.ndarray:
        """加载分割掩码"""
        import cv2
        full_path = os.path.join(self.data_root, mask_path)

        if not os.path.exists(full_path):
            # 演示模式：返回随机掩码
            return np.random.randint(0, 2, (h, w), dtype=np.int64)

        mask = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return (mask > 128).astype(np.int64)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        获取一个样本

        Returns:
            {
                'images': Tensor [N_cam, 3, H, W],
                'camera_params': Tensor [N_cam, 16],
                'boxes_3d': Tensor [N, 9],
                'labels': Tensor [N],
                'drivable_mask': Tensor [H_bev, W_bev],
                'lane_mask': Tensor [H_bev, W_bev],
                'token': str,
            }
        """
        sample = self.samples[idx]

        # ---- 加载图像 ----
        images = []
        for img_rel_path in sample['images']:
            img = self._load_image(img_rel_path)
            images.append(img)
        images = np.stack(images, axis=0)  # [N_cam, H, W, 3]

        # ---- 加载相机参数 ----
        camera_params = np.array(sample.get('camera_params', np.zeros((self.num_cams, 16))),
                                 dtype=np.float32)

        # ---- 加载 3D 框标签 ----
        boxes_3d = np.array(sample.get('boxes_3d', np.zeros((0, 9))), dtype=np.float32)
        labels = np.array(sample.get('labels', np.zeros(0, dtype=np.int64)), dtype=np.int64)

        # ---- 加载分割掩码 ----
        if 'drivable_mask' in sample:
            drivable_mask = self._load_mask(sample['drivable_mask'], self.bev_h, self.bev_w)
        else:
            drivable_mask = np.zeros((self.bev_h, self.bev_w), dtype=np.int64)

        if 'lane_mask' in sample:
            lane_mask = self._load_mask(sample['lane_mask'], self.bev_h, self.bev_w)
        else:
            lane_mask = np.zeros((self.bev_h, self.bev_w), dtype=np.int64)

        # ---- 构建初始数据字典 ----
        item = {
            'images': images,
            'camera_params': camera_params,
            'boxes_3d': boxes_3d,
            'labels': labels,
            'drivable_mask': drivable_mask,
            'lane_mask': lane_mask,
            'token': sample['token'],
        }

        # ---- 数据变换 ----
        if self.transform is not None:
            item = self.transform(item)

        return item


def collate_fn(batch: List[Dict]) -> Dict:
    """
    自定义 batch 整理函数
    —— 处理不同数量的 3D 框
    """
    images = torch.stack([item['images'] for item in batch])          # [B, N_cam, 3, H, W]
    camera_params = torch.stack([torch.as_tensor(
        item['camera_params'], dtype=torch.float32
    ) for item in batch])

    # 填充到相同尺寸
    max_boxes = max(item['boxes_3d'].shape[0] if isinstance(item['boxes_3d'], torch.Tensor)
                    else len(item['boxes_3d']) for item in batch)

    boxes_3d_list = []
    labels_list = []
    num_boxes = []

    for item in batch:
        boxes = torch.as_tensor(item['boxes_3d'], dtype=torch.float32)
        lbls = torch.as_tensor(item['labels'], dtype=torch.long)
        n = boxes.shape[0]
        num_boxes.append(n)

        if n < max_boxes:
            pad_boxes = torch.zeros(max_boxes - n, 9)
            pad_lbls = torch.full((max_boxes - n,), -1, dtype=torch.long)
            boxes = torch.cat([boxes, pad_boxes], dim=0)
            lbls = torch.cat([lbls, pad_lbls], dim=0)

        boxes_3d_list.append(boxes)
        labels_list.append(lbls)

    boxes_3d = torch.stack(boxes_3d_list)
    labels = torch.stack(labels_list)

    drivable_mask = torch.stack([
        torch.as_tensor(item['drivable_mask'].copy(), dtype=torch.long)
        for item in batch
    ])
    lane_mask = torch.stack([
        torch.as_tensor(item['lane_mask'].copy(), dtype=torch.long)
        for item in batch
    ])

    return {
        'images': images,
        'camera_params': camera_params,
        'boxes_3d': boxes_3d,
        'labels': labels,
        'num_boxes': torch.tensor(num_boxes, dtype=torch.long),
        'drivable_mask': drivable_mask,
        'lane_mask': lane_mask,
    }


def build_dataloader(
    data_root: str,
    split: str = 'train',
    batch_size: int = 1,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (900, 1600),
    bev_h: int = 128,
    bev_w: int = 128,
    num_cams: int = 6,
    use_nuscenes: bool = False,
    ann_file: Optional[str] = None,
    shuffle: bool = True,
) -> DataLoader:
    """
    构建 DataLoader

    Args:
        data_root: 数据根目录
        split: 'train' / 'val' / 'test'
        batch_size: 批次大小
        num_workers: 数据加载线程数
        image_size: 图像尺寸 (H, W)
        bev_h, bev_w: BEV 网格尺寸
        num_cams: 环视相机数量
        use_nuscenes: 是否使用 NuScenes 数据集
        ann_file: 自定义标注文件路径
        shuffle: 是否打乱数据
    """
    dataset = BEVPerceptionDataset(
        data_root=data_root,
        split=split,
        image_size=image_size,
        bev_h=bev_h,
        bev_w=bev_w,
        num_cams=num_cams,
        use_nuscenes=use_nuscenes,
        ann_file=ann_file,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=split == 'train',
    )

    return dataloader
