"""
数据预处理与增强
================
BEV 感知任务的数据增广与预处理操作。
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import random


# ============================================================================
# 图像变换
# ============================================================================

class ResizeImage:
    """图像缩放"""

    def __init__(self, size: Tuple[int, int] = (900, 1600)):
        self.size = size  # (H, W)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, (self.size[1], self.size[0]))


class NormalizeImage:
    """图像归一化"""

    def __init__(
        self,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        return image


class RandomHorizontalFlip:
    """随机水平翻转"""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: np.ndarray, targets: Optional[Dict] = None):
        if random.random() < self.p:
            image = np.ascontiguousarray(image[:, ::-1, :])
            if targets is not None:
                # 需要同步翻转 BEV 标签
                if 'drivable_mask' in targets:
                    targets['drivable_mask'] = targets['drivable_mask'][:, ::-1]
                if 'lane_mask' in targets:
                    targets['lane_mask'] = targets['lane_mask'][:, ::-1]
                if 'boxes_3d' in targets:
                    boxes = targets['boxes_3d']
                    boxes[:, 1] = -boxes[:, 1]  # 翻转 y 坐标
                    boxes[:, 6] = -boxes[:, 6]  # 翻转朝向角
                    targets['boxes_3d'] = boxes
        return image, targets


class RandomColorJitter:
    """随机颜色抖动"""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        p: float = 0.5,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            # 亮度
            if random.random() < 0.5:
                delta = random.uniform(-self.brightness, self.brightness)
                image = np.clip(image + delta * 255, 0, 255).astype(np.uint8)

            # 对比度和饱和度（转 HSV）
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            if random.random() < 0.5:
                hsv[:, :, 1] *= random.uniform(1 - self.saturation, 1 + self.saturation)
            if random.random() < 0.5:
                hsv[:, :, 2] *= random.uniform(1 - self.contrast, 1 + self.contrast)
            if random.random() < 0.5:
                hsv[:, :, 0] += random.uniform(-self.hue, self.hue) * 180
                hsv[:, :, 0] = hsv[:, :, 0] % 180
            image = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)

        return image


# ============================================================================
# BEV 标签生成
# ============================================================================

def generate_heatmap(
    boxes_3d: np.ndarray,
    classes: np.ndarray,
    bev_h: int,
    bev_w: int,
    bev_range: Tuple[float, float, float] = (-51.2, 51.2, 0.8),
    radius: int = 2,
) -> np.ndarray:
    """
    根据 3D 框生成 BEV 热力图标签

    Args:
        boxes_3d: [N, 7] (cx, cy, cz, l, w, h, yaw)
        classes: [N] 类别索引
        bev_h, bev_w: BEV 网格尺寸
        bev_range: (min, max, resolution)
        radius: 高斯半径
    Returns:
        heatmap: [num_classes, bev_h, bev_w]
    """
    x_min, x_max, x_res = bev_range
    y_min, y_max, y_res = bev_range
    num_classes = int(classes.max()) + 1 if len(classes) > 0 else 10

    heatmap = np.zeros((num_classes, bev_h, bev_w), dtype=np.float32)

    for box, cls_id in zip(boxes_3d, classes):
        cx, cy = box[0], box[1]

        # 世界坐标 → 网格
        h_idx = int((cx - x_min) / x_res)
        w_idx = int((cy - y_min) / y_res)

        if 0 <= h_idx < bev_h and 0 <= w_idx < bev_w:
            # 高斯核
            y_grid, x_grid = np.ogrid[-radius:radius+1, -radius:radius+1]
            gaussian = np.exp(-(x_grid**2 + y_grid**2) / (2 * (radius / 3) ** 2))

            h_start = max(0, h_idx - radius)
            h_end = min(bev_h, h_idx + radius + 1)
            w_start = max(0, w_idx - radius)
            w_end = min(bev_w, w_idx + radius + 1)

            g_h_start = radius - (h_idx - h_start)
            g_h_end = radius + (h_end - h_idx)
            g_w_start = radius - (w_idx - w_start)
            g_w_end = radius + (w_end - w_idx)

            heatmap[int(cls_id), h_start:h_end, w_start:w_end] = np.maximum(
                heatmap[int(cls_id), h_start:h_end, w_start:w_end],
                gaussian[g_h_start:g_h_end, g_w_start:g_w_end],
            )

    return heatmap


# ============================================================================
# 组合变换
# ============================================================================

class Compose:
    """组合多个变换"""

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, item: Dict) -> Dict:
        for t in self.transforms:
            item = t(item)
        return item


class TrainTransform:
    """训练数据变换流水线"""

    def __init__(
        self,
        image_size: Tuple[int, int] = (900, 1600),
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        use_flip: bool = True,
        use_color_jitter: bool = True,
    ):
        self.resize = ResizeImage(image_size)
        self.normalize = NormalizeImage(mean, std)
        self.flip = RandomHorizontalFlip(p=0.5) if use_flip else None
        self.color_jitter = RandomColorJitter(p=0.5) if use_color_jitter else None

    def __call__(self, item: Dict) -> Dict:
        images = item['images']  # [N_cam, H, W, 3]

        processed_images = []
        for img in images:
            img = self.resize(img)
            if self.color_jitter is not None:
                img = self.color_jitter(img)
            img = self.normalize(img)
            processed_images.append(img)

        item['images'] = np.stack(processed_images, axis=0)  # [N_cam, H, W, 3]

        # 随机翻转
        if self.flip is not None:
            flipped_images = []
            targets = {
                'drivable_mask': item.get('drivable_mask'),
                'lane_mask': item.get('lane_mask'),
                'boxes_3d': item.get('boxes_3d'),
            }
            for img in processed_images:
                img, targets = self.flip(img, targets)
                flipped_images.append(img)
            item['images'] = np.stack(flipped_images, axis=0)
            item['drivable_mask'] = targets['drivable_mask']
            item['lane_mask'] = targets['lane_mask']
            item['boxes_3d'] = targets['boxes_3d']

        # 转换为 Tensor
        item['images'] = torch.from_numpy(item['images']).float().permute(0, 3, 1, 2)

        return item


class ValTransform:
    """验证/测试数据变换"""

    def __init__(
        self,
        image_size: Tuple[int, int] = (900, 1600),
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ):
        self.resize = ResizeImage(image_size)
        self.normalize = NormalizeImage(mean, std)

    def __call__(self, item: Dict) -> Dict:
        images = item['images']

        processed_images = []
        for img in images:
            img = self.resize(img)
            img = self.normalize(img)
            processed_images.append(img)

        item['images'] = torch.from_numpy(
            np.stack(processed_images, axis=0)
        ).float().permute(0, 3, 1, 2)

        return item
