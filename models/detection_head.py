"""
3D 目标检测头
=============
基于 CenterPoint 范式的 3D 检测头，在 BEV 特征上进行目标检测。
- 热力图预测（类别 + 位置）
- 回归头（尺寸、朝向、高度、速度）
- 匈牙利匹配目标分配策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


# ============================================================================
# 检测头组件
# ============================================================================

class ConvBlock(nn.Module):
    """Conv + BN + ReLU 基本块"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DetectionHead(nn.Module):
    """
    3D 目标检测头

    输出:
    - heatmap: [B, num_classes, H, W] — 类别热力图
    - size: [B, 3, H, W] — 目标尺寸 (length, width, height)
    - offset: [B, 2, H, W] — 位置偏移
    - rotation: [B, 2, H, W] — 朝向角 (sin, cos)
    - velocity: [B, 2, H, W] — 速度 (vx, vy)
    - height_z: [B, 1, H, W] — 高度 (z轴)
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 10,
        num_blocks: int = 2,
        head_conv_channels: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # 共享特征提取
        shared_convs = []
        for i in range(num_blocks):
            in_ch = in_channels if i == 0 else head_conv_channels
            shared_convs.append(ConvBlock(in_ch, head_conv_channels))
        self.shared_conv = nn.Sequential(*shared_convs)

        # 分类分支 — 热力图
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(head_conv_channels, head_conv_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv_channels, num_classes, kernel_size=1),
        )

        # 回归分支
        self.reg_head = nn.Sequential(
            nn.Conv2d(head_conv_channels, head_conv_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 各回归输出
        self.size_head = nn.Conv2d(head_conv_channels, 3, kernel_size=1)       # l, w, h
        self.offset_head = nn.Conv2d(head_conv_channels, 2, kernel_size=1)     # dx, dy
        self.rot_head = nn.Conv2d(head_conv_channels, 2, kernel_size=1)        # sin, cos
        self.vel_head = nn.Conv2d(head_conv_channels, 2, kernel_size=1)        # vx, vy
        self.z_head = nn.Conv2d(head_conv_channels, 1, kernel_size=1)          # z

        self._init_weights()

    def _init_weights(self):
        """初始化检测头权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # heatmap 偏置初始化为负值（参考 CenterPoint）
        nn.init.constant_(self.heatmap_head[-1].bias, -2.19)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: BEV 特征 [B, C, H_bev, W_bev]
        Returns:
            检测输出字典
        """
        if x.dim() == 3:
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            x = x.permute(0, 2, 1).reshape(B, C, H, W)

        shared = self.shared_conv(x)

        # 分类
        heatmap = self.heatmap_head(shared)

        # 回归
        reg_features = self.reg_head(shared)

        size = self.size_head(reg_features)
        offset = self.offset_head(reg_features)
        rotation = self.rot_head(reg_features)
        velocity = self.vel_head(reg_features)
        z = self.z_head(reg_features)

        return {
            'heatmap': heatmap,        # [B, num_classes, H, W]
            'size': size,              # [B, 3, H, W]
            'offset': offset,          # [B, 2, H, W]
            'rotation': rotation,      # [B, 2, H, W]
            'velocity': velocity,      # [B, 2, H, W]
            'z': z,                    # [B, 1, H, W]
        }


# ============================================================================
# 检测结果解码
# ============================================================================

def decode_detections(
    predictions: Dict[str, torch.Tensor],
    bev_range: Tuple[float, float, float] = (-51.2, 51.2, 0.8),
    score_threshold: float = 0.1,
    top_k: int = 100,
) -> List[Dict[str, torch.Tensor]]:
    """
    从热力图解码 3D 检测框

    Args:
        predictions: 检测头输出字典
        bev_range: BEV 空间范围 (min, max, resolution)
        score_threshold: 置信度阈值
        top_k: 每帧最多保留的检测数
    Returns:
        检测结果列表 [{'boxes_3d': [N, 7], 'scores': [N], 'labels': [N]}, ...]
    """
    heatmap = predictions['heatmap']  # [B, num_classes, H, W]
    B, num_classes, H, W = heatmap.shape

    x_min, x_max, x_res = bev_range
    y_min, y_max, y_res = bev_range

    batch_results = []

    for b in range(B):
        scores_b = []
        boxes_b = []
        labels_b = []

        for cls_id in range(num_classes):
            hm = heatmap[b, cls_id]  # [H, W]
            hm = torch.sigmoid(hm)

            # 找峰值（NMS 简化：top-k）
            scores_flat = hm.flatten()
            top_scores, top_indices = torch.topk(scores_flat, min(top_k, scores_flat.numel()))

            for score, idx in zip(top_scores, top_indices):
                if score < score_threshold:
                    continue

                h_idx = idx // W
                w_idx = idx % W

                # 世界坐标 (BEV 网格 → 米)
                cx = x_min + (h_idx.float() + predictions['offset'][b, 0, h_idx, w_idx]) * x_res
                cy = y_min + (w_idx.float() + predictions['offset'][b, 1, h_idx, w_idx]) * y_res

                # 尺寸编码
                size = predictions['size'][b, :, h_idx, w_idx]  # [3]
                l, w, h_box = size[0], size[1], size[2]

                # 朝向角
                rot_sin = predictions['rotation'][b, 0, h_idx, w_idx]
                rot_cos = predictions['rotation'][b, 1, h_idx, w_idx]
                yaw = torch.atan2(rot_sin, rot_cos)

                # z 高度
                z = predictions['z'][b, 0, h_idx, w_idx]

                # 速度
                vx = predictions['velocity'][b, 0, h_idx, w_idx]
                vy = predictions['velocity'][b, 1, h_idx, w_idx]

                # 7-DOF 3D 框: [cx, cy, z, l, w, h, yaw, vx, vy]
                box = torch.tensor([cx, cy, z, l, w, h_box, yaw, vx, vy],
                                   device=heatmap.device)

                scores_b.append(score)
                boxes_b.append(box)
                labels_b.append(cls_id)

        if len(scores_b) > 0:
            scores_b = torch.stack(scores_b)
            boxes_b = torch.stack(boxes_b)
            labels_b = torch.tensor(labels_b, device=heatmap.device)

            # 按分数排序
            sorted_idx = torch.argsort(scores_b, descending=True)[:top_k]
            scores_b = scores_b[sorted_idx]
            boxes_b = boxes_b[sorted_idx]
            labels_b = labels_b[sorted_idx]
        else:
            scores_b = torch.zeros(0, device=heatmap.device)
            boxes_b = torch.zeros(0, 9, device=heatmap.device)
            labels_b = torch.zeros(0, dtype=torch.long, device=heatmap.device)

        batch_results.append({
            'boxes_3d': boxes_b,
            'scores': scores_b,
            'labels': labels_b,
        })

    return batch_results
