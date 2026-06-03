"""
分割头（可行驶区域 + 车道线）
==============================
在 BEV 特征上进行语义分割，输出：
- 可行驶区域 (drivable area)：二分类分割
- 车道线 (lane): 二分类分割

采用轻量级 FCN 解码器 + 跳跃连接设计。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


# ============================================================================
# 轻量分割解码器
# ============================================================================

class ConvBNReLU(nn.Module):
    """Conv + BN + ReLU 单元"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleBlock(nn.Module):
    """上采样 + 卷积块"""

    def __init__(self, in_ch: int, out_ch: int, scale_factor: int = 2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = ConvBNReLU(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        return self.conv(x)


class SegmentationDecoder(nn.Module):
    """
    轻量级分割解码器
    —— 逐步上采样 BEV 特征到目标分辨率
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int,
                 num_upsample: int = 2):
        super().__init__()
        self.num_upsample = num_upsample

        layers = []
        ch = in_channels
        for i in range(num_upsample):
            next_ch = hidden_channels // (2 ** (num_upsample - 1 - i))
            layers.append(UpsampleBlock(ch, max(next_ch, 32)))
            ch = max(next_ch, 32)

        # 最终分类层
        layers.append(
            nn.Conv2d(ch, num_classes, kernel_size=1)
        )

        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return self.decoder(x)


# ============================================================================
# 多任务分割头
# ============================================================================

class MultiTaskSegmentationHead(nn.Module):
    """
    多任务分割头
    ————————————
    在共享特征基础上，使用两个独立解码器分别预测：
    1. 可行驶区域 (Drivable Area)
    2. 车道线 (Lane)

    架构:
      BEV Features [B, C, H, W]
          │
      ┌───▼────────────────────┐
      │  共享特征提取            │
      └───┬────────────────────┘
          │
      ┌───┴───────────┐
      │               │
  可行驶区域解码器   车道线解码器
      │               │
      ▼               ▼
   Drivable Map    Lane Map
    [B,2,H,W]      [B,2,H,W]
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 128,
        num_drivable_classes: int = 2,
        num_lane_classes: int = 2,
        num_upsample: int = 2,
    ):
        super().__init__()

        # 共享特征提取
        self.shared_conv = nn.Sequential(
            ConvBNReLU(in_channels, hidden_channels),
            ConvBNReLU(hidden_channels, hidden_channels),
        )

        # 可行驶区域解码器
        self.drivable_decoder = SegmentationDecoder(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_classes=num_drivable_classes,
            num_upsample=num_upsample,
        )

        # 车道线解码器
        self.lane_decoder = SegmentationDecoder(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_classes=num_lane_classes,
            num_upsample=num_upsample,
        )

        # 注意力引导融合（利用车道线和可行驶区域的关联性）
        self.cross_attention = nn.ModuleDict({
            'drivable_to_lane': nn.Sequential(
                nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=1),
                nn.Sigmoid(),
            ),
            'lane_to_drivable': nn.Sequential(
                nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=1),
                nn.Sigmoid(),
            ),
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: BEV 特征 [B, C, H, W] or [B, N, C]
        Returns:
            {
                'drivable': [B, num_drivable_classes, H', W'],
                'lane': [B, num_lane_classes, H', W']
            }
        """
        if x.dim() == 3:
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            x = x.permute(0, 2, 1).reshape(B, C, H, W)

        # 共享特征
        shared = self.shared_conv(x)

        # 任务间交叉引导
        drivable_feat = self.drivable_decoder.decoder[:-1](shared)
        lane_feat = self.lane_decoder.decoder[:-1](shared)

        # 互相增强
        drivable_gate = self.cross_attention['lane_to_drivable'](
            torch.cat([drivable_feat, lane_feat], dim=1)
        )
        lane_gate = self.cross_attention['drivable_to_lane'](
            torch.cat([drivable_feat, lane_feat], dim=1)
        )

        drivable_feat = drivable_feat * drivable_gate
        lane_feat = lane_feat * lane_gate

        # 最终分类
        drivable_pred = self.drivable_decoder.decoder[-1](drivable_feat)
        lane_pred = self.lane_decoder.decoder[-1](lane_feat)

        return {
            'drivable': drivable_pred,
            'lane': lane_pred,
        }
