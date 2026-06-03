"""
多尺度注意力模块
================
自主改进的多尺度注意力模块，对 BEV 特征进行多尺度增强与融合。
- 多尺度特征金字塔 (FPN / BiFPN)
- 跨尺度自注意力融合
- 自适应尺度权重学习

创新点：
  在标准 FPN 基础上引入自注意力机制，使不同尺度的 BEV 特征可以
  相互关注，学习全局上下文与局部细节的最优融合方式。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ============================================================================
# 跨尺度自注意力
# ============================================================================

class CrossScaleSelfAttention(nn.Module):
    """
    跨尺度自注意力
    —— 允许不同尺度的特征相互关注，学习最优的尺度间信息流动
    """

    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert self.head_dim * num_heads == channels

        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            feats: 多尺度特征列表 [feat_s0, feat_s1, feat_s2]
                   每个形状 [B, C, H_i, W_i]
        Returns:
            增强后的多尺度特征
        """
        B = feats[0].shape[0]
        C = self.channels

        # 展平各尺度特征
        flat_feats = []
        shapes = []
        for feat in feats:
            _, _, H, W = feat.shape
            shapes.append((H, W))
            flat_feats.append(feat.flatten(2).permute(0, 2, 1))  # [B, H*W, C]

        # 拼接所有尺度的 token
        tokens = torch.cat(flat_feats, dim=1)  # [B, sum(HW), C]

        # 自注意力
        tokens_norm = self.norm(tokens)
        qkv = self.qkv(tokens_norm).reshape(B, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, sum(HW), head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, num_heads, sum(HW), head_dim]
        out = out.transpose(1, 2).reshape(B, -1, C)  # [B, sum(HW), C]
        out = self.proj(out)
        out = tokens + out  # 残差连接

        # 拆分回各尺度
        results = []
        start = 0
        for H, W in shapes:
            n = H * W
            feat_out = out[:, start:start + n, :]
            feat_out = feat_out.permute(0, 2, 1).reshape(B, C, H, W)
            results.append(feat_out)
            start += n

        return results


# ============================================================================
# BiFPN 层
# ============================================================================

class BiFPNLayer(nn.Module):
    """
    BiFPN 单层 —— 带权重的双向特征金字塔
    参考: EfficientDet (Tan et al., CVPR 2020)
    """

    def __init__(self, channels: int, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales

        # Top-down 通路
        self.td_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
            )
            for _ in range(num_scales - 1)
        ])

        # Bottom-up 通路
        self.bu_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
            )
            for _ in range(num_scales - 1)
        ])

        # 可学习融合权重
        self.td_weights = nn.ParameterList([
            nn.Parameter(torch.ones(2), requires_grad=True)
            for _ in range(num_scales - 1)
        ])
        self.bu_weights = nn.ParameterList([
            nn.Parameter(torch.ones(2), requires_grad=True)
            for _ in range(num_scales - 1)
        ])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            feats: 从低分辨率到高分辨率 [P3, P4, P5]
        Returns:
            融合后的特征（相同顺序）
        """
        # ---- Top-down ----
        td_feats = [feats[-1]]  # 从最高层开始
        for i in range(self.num_scales - 2, -1, -1):
            up = F.interpolate(td_feats[-1], size=feats[i].shape[2:],
                               mode='bilinear', align_corners=False)
            w = F.relu(self.td_weights[i])
            w = w / (w.sum() + 1e-4)
            fused = w[0] * feats[i] + w[1] * up
            fused = self.td_convs[i](fused)
            td_feats.insert(0, fused)

        # ---- Bottom-up ----
        bu_feats = [td_feats[0]]
        for i in range(1, self.num_scales):
            down = F.max_pool2d(bu_feats[-1], kernel_size=3, stride=2, padding=1)
            w = F.relu(self.bu_weights[i - 1])
            w = w / (w.sum() + 1e-4)
            fused = w[0] * td_feats[i] + w[1] * down
            fused = self.bu_convs[i - 1](fused)
            bu_feats.append(fused)

        return bu_feats


# ============================================================================
# 多尺度注意力模块
# ============================================================================

class MultiScaleAttention(nn.Module):
    """
    多尺度注意力模块（自主改进版）
    —————————————————————————————
    结合 BiFPN 多尺度融合 + 跨尺度自注意力，增强 BEV 特征的多尺度表达能力。

    架构:
      Input BEV Features
          │
      ┌───▼────────────────────┐
      │  BiFPN 多尺度变换        │
      └───┬────────────────────┘
          │
      ┌───▼────────────────────┐
      │  跨尺度自注意力          │  ← 核心创新：不同尺度相互关注
      └───┬────────────────────┘
          │
      ┌───▼────────────────────┐
      │  自适应尺度聚合          │
      └───┬────────────────────┘
          │
      Enhanced BEV Features

    轻量化设计：
    - 使用深度可分离卷积减少参数
    - 可学习权重实现软特征选择
    """

    def __init__(
        self,
        in_channels: List[int] = None,
        out_channels: int = 256,
        num_scales: int = 3,
        num_heads: int = 8,
        num_bifpn_layers: int = 2,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        if in_channels is None:
            in_channels = [256, 512, 1024]
        self.num_scales = num_scales
        self.out_channels = out_channels

        # 输入通道对齐
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for in_ch in in_channels[:num_scales]
        ])

        # BiFPN 层堆叠
        self.bifpn_layers = nn.ModuleList([
            BiFPNLayer(out_channels, num_scales)
            for _ in range(num_bifpn_layers)
        ])

        # 跨尺度自注意力
        self.cross_scale_attn = CrossScaleSelfAttention(
            channels=out_channels,
            num_heads=num_heads,
            dropout=attention_dropout,
        )

        # 尺度聚合权重
        self.scale_weights = nn.Parameter(torch.ones(num_scales), requires_grad=True)

        # 输出卷积
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _resize_to_target(self, feats: List[torch.Tensor],
                          target_size: tuple) -> List[torch.Tensor]:
        """将所有特征调整到目标尺寸"""
        return [
            F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            for f in feats
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: BEV 特征 [B, N_bev, C] 或 [B, C, H_bev, W_bev]
        Returns:
            增强后的 BEV 特征 [B, C, H_bev, W_bev]
        """
        # 统一为 4D 格式
        if x.dim() == 3:
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            x = x.permute(0, 2, 1).reshape(B, C, H, W)

        B, C, H, W = x.shape

        # ---- 构建多尺度金字塔 ----
        pyramid = [x]  # 原始尺度
        for _ in range(self.num_scales - 1):
            pyramid.append(
                F.max_pool2d(pyramid[-1], kernel_size=3, stride=2, padding=1)
            )

        # 通道对齐
        pyramid = [
            proj(feat) for proj, feat in zip(self.input_proj, pyramid)
        ]

        # ---- BiFPN 多尺度融合 ----
        for bifpn_layer in self.bifpn_layers:
            pyramid = bifpn_layer(pyramid)

        # ---- 跨尺度自注意力 ----
        pyramid = self.cross_scale_attn(pyramid)

        # ---- 自适应尺度聚合 ----
        # 将所有尺度上采样到原始分辨率并加权融合
        w = F.softmax(self.scale_weights, dim=0)
        aggregated = torch.zeros_like(pyramid[0])
        for i, feat in enumerate(pyramid):
            up = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            aggregated = aggregated + w[i] * up

        # 输出卷积
        out = self.output_conv(aggregated)

        return out


def build_multi_scale_attention(config) -> MultiScaleAttention:
    """根据配置构建多尺度注意力模块"""
    from configs.bevformer_config import MultiScaleAttentionConfig

    cfg = config.multi_scale_attn if hasattr(config, 'multi_scale_attn') else config

    return MultiScaleAttention(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        num_scales=cfg.num_scales,
        num_heads=cfg.num_heads,
        attention_dropout=cfg.attention_dropout,
    )
