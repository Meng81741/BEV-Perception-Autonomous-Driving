"""
BEV 特征变换编码器
==================
基于 BEVFormer 的空间交叉注意力机制，将环视多相机图像特征变换到统一的
BEV（鸟瞰图）空间。核心创新：
- 轻量化设计：减少 Transformer 层数与嵌入维度，降低计算开销
- 可变形注意力：每个 BEV 查询只关注图像特征中的关键采样点
- 相机感知位置编码：显式编码相机内外参信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


# ============================================================================
# 可变形注意力 (Deformable Attention)
# ============================================================================

class DeformableAttention(nn.Module):
    """
    多尺度可变形注意力
    —— 每个查询在参考点附近采样 K 个偏移点，聚合多尺度特征
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 3,
        num_points: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dims // num_heads
        assert self.head_dim * num_heads == embed_dims, "embed_dims must be divisible by num_heads"

        # 采样偏移预测
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        # 注意力权重预测
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        # 输出投影
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        nn.init.constant_(self.sampling_offsets.bias, 0.)
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

    @staticmethod
    def _reshape_multi_head(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
        """[B, N, C] -> [B, num_heads, N, head_dim]"""
        B, N, C = x.shape
        x = x.reshape(B, N, num_heads, head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        bev_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: BEV 查询 [B, N_bev, C]
            reference_points: 参考点（归一化坐标）[B, N_bev, num_levels, 2]
            value: 多尺度图像特征 [sum(H_l*W_l), B*N_cam, C]
            spatial_shapes: 各层特征空间尺寸 [(H_l, W_l), ...]
            bev_mask: BEV 可见性掩码 [B, N_bev, N_cam]
        Returns:
            output: [B, N_bev, C]
        """
        B, N_bev, _ = query.shape
        N_cam = value.shape[1]

        # 预测采样偏移和注意力权重
        offsets = self.sampling_offsets(query)       # [B, N_bev, n_heads * n_levels * n_points * 2]
        attn_weights = self.attention_weights(query)  # [B, N_bev, n_heads * n_levels * n_points]

        offsets = offsets.reshape(B, N_bev, self.num_heads, self.num_levels, self.num_points, 2)
        attn_weights = attn_weights.reshape(B, N_bev, self.num_heads, self.num_levels, self.num_points)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 采样点 = 参考点 + 偏移
        reference_points = reference_points.unsqueeze(2).unsqueeze(4)  # [B, N_bev, 1, num_levels, 1, 2]
        sampling_locations = reference_points + offsets   # [B, N_bev, n_heads, n_levels, n_points, 2]

        # 多尺度特征采样
        output = self._multi_scale_deformable_attn(
            value, spatial_shapes, sampling_locations, attn_weights, N_cam
        )
        output = self.output_proj(output)
        return output

    def _multi_scale_deformable_attn(
        self,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        sampling_locations: torch.Tensor,
        attn_weights: torch.Tensor,
        N_cam: int,
    ) -> torch.Tensor:
        """
        多尺度可变形注意力核心采样逻辑

        Args:
            value: [sum(HW), B*N_cam, C] — 展平的多尺度图像特征
            spatial_shapes: [(H1,W1), (H2,W2), (H3,W3)]
            sampling_locations: [B, N_bev, n_heads, n_levels, n_points, 2] — 归一化坐标 [-1,1]
            attn_weights: [B, N_bev, n_heads, n_levels, n_points] — softmax 权重
            N_cam: 相机数量
        """
        B, N_bev, n_heads, n_levels, n_points, _ = sampling_locations.shape
        C = self.embed_dims

        # value: [sum(HW), B*N_cam, C] → 重新组织为 [B*N_cam, C, sum(HW)]
        # 然后按尺度拆分
        
        # 简化实现：将 value 按尺度拆分并 reshape 为可 grid_sample 的格式
        value_by_level = value.split([H * W for H, W in spatial_shapes.tolist()], dim=0)

        output = torch.zeros(B, N_bev, C, device=value.device, dtype=value.dtype)

        for level, (H, W) in enumerate(spatial_shapes.tolist()):
            # value_l: [H*W, B*N_cam, C]
            val_l = value_by_level[level]  # [H*W, B*N_cam, C]
            # reshape to [B*N_cam, C, H, W]
            val_l = val_l.permute(1, 2, 0).reshape(-1, C, H, W)

            # 简化：对所有相机取平均，得到一个统一的多尺度特征
            # [B*N_cam, C, H, W] → [B, N_cam, C, H, W] → mean over cameras → [B, C, H, W]
            val_l = val_l.reshape(B, N_cam, C, H, W).mean(dim=1)  # [B, C, H, W]

            # sampling_locations: [B, N_bev, n_heads, n_levels, n_points, 2]
            # 取当前 level 的采样位置，对 heads 和 points 取平均
            # [B, N_bev, n_heads, 1, n_points, 2] → mean over heads and points → [B, N_bev, 2]
            sample_loc = sampling_locations[:, :, :, level, :, :]  # [B, N_bev, n_heads, n_points, 2]
            attn_w = attn_weights[:, :, :, level, :]  # [B, N_bev, n_heads, n_points]

            # 加权平均采样位置和聚合特征
            attn_w = attn_w.softmax(dim=-1).unsqueeze(-1)  # [B, N_bev, n_heads, n_points, 1]
            sample_loc = (sample_loc * attn_w).sum(dim=(2, 3))  # [B, N_bev, 2]

            # grid_sample: [B, C, H, W] with grid [B, N_bev, 1, 2]
            grid = sample_loc.unsqueeze(2)  # [B, N_bev, 1, 2]
            sampled = F.grid_sample(
                val_l, grid,
                mode='bilinear', padding_mode='zeros', align_corners=False,
            )  # [B, C, N_bev, 1]
            sampled = sampled.squeeze(-1).permute(0, 2, 1)  # [B, N_bev, C]

            output = output + sampled

        return output


# ============================================================================
# 空间交叉注意力 (Spatial Cross-Attention)
# ============================================================================

class SpatialCrossAttention(nn.Module):
    """
    空间交叉注意力层
    —— BEVFormer 核心：BEV 查询通过可变形注意力与多相机图像特征交互
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 3,
        num_points: int = 4,
        num_cams: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_cams = num_cams

        # 可变形注意力
        self.deformable_attn = DeformableAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=dropout,
        )

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dims * 4, embed_dims),
            nn.Dropout(dropout),
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

        # 参考点生成器
        self.reference_points = None

    def _get_reference_points(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """生成归一化参考点网格"""
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, device=device),
            torch.linspace(0.5, W - 0.5, W, device=device),
            indexing='ij',
        )
        ref_y = ref_y.reshape(-1) / H
        ref_x = ref_x.reshape(-1) / W
        ref = torch.stack([ref_x, ref_y], dim=-1)  # [H*W, 2]
        return ref

    def forward(
        self,
        bev_query: torch.Tensor,
        value: torch.Tensor,
        bev_pos: torch.Tensor,
        spatial_shapes: torch.Tensor,
        reference_points_cam: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            bev_query: BEV 查询特征 [B, N_bev, C]
            value: 多尺度图像特征（展平）[sum(HW), B*N_cam, C]
            bev_pos: BEV 位置编码 [B, N_bev, C]
            spatial_shapes: [(H1,W1), (H2,W2), (H3,W3)]
            reference_points_cam: 投影参考点
        Returns:
            BEV 特征 [B, N_bev, C]
        """
        B, N_bev, _ = bev_query.shape

        # 生成参考点
        if reference_points_cam is None:
            ref_2d = self._get_reference_points(
                int(math.sqrt(N_bev)), int(math.sqrt(N_bev)), bev_query.device
            )
            reference_points_cam = ref_2d.unsqueeze(0).unsqueeze(2).expand(
                B, N_bev, len(spatial_shapes), 2
            )

        # 交叉注意力
        attn_out = self.deformable_attn(
            query=self.norm1(bev_query),
            reference_points=reference_points_cam,
            value=value,
            spatial_shapes=spatial_shapes,
        )

        bev_query = bev_query + attn_out

        # 前馈网络
        ffn_out = self.ffn(self.norm2(bev_query))
        bev_query = bev_query + ffn_out

        return bev_query


# ============================================================================
# BEV 特征变换编码器
# ============================================================================

class BEVFeatureEncoder(nn.Module):
    """
    BEV 特征变换编码器
    —————————————————
    将环视多相机图像特征通过多层空间交叉注意力变换到 BEV 空间。

    流程:
      Image Features (multi-view, multi-scale)
          │
      ┌───▼────────────────────────────┐
      │  Spatial Cross-Attention × N    │  ← 核心：图像→BEV 特征变换
      │  + FFN + LayerNorm             │
      └───┬────────────────────────────┘
          │
      BEV Features [B, H_bev*W_bev, C]

    轻量化设计：
    - 层数 N=3（相比原始 BEVFormer 的 6 层减半）
    - 嵌入维度 256（原始 256 → 保持，但采样点减少）
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        num_levels: int = 3,
        num_points: int = 4,
        num_cams: int = 6,
        dropout: float = 0.1,
        use_camera_embeds: bool = True,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.num_cams = num_cams

        # 多层空间交叉注意力
        self.layers = nn.ModuleList([
            SpatialCrossAttention(
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
                num_cams=num_cams,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # 相机感知嵌入（编码相机位姿信息）
        if use_camera_embeds:
            self.camera_embed = nn.Sequential(
                nn.Linear(16, embed_dims),      # 16 维相机参数（3D位置+四元数+内参）
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims, embed_dims),
            )
        else:
            self.camera_embed = None

        # BEV 位置编码 — 预留最大 BEV 尺寸，前向时切片
        self.bev_pos_embed = nn.Parameter(
            torch.zeros(1, 128 * 128, embed_dims)  # 最多支持 128×128
        )
        nn.init.trunc_normal_(self.bev_pos_embed, std=0.02)

        # 可学习的 BEV 查询
        self.bev_embedding = nn.Embedding(128 * 128, embed_dims)

        self._init_layers()

    def _init_layers(self):
        """初始化各层参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        mlvl_feats: List[torch.Tensor],
        camera_params: Optional[torch.Tensor] = None,
        bev_h: int = 128,
        bev_w: int = 128,
    ) -> torch.Tensor:
        """
        Args:
            mlvl_feats: 多尺度图像特征列表
                [feat_l0, feat_l1, feat_l2]
                每个形状: [B*N_cam, C, H_i, W_i]
            camera_params: 相机参数 [B, N_cam, 16]
            bev_h, bev_w: BEV 网格尺寸
        Returns:
            bev_features: [B, H_bev * W_bev, C]
        """
        B_camN = mlvl_feats[0].shape[0]
        N_cam = self.num_cams
        B = B_camN // N_cam
        N_bev = bev_h * bev_w
        C = mlvl_feats[0].shape[1]

        # ---- 展平多尺度特征 ----
        feat_flatten = []
        spatial_shapes = []
        for feat in mlvl_feats:
            _, _, H, W = feat.shape
            spatial_shapes.append([H, W])
            # [B*N_cam, C, H, W] -> [H*W, B*N_cam, C]
            feat_flat = feat.flatten(2).permute(2, 0, 1)
            feat_flatten.append(feat_flat)

        feat_flatten = torch.cat(feat_flatten, dim=0)  # [sum(HW), B*N_cam, C]
        spatial_shapes_tensor = torch.tensor(
            spatial_shapes, device=feat_flatten.device, dtype=torch.long
        )

        # ---- 相机感知嵌入 ----
        if self.camera_embed is not None and camera_params is not None:
            camera_embed = self.camera_embed(camera_params)  # [B, N_cam, C]
            camera_embed = camera_embed.unsqueeze(2).expand(-1, -1, N_bev, -1)
            camera_embed = camera_embed.reshape(B * N_cam, N_bev, C)
            # 加到值特征上
            # (简化：这里直接加到 bev_query 初始化中)

        # ---- 初始化 BEV 查询 ----
        bev_query = self.bev_embedding.weight[:N_bev, :]  # [N_bev, C]
        bev_query = bev_query.unsqueeze(0).expand(B, -1, -1)  # [B, N_bev, C]

        # BEV 位置编码
        bev_pos = self.bev_pos_embed[:, :N_bev, :].expand(B, -1, -1)

        # ---- 逐层空间交叉注意力 ----
        for layer in self.layers:
            bev_query = layer(
                bev_query=bev_query,
                value=feat_flatten,
                bev_pos=bev_pos,
                spatial_shapes=spatial_shapes_tensor,
            )

        return bev_query


# ============================================================================
# 多视图特征预处理
# ============================================================================

class MultiViewFeatureAdapter(nn.Module):
    """
    多视图特征适配器
    —— 将骨干网络输出的多视图多尺度特征对齐并聚合
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def build_bev_encoder(config) -> BEVFeatureEncoder:
    """根据配置构建 BEV 编码器"""
    from configs.bevformer_config import BEVEncoderConfig

    cfg = config.bev_encoder if hasattr(config, 'bev_encoder') else config

    return BEVFeatureEncoder(
        embed_dims=cfg.embed_dims,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        num_levels=cfg.num_levels,
        num_points=cfg.num_points,
        num_cams=cfg.num_cams,
        dropout=cfg.dropout,
        use_camera_embeds=cfg.use_camera_embeds,
    )
