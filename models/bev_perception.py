"""
BEV 多任务感知模型
==================
端到端的 BEV 视角多任务自动驾驶感知模型。

完整架构:
  环视相机图像 [B, N_cam, 3, H, W]
      │
  ┌───▼─────────────────────────────────────┐
  │  双分支 ResNet + 注意力骨干网络           │  ← 借鉴遥感变化检测
  │  (空间分支 + 语义分支 + CBAM + 跨分支融合) │
  └───┬─────────────────────────────────────┘
      │  多尺度图像特征
  ┌───▼─────────────────────────────────────┐
  │  BEV 特征变换编码器 (BEVFormer)           │
  │  空间交叉注意力 × N                       │
  └───┬─────────────────────────────────────┘
      │  BEV 特征
  ┌───▼─────────────────────────────────────┐
  │  时序融合模块 (可选)                      │
  │  时序自注意力 → 历史帧融合                │
  └───┬─────────────────────────────────────┘
      │
  ┌───▼─────────────────────────────────────┐
  │  多尺度注意力模块 (自主改进)              │
  │  BiFPN + 跨尺度自注意力                   │
  └───┬─────────────────────────────────────┘
      │  增强的 BEV 特征
  ┌───┴───────────────┬─────────────────────┐
  │                   │                     │
  ▼                   ▼                     ▼
┌──────────┐   ┌──────────────┐   ┌──────────────┐
│ 3D 检测头 │   │可行驶区域分割头│   │ 车道线分割头  │
│CenterPoint│   │   FCN 解码器  │   │  FCN 解码器   │
└────┬─────┘   └──────┬───────┘   └──────┬───────┘
     │                │                   │
     ▼                ▼                   ▼
  3D 检测框      可行驶区域图          车道线图
  (x,y,z,l,w,h,θ)  (H×W)              (H×W)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from .backbone import DualBranchResNet, build_backbone
from .bev_encoder import BEVFeatureEncoder, build_bev_encoder
from .multi_scale_attention import MultiScaleAttention, build_multi_scale_attention
from .temporal_fusion import TemporalFusion
from .detection_head import DetectionHead, decode_detections
from .segmentation_head import MultiTaskSegmentationHead


class BEVPerception(nn.Module):
    """
    BEV 多任务自动驾驶感知模型

    端到端地从环视相机图像预测 BEV 空间中的：
    1. 3D 目标检测框（类别、位置、尺寸、朝向、速度）
    2. 可行驶区域分割图
    3. 车道线分割图
    """

    def __init__(self, config=None):
        super().__init__()

        if config is None:
            from configs.bevformer_config import default_config
            config = default_config

        self.config = config

        # ---- 1. 双分支 ResNet 骨干网络 ----
        self.backbone = build_backbone(config)

        # ---- 2. BEV 特征变换编码器 ----
        self.bev_encoder = build_bev_encoder(config)

        # ---- 3. 时序融合模块 ----
        if config.temporal.use_temporal:
            self.temporal_fusion = TemporalFusion(
                embed_dims=config.bev_encoder.embed_dims,
                num_heads=config.bev_encoder.num_heads,
                num_history=config.temporal.num_history,
            )
        else:
            self.temporal_fusion = None

        # ---- 4. 多尺度注意力模块 ----
        self.multi_scale_attn = build_multi_scale_attention(config)

        # ---- 5. 任务头 ----
        self.detection_head = DetectionHead(
            in_channels=config.detection_head.in_channels,
            num_classes=config.detection_head.num_classes,
        )
        self.segmentation_head = MultiTaskSegmentationHead(
            in_channels=config.segmentation_head.in_channels,
            hidden_channels=config.segmentation_head.hidden_channels,
            num_drivable_classes=config.segmentation_head.num_drivable_classes,
            num_lane_classes=config.segmentation_head.num_lane_classes,
        )

        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        # 骨干网络使用预训练权重，其他模块已各自初始化
        pass

    def extract_image_features(
        self,
        images: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        提取多视图图像的多尺度特征

        Args:
            images: 环视图像 [B, N_cam, 3, H_img, W_img]
        Returns:
            多尺度特征列表
        """
        B, N_cam, C, H, W = images.shape

        # 合并 batch 和相机维度
        images = images.reshape(B * N_cam, C, H, W)

        # 双分支 ResNet 前向
        mlvl_feats = self.backbone(images)  # List[[B*N_cam, 256, H_i, W_i]]

        return mlvl_feats

    def forward(
        self,
        images: torch.Tensor,
        camera_params: Optional[torch.Tensor] = None,
        can_bus: Optional[torch.Tensor] = None,
        return_bev_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            images: 环视相机图像 [B, N_cam, 3, H_img, W_img]
            camera_params: 相机内外参 [B, N_cam, 16]
            can_bus: CAN 总线数据（用于时序对齐）[B, ...]
            return_bev_features: 是否返回中间 BEV 特征
        Returns:
            {
                'detections': {
                    'heatmap': [B, num_classes, H_bev, W_bev],
                    'size': [B, 3, H_bev, W_bev],
                    'offset': [B, 2, H_bev, W_bev],
                    'rotation': [B, 2, H_bev, W_bev],
                    'velocity': [B, 2, H_bev, W_bev],
                    'z': [B, 1, H_bev, W_bev],
                },
                'segmentation': {
                    'drivable': [B, 2, H_bev_out, W_bev_out],
                    'lane': [B, 2, H_bev_out, W_bev_out],
                },
                'bev_features': [B, C, H_bev, W_bev]  (if return_bev_features)
            }
        """
        B, N_cam, C_img, H_img, W_img = images.shape
        bev_h = self.config.bev_grid.bev_h
        bev_w = self.config.bev_grid.bev_w

        # ---- Step 1: 图像特征提取 ----
        mlvl_feats = self.extract_image_features(images)

        # ---- Step 2: BEV 特征变换 ----
        bev_features = self.bev_encoder(
            mlvl_feats=mlvl_feats,
            camera_params=camera_params,
            bev_h=bev_h,
            bev_w=bev_w,
        )  # [B, N_bev, C]

        # ---- Step 3: 时序融合 ----
        if self.temporal_fusion is not None:
            bev_features = self.temporal_fusion(bev_features, can_bus)

        # ---- Step 4: 多尺度注意力增强 ----
        bev_features_4d = bev_features.permute(0, 2, 1).reshape(
            B, self.config.bev_encoder.embed_dims, bev_h, bev_w
        )
        enhanced_bev = self.multi_scale_attn(bev_features_4d)  # [B, C, H_bev, W_bev]

        # ---- Step 5: 多任务预测 ----
        # 3D 检测
        detections = self.detection_head(enhanced_bev)

        # 分割（可行驶区域 + 车道线）
        segmentation = self.segmentation_head(enhanced_bev)

        outputs = {
            'detections': detections,
            'segmentation': segmentation,
        }

        if return_bev_features:
            outputs['bev_features'] = enhanced_bev

        return outputs

    def predict(
        self,
        images: torch.Tensor,
        camera_params: Optional[torch.Tensor] = None,
        score_threshold: float = 0.1,
        top_k: int = 100,
    ) -> Dict:
        """
        推理接口 —— 返回解码后的结构化预测结果

        Returns:
            {
                'boxes_3d': List[Dict],     # 3D 检测框列表
                'drivable_map': Tensor,     # 可行驶区域概率图
                'lane_map': Tensor,         # 车道线概率图
            }
        """
        outputs = self.forward(images, camera_params)

        # 解码 3D 检测框
        boxes_3d = decode_detections(
            outputs['detections'],
            bev_range=(
                self.config.bev_grid.x_bound[0],
                self.config.bev_grid.x_bound[1],
                self.config.bev_grid.x_bound[2],
            ),
            score_threshold=score_threshold,
            top_k=top_k,
        )

        # 分割结果
        drivable_map = torch.softmax(outputs['segmentation']['drivable'], dim=1)
        lane_map = torch.softmax(outputs['segmentation']['lane'], dim=1)

        return {
            'boxes_3d': boxes_3d,
            'drivable_map': drivable_map,
            'lane_map': lane_map,
        }

    def reset_temporal(self):
        """重置时序状态（新场景开始前调用）"""
        if self.temporal_fusion is not None:
            self.temporal_fusion.reset_queue()


def build_bev_perception(config_path: Optional[str] = None):
    """
    构建 BEV 多任务感知模型

    Args:
        config_path: 配置文件路径（可选）
    Returns:
        BEVPerception 模型实例
    """
    if config_path is not None:
        import yaml
        with open(config_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        from configs.bevformer_config import BEVPerceptionConfig
        config = BEVPerceptionConfig(**cfg_dict)
    else:
        from configs.bevformer_config import default_config
        config = default_config

    return BEVPerception(config)
