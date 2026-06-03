"""
BEVFormer 多任务感知模型配置
===============================
基于 BEVFormer 框架，融合双分支 ResNet 骨干网络、多尺度注意力模块，
同时输出 3D 检测框、可行驶区域与车道线。
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class BEVGridConfig:
    """BEV 网格参数"""
    # BEV 空间范围 (米) — 自车前后左右
    x_bound: Tuple[float, float, float] = (-51.2, 51.2, 0.8)   # (min, max, resolution)
    y_bound: Tuple[float, float, float] = (-51.2, 51.2, 0.8)
    z_bound: Tuple[float, float, float] = (-5.0, 3.0, 8.0)

    @property
    def bev_h(self) -> int:
        return int((self.x_bound[1] - self.x_bound[0]) / self.x_bound[2])

    @property
    def bev_w(self) -> int:
        return int((self.y_bound[1] - self.y_bound[0]) / self.y_bound[2])

    @property
    def bev_d(self) -> int:
        return int((self.z_bound[1] - self.z_bound[0]) / self.z_bound[2])


@dataclass
class BackboneConfig:
    """双分支 ResNet 骨干网络配置"""
    # 骨干类型
    backbone_type: str = "resnet50"          # resnet50 / resnet101
    pretrained: bool = True

    # 双分支参数
    dual_branch: bool = True                  # 是否启用双分支
    branch_channels: List[int] = None         # 各分支输出通道数

    # 注意力机制
    use_channel_attention: bool = True
    use_spatial_attention: bool = True
    attention_reduction: int = 16             # 通道注意力压缩比

    # 输出多尺度特征层
    out_indices: Tuple[int, ...] = (1, 2, 3)  # 输出 C3, C4, C5
    out_channels: int = 256                    # FPN 统一通道数

    def __post_init__(self):
        if self.branch_channels is None:
            self.branch_channels = [256, 512, 1024]


@dataclass
class BEVEncoderConfig:
    """BEV 特征变换编码器配置（BEVFormer 空间交叉注意力）"""
    # Transformer 参数
    num_layers: int = 6                       # 编码器层数
    num_heads: int = 8                        # 多头注意力头数
    embed_dims: int = 256                     # 嵌入维度
    num_points: int = 4                       # 每个查询的采样点数
    num_levels: int = 3                       # 多尺度特征层数
    num_cams: int = 6                         # 环视相机数量

    # 前馈网络
    ffn_dim: int = 1024
    dropout: float = 0.1

    # 位置编码
    use_camera_embeds: bool = True


@dataclass
class MultiScaleAttentionConfig:
    """多尺度注意力模块配置"""
    in_channels: List[int] = None             # 输入多尺度通道数
    out_channels: int = 256                   # 输出统一通道数
    num_scales: int = 3                       # 尺度数量

    # 注意力参数
    num_heads: int = 8
    attention_dropout: float = 0.1

    # FPN 类型
    fpn_type: str = "bifpn"                   # fpn / pafpn / bifpn

    def __post_init__(self):
        if self.in_channels is None:
            self.in_channels = [256, 512, 1024]


@dataclass
class TemporalFusionConfig:
    """时序融合配置"""
    use_temporal: bool = True
    num_history: int = 3                      # 历史帧数
    fusion_method: str = "concat"             # concat / attention


@dataclass
class DetectionHeadConfig:
    """3D 检测头配置（基于 CenterPoint）"""
    num_classes: int = 10                     # 检测类别数
    in_channels: int = 256

    # 检测头结构
    num_blocks: int = 2
    head_conv_channels: int = 64

    # 目标分配
    assigner: str = "hungarian"               # 匈牙利匹配
    topk_candidates: int = 100


@dataclass
class SegmentationHeadConfig:
    """分割头配置（可行驶区域 + 车道线）"""
    in_channels: int = 256
    num_drivable_classes: int = 2             # 背景 / 可行驶区域
    num_lane_classes: int = 2                 # 背景 / 车道线

    # 分割头结构
    hidden_channels: int = 128
    num_upsample: int = 2                     # 上采样次数


@dataclass
class BEVPerceptionConfig:
    """BEV 多任务感知总配置"""
    # 子模块配置
    bev_grid: BEVGridConfig = None
    backbone: BackboneConfig = None
    bev_encoder: BEVEncoderConfig = None
    multi_scale_attn: MultiScaleAttentionConfig = None
    temporal: TemporalFusionConfig = None
    detection_head: DetectionHeadConfig = None
    segmentation_head: SegmentationHeadConfig = None

    # 训练参数
    image_size: Tuple[int, int] = (900, 1600)  # (H, W)
    batch_size: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    max_epochs: int = 24
    warmup_epochs: int = 2

    # 损失权重
    loss_det_weight: float = 1.0
    loss_seg_drivable_weight: float = 0.5
    loss_seg_lane_weight: float = 0.5

    def __post_init__(self):
        if self.bev_grid is None:
            self.bev_grid = BEVGridConfig()
        if self.backbone is None:
            self.backbone = BackboneConfig()
        if self.bev_encoder is None:
            self.bev_encoder = BEVEncoderConfig()
        if self.multi_scale_attn is None:
            self.multi_scale_attn = MultiScaleAttentionConfig()
        if self.temporal is None:
            self.temporal = TemporalFusionConfig()
        if self.detection_head is None:
            self.detection_head = DetectionHeadConfig()
        if self.segmentation_head is None:
            self.segmentation_head = SegmentationHeadConfig()


# 默认配置实例
default_config = BEVPerceptionConfig()


def lightweight_config() -> BEVPerceptionConfig:
    """
    轻量化配置 — 显存友好，适合单卡训练和调试

    改动:
    - Backbone: ResNet18 (11M vs ResNet50 25M)
    - BEV Grid: 64×64 (原 128×128，减少 4× 显存)
    - BEV Encoder: embed_dims 128, layers 2 (原 256, 3)
    - Multi-Scale Attn: out_channels 128 (原 256)
    - Detection Head: in_channels 128, conv 32 (原 256, 64)
    - Segmentation Head: in_channels 128, hidden 64 (原 256, 128)
    - FPN: out_channels 128 (原 256)
    - Image size: 450×800 (原 900×1600)

    预估总参数: ~15M (原 ~50M), 显存降低约 75%
    """
    cfg = BEVPerceptionConfig()

    # -- BEV 网格: 缩小 4 倍 --
    cfg.bev_grid.x_bound = (-51.2, 51.2, 1.6)   # 64 格 (原 128)
    cfg.bev_grid.y_bound = (-51.2, 51.2, 1.6)
    cfg.image_size = (450, 800)                   # 降采样 (原 900,1600)

    # -- 骨干网络 --
    cfg.backbone.backbone_type = 'resnet18'
    cfg.backbone.out_channels = 128
    cfg.backbone.branch_channels = [128, 256, 512]

    # -- BEV 编码器 --
    cfg.bev_encoder.embed_dims = 128
    cfg.bev_encoder.num_layers = 2
    cfg.bev_encoder.num_heads = 4
    cfg.bev_encoder.ffn_dim = 512
    cfg.bev_encoder.num_points = 2

    # -- 多尺度注意力 --
    cfg.multi_scale_attn.out_channels = 128
    cfg.multi_scale_attn.num_heads = 4
    cfg.multi_scale_attn.in_channels = [128, 128, 128]

    # -- 检测头 --
    cfg.detection_head.in_channels = 128
    cfg.detection_head.head_conv_channels = 32

    # -- 分割头 --
    cfg.segmentation_head.in_channels = 128
    cfg.segmentation_head.hidden_channels = 64
    cfg.segmentation_head.num_upsample = 0   # 不扩大，直接输出

    return cfg


lightweight_config = lightweight_config()  # 实例化供直接 import
