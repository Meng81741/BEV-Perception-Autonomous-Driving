"""
双分支 ResNet + 注意力机制骨干网络
====================================
借鉴遥感变化检测中的双分支架构，设计用于环视相机图像特征提取的骨干网络。
- 分支一（空间细节分支）：标准 ResNet 残差块，保留细粒度空间信息
- 分支二（语义分支）：引入空洞卷积与通道/空间注意力，增强语义表达能力
- 输出多尺度特征图供后续 BEV 变换使用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Tuple, Optional


# ============================================================================
# 注意力模块
# ============================================================================

class ChannelAttention(nn.Module):
    """
    通道注意力模块 (SENet 风格)
    —— 自适应学习通道间的重要性权重
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """
    空间注意力模块 (CBAM 风格)
    —— 自适应关注空间维度上的关键区域
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.conv(torch.cat([avg_out, max_out], dim=1))
        return x * self.sigmoid(attn)


class CBAM(nn.Module):
    """串联的通道 + 空间注意力"""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# ============================================================================
# 双分支模块
# ============================================================================

class DilatedBottleneck(nn.Module):
    """
    带空洞卷积的瓶颈残差块
    —— 在保持分辨率的同时扩大感受野，增强语义上下文
    """

    expansion: int = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 dilation: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class SemanticBranch(nn.Module):
    """
    语义分支 —— 在 ResNet 的 layer3/layer4 中用空洞卷积替换步长下采样，
    并加入 CBAM 注意力机制增强语义表达。
    """

    def __init__(self, backbone: nn.Module, replace_layers: List[str] = None):
        super().__init__()
        if replace_layers is None:
            replace_layers = ['layer3', 'layer4']

        # 复制骨干网络的前几层
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2

        # layer3 和 layer4 用空洞卷积版本
        self.layer3 = self._make_dilated_layer(backbone.layer3, dilation=2)
        self.layer4 = self._make_dilated_layer(backbone.layer4, dilation=4)

    @staticmethod
    def _make_dilated_layer(layer: nn.Sequential, dilation: int) -> nn.Sequential:
        """将 ResNet layer 中的步长替换为空洞卷积，兼容 BasicBlock 和 Bottleneck"""
        blocks = []
        for block in layer:
            # 判断块类型：Bottleneck 有 conv3，BasicBlock 只有 conv1+conv2
            is_bottleneck = hasattr(block, 'conv3')

            if is_bottleneck:
                # Bottleneck: 替换 conv2 为空洞卷积
                new_conv2 = nn.Conv2d(
                    block.conv2.in_channels,
                    block.conv2.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    bias=False,
                )
                block.conv2 = new_conv2
            else:
                # BasicBlock: 替换 conv1（BasicBlock 中 3×3 卷积名为 conv1）
                # 注意 torchvision 的 BasicBlock: conv1=3×3, conv2=3×3
                if hasattr(block, 'conv1'):
                    new_conv1 = nn.Conv2d(
                        block.conv1.in_channels,
                        block.conv1.out_channels,
                        kernel_size=3,
                        stride=1 if block.stride != 1 else block.stride,
                        padding=dilation,
                        dilation=dilation,
                        bias=False,
                    )
                    block.conv1 = new_conv1

            block.stride = 1

            # 修复 downsample 中的 stride
            if block.downsample is not None:
                ds_layers = []
                for ds_m in block.downsample:
                    if isinstance(ds_m, nn.Conv2d):
                        ds_layers.append(nn.Conv2d(
                            ds_m.in_channels, ds_m.out_channels,
                            kernel_size=1, stride=1, bias=(ds_m.bias is not None),
                        ))
                    elif isinstance(ds_m, nn.BatchNorm2d):
                        ds_layers.append(nn.BatchNorm2d(ds_m.num_features))
                if len(ds_layers) > 0:
                    block.downsample = nn.Sequential(*ds_layers)
            blocks.append(block)
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return [c2, c3, c4, c5]


class SpatialDetailBranch(nn.Module):
    """
    空间细节分支 —— 保持标准 ResNet 的下采样节奏，
    保留高分辨率的空间定位信息。
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return [c2, c3, c4, c5]


# ============================================================================
# 特征融合模块
# ============================================================================

class CrossBranchFusion(nn.Module):
    """
    跨分支特征融合 —— 将空间细节分支与语义分支的特征进行自适应融合
    使用门控机制学习两个分支的重要性权重
    """

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.Softmax(dim=1),
        )

    def forward(self, spatial_feat: torch.Tensor,
                semantic_feat: torch.Tensor) -> torch.Tensor:
        # 对齐尺寸
        if spatial_feat.shape[2:] != semantic_feat.shape[2:]:
            semantic_feat = F.interpolate(
                semantic_feat, size=spatial_feat.shape[2:], mode='bilinear', align_corners=False
            )

        concat = torch.cat([spatial_feat, semantic_feat], dim=1)
        weights = self.gate(concat)  # [B, 2, H, W]

        w_spatial = weights[:, 0:1, :, :]
        w_semantic = weights[:, 1:2, :, :]

        return spatial_feat * w_spatial + semantic_feat * w_semantic


# ============================================================================
# 双分支 ResNet 骨干网络
# ============================================================================

class DualBranchResNet(nn.Module):
    """
    双分支 ResNet + 注意力机制骨干网络

    架构:
      输入图像 (N_cam, 3, H, W)
           │
      ┌────┴────┐
      │         │
  空间分支   语义分支 (空洞卷积 + CBAM)
      │         │
      └────┬────┘
           │
      跨分支融合 (门控)
           │
      FPN 多尺度输出
           │
      [C3, C4, C5] 多尺度特征

    参考:
    - 遥感变化检测中的双分支架构 (e.g. ChangeNet, BIT)
    - CBAM: Convolutional Block Attention Module (Woo et al., ECCV 2018)
    """

    def __init__(
        self,
        backbone_type: str = 'resnet50',
        pretrained: bool = True,
        out_indices: Tuple[int, ...] = (1, 2, 3),
        out_channels: int = 256,
        use_attention: bool = True,
        attention_reduction: int = 16,
    ):
        super().__init__()
        self.out_indices = out_indices

        # ---- 构建基础 ResNet ----
        if backbone_type == 'resnet18':
            backbone_fn = models.resnet18
            self.layer_channels = [64, 128, 256, 512]
        elif backbone_type == 'resnet34':
            backbone_fn = models.resnet34
            self.layer_channels = [64, 128, 256, 512]
        elif backbone_type == 'resnet50':
            backbone_fn = models.resnet50
            self.layer_channels = [256, 512, 1024, 2048]
        elif backbone_type == 'resnet101':
            backbone_fn = models.resnet101
            self.layer_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")

        # 空间分支
        # ResNet18/34 不支持 replace_stride_with_dilation 参数
        if backbone_type in ('resnet18', 'resnet34'):
            spatial_backbone = backbone_fn(pretrained=pretrained)
            semantic_backbone = backbone_fn(pretrained=pretrained)
        else:
            spatial_backbone = backbone_fn(pretrained=pretrained, replace_stride_with_dilation=[False, False, False])
            semantic_backbone = backbone_fn(pretrained=pretrained, replace_stride_with_dilation=[False, False, False])

        self.spatial_branch = SpatialDetailBranch(spatial_backbone)

        # 语义分支 (共享 conv1-layer2，使用空洞卷积的 layer3-layer4)
        self.semantic_branch = SemanticBranch(semantic_backbone)

        # ---- 注意力模块 ----
        if use_attention:
            self.cbam_layers = nn.ModuleDict({
                f'cbam_{i}': CBAM(self.layer_channels[i], attention_reduction)
                for i in range(4)
            })
        else:
            self.cbam_layers = None

        # ---- 跨分支融合 ----
        self.fusion_layers = nn.ModuleDict({
            f'fusion_{i}': CrossBranchFusion(self.layer_channels[i])
            for i in range(4)
        })

        # ---- FPN 颈部 ----
        self.fpn = self._build_fpn(out_channels)

        self._init_weights()

    def _build_fpn(self, out_channels: int) -> nn.ModuleDict:
        """构建 FPN 特征金字塔"""
        in_channels_list = [self.layer_channels[i] for i in self.out_indices]

        lateral_convs = nn.ModuleDict()
        output_convs = nn.ModuleDict()

        for i, in_ch in enumerate(in_channels_list):
            lateral_convs[f'lateral_{i}'] = nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False)
            output_convs[f'output_{i}'] = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        return nn.ModuleDict({
            'lateral': lateral_convs,
            'output': output_convs,
        })

    def _init_weights(self):
        """初始化新增层的权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m not in [mod for mod in self.spatial_branch.modules()
                             if isinstance(mod, nn.Conv2d)]:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: 多视图图像特征 [B * N_cam, 3, H, W]
        Returns:
            多尺度特征列表 [feat_C3, feat_C4, feat_C5]
            每个特征形状: [B * N_cam, 256, H_i, W_i]
        """
        # 双分支前向
        spatial_feats = self.spatial_branch(x)   # [C2, C3, C4, C5]
        semantic_feats = self.semantic_branch(x)  # [C2, C3, C4, C5]

        # 应用注意力 + 融合
        fused_feats = []
        for i in range(4):
            sf = spatial_feats[i]
            smf = semantic_feats[i]

            if self.cbam_layers is not None:
                sf = self.cbam_layers[f'cbam_{i}'](sf)
                smf = self.cbam_layers[f'cbam_{i}'](smf)

            fused = self.fusion_layers[f'fusion_{i}'](sf, smf)
            fused_feats.append(fused)

        # FPN 多尺度输出
        # 只取 out_indices 指定的层
        selected = [fused_feats[i] for i in self.out_indices]

        # Top-down FPN
        lateral = self.fpn['lateral']
        output = self.fpn['output']
        num_levels = len(selected)

        results = []
        prev = None
        for i in range(num_levels - 1, -1, -1):
            lat = lateral[f'lateral_{i}'](selected[i])
            if prev is not None:
                prev = F.interpolate(prev, size=lat.shape[2:], mode='bilinear', align_corners=False)
                lat = lat + prev
            out = output[f'output_{i}'](lat)
            results.insert(0, out)
            prev = lat

        return results


# ============================================================================
# 便捷构建函数
# ============================================================================

def build_backbone(config) -> DualBranchResNet:
    """根据配置构建骨干网络"""
    from configs.bevformer_config import BackboneConfig

    cfg = config.backbone if hasattr(config, 'backbone') else config

    return DualBranchResNet(
        backbone_type=cfg.backbone_type,
        pretrained=cfg.pretrained,
        out_indices=cfg.out_indices,
        out_channels=cfg.out_channels,
        use_attention=cfg.use_channel_attention or cfg.use_spatial_attention,
        attention_reduction=cfg.attention_reduction,
    )
