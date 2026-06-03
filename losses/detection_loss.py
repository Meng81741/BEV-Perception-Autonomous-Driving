"""
检测损失函数
============
基于 CenterPoint 的 3D 检测损失函数集合。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math


class FocalLoss(nn.Module):
    """
    Focal Loss 用于热力图分类
    —— 解决正负样本不平衡问题
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: [B, C, H, W] 预测热力图 (sigmoid 后)
            target: [B, C, H, W] 目标热力图 (高斯平滑)
        """
        pos_mask = (target == 1).float()
        neg_mask = (target < 1).float()

        # 正样本损失
        pos_loss = -pos_mask * torch.log(pred + 1e-6) * (1 - pred) ** self.alpha

        # 负样本损失（降权）
        neg_loss = -neg_mask * torch.log(1 - pred + 1e-6) * pred ** self.alpha * (1 - target) ** self.beta

        loss = pos_loss + neg_loss

        if self.reduction == 'mean':
            num_pos = pos_mask.sum().clamp(min=1)
            return loss.sum() / num_pos
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class L1Loss(nn.Module):
    """Smooth L1 回归损失"""

    def __init__(self, beta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(pred, target, beta=self.beta, reduction=self.reduction)


class DetectionLoss(nn.Module):
    """
    3D 检测总损失

    损失组成:
    - heatmap_loss: Focal Loss (分类 + 定位)
    - size_loss: L1 Loss (尺寸回归)
    - offset_loss: L1 Loss (位置偏移)
    - rotation_loss: L1 Loss (朝向角 sin/cos)
    - velocity_loss: L1 Loss (速度)
    - z_loss: L1 Loss (高度)
    """

    def __init__(
        self,
        heatmap_weight: float = 1.0,
        size_weight: float = 0.2,
        offset_weight: float = 0.2,
        rotation_weight: float = 0.2,
        velocity_weight: float = 0.1,
        z_weight: float = 0.2,
    ):
        super().__init__()
        self.heatmap_loss_fn = FocalLoss()
        self.reg_loss_fn = L1Loss()

        self.heatmap_weight = heatmap_weight
        self.size_weight = size_weight
        self.offset_weight = offset_weight
        self.rotation_weight = rotation_weight
        self.velocity_weight = velocity_weight
        self.z_weight = z_weight

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: 检测头输出
            targets: 目标标签
        Returns:
            各项损失字典
        """
        B = predictions['heatmap'].shape[0]

        # ---- Heatmap Loss ----
        pred_heatmap = torch.sigmoid(predictions['heatmap'])
        target_heatmap = targets['heatmap']
        loss_heatmap = self.heatmap_loss_fn(pred_heatmap, target_heatmap)

        # ---- 仅对正样本计算回归损失 ----
        pos_mask = targets.get('reg_mask', target_heatmap.max(dim=1, keepdim=True)[0] > 0.5)

        # Size Loss
        loss_size = self._masked_loss(
            predictions['size'], targets['size'], pos_mask, self.reg_loss_fn
        )

        # Offset Loss
        loss_offset = self._masked_loss(
            predictions['offset'], targets['offset'], pos_mask, self.reg_loss_fn
        )

        # Rotation Loss
        loss_rotation = self._masked_loss(
            predictions['rotation'], targets['rotation'], pos_mask, self.reg_loss_fn
        )

        # Velocity Loss
        loss_velocity = self._masked_loss(
            predictions['velocity'], targets['velocity'], pos_mask, self.reg_loss_fn
        )

        # Z Loss
        loss_z = self._masked_loss(
            predictions['z'], targets['z'], pos_mask, self.reg_loss_fn
        )

        # ---- 总损失 ----
        total_loss = (
            self.heatmap_weight * loss_heatmap +
            self.size_weight * loss_size +
            self.offset_weight * loss_offset +
            self.rotation_weight * loss_rotation +
            self.velocity_weight * loss_velocity +
            self.z_weight * loss_z
        )

        return {
            'det_loss': total_loss,
            'det_heatmap': loss_heatmap,
            'det_size': loss_size,
            'det_offset': loss_offset,
            'det_rotation': loss_rotation,
            'det_velocity': loss_velocity,
            'det_z': loss_z,
        }

    @staticmethod
    def _masked_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        loss_fn: nn.Module,
    ) -> torch.Tensor:
        """对正样本区域计算损失"""
        # 确保 mask 与 pred 维度匹配: 沿通道维度广播
        while mask.dim() < pred.dim():
            mask = mask.unsqueeze(1)
        if mask.shape[1] == 1 and pred.shape[1] > 1:
            mask = mask.expand(-1, pred.shape[1], -1, -1)

        pred_masked = pred[mask.bool()]
        target_masked = target[mask.bool()]

        if pred_masked.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        return loss_fn(pred_masked, target_masked)
