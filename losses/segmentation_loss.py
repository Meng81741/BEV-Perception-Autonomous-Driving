"""
分割损失函数
============
可行驶区域和车道线分割的损失函数。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class DiceLoss(nn.Module):
    """
    Dice Loss
    —— 优化分割的区域重叠度
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, H, W] 预测 logits
            target: [B, H, W] 目标标签 (long)
        """
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        pred_softmax = F.softmax(pred, dim=1)

        # 对每个类别计算 Dice
        intersection = (pred_softmax * target_one_hot).sum(dim=(2, 3))
        union = pred_softmax.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    """
    分割总损失 = CrossEntropy + Dice

    损失组成:
    - ce_loss: 交叉熵损失（像素级分类）
    - dice_loss: Dice 损失（区域重叠）
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice_loss_fn = DiceLoss()
        self.ce_loss_fn = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=255,
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: [B, C, H, W] 预测 logits
            target: [B, H, W] 目标标签 (long)
        Returns:
            损失字典
        """
        ce_loss = self.ce_loss_fn(pred, target)
        dice_loss = self.dice_loss_fn(pred, target)

        total = self.ce_weight * ce_loss + self.dice_weight * dice_loss

        return {
            'total': total,
            'ce': ce_loss,
            'dice': dice_loss,
        }


class MultiTaskSegmentationLoss(nn.Module):
    """
    多任务分割损失
    —— 同时计算可行驶区域和车道线的分割损失
    """

    def __init__(
        self,
        drivable_ce_weight: float = 1.0,
        drivable_dice_weight: float = 1.0,
        lane_ce_weight: float = 1.0,
        lane_dice_weight: float = 1.0,
        drivable_weight: float = 1.0,
        lane_weight: float = 1.0,
    ):
        super().__init__()
        self.drivable_loss_fn = SegmentationLoss(
            ce_weight=drivable_ce_weight,
            dice_weight=drivable_dice_weight,
        )
        self.lane_loss_fn = SegmentationLoss(
            ce_weight=lane_ce_weight,
            dice_weight=lane_dice_weight,
        )
        self.drivable_weight = drivable_weight
        self.lane_weight = lane_weight

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: {
                'drivable': [B, 2, Hp, Wp],
                'lane': [B, 2, Hp, Wp],
            }
            targets: {
                'drivable_mask': [B, Ht, Wt],
                'lane_mask': [B, Ht, Wt],
            }
        """
        import torch.nn.functional as F

        # 对齐预测和目标的尺寸
        pred_drivable = predictions['drivable']
        pred_lane = predictions['lane']
        target_drivable = targets['drivable_mask']
        target_lane = targets['lane_mask']

        if pred_drivable.shape[-2:] != target_drivable.shape[-2:]:
            pred_drivable = F.interpolate(
                pred_drivable, size=target_drivable.shape[-2:],
                mode='bilinear', align_corners=False,
            )
            pred_lane = F.interpolate(
                pred_lane, size=target_lane.shape[-2:],
                mode='bilinear', align_corners=False,
            )

        drivable_losses = self.drivable_loss_fn(pred_drivable, target_drivable)
        lane_losses = self.lane_loss_fn(pred_lane, target_lane)

        total_seg_loss = (
            self.drivable_weight * drivable_losses['total'] +
            self.lane_weight * lane_losses['total']
        )

        return {
            'seg_loss': total_seg_loss,
            'seg_drivable_total': drivable_losses['total'],
            'seg_drivable_ce': drivable_losses['ce'],
            'seg_drivable_dice': drivable_losses['dice'],
            'seg_lane_total': lane_losses['total'],
            'seg_lane_ce': lane_losses['ce'],
            'seg_lane_dice': lane_losses['dice'],
        }
