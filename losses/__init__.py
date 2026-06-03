"""
损失函数模块导出
"""

from .detection_loss import DetectionLoss, FocalLoss, L1Loss
from .segmentation_loss import SegmentationLoss, MultiTaskSegmentationLoss, DiceLoss

__all__ = [
    'DetectionLoss',
    'FocalLoss',
    'L1Loss',
    'SegmentationLoss',
    'MultiTaskSegmentationLoss',
    'DiceLoss',
]
