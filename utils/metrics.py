"""
评估指标
========
BEV 多任务感知的评估指标：
- 3D 检测: mAP, NDS (NuScenes Detection Score)
- 分割: mIoU (可行驶区域 / 车道线)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# ============================================================================
# 3D 检测指标
# ============================================================================

def compute_iou_3d(
    box_a: torch.Tensor,
    box_b: torch.Tensor,
) -> torch.Tensor:
    """
    计算两个 3D 边界框的 BEV IoU

    Args:
        box_a: [N, 7] (cx, cy, cz, l, w, h, yaw)
        box_b: [M, 7]
    Returns:
        iou: [N, M]
    """
    # 简化实现：仅计算 BEV (x-y) IoU

    def _corners(box: torch.Tensor) -> torch.Tensor:
        """将 3D 框转为 4 个角点 (BEV)"""
        cx, cy = box[..., 0], box[..., 1]
        l, w = box[..., 3], box[..., 4]
        yaw = box[..., 6]

        # 局部角点
        corners_local = torch.tensor([
            [ 0.5,  0.5],
            [-0.5,  0.5],
            [-0.5, -0.5],
            [ 0.5, -0.5],
        ], device=box.device, dtype=box.dtype)

        corners_local = corners_local * torch.stack([l, w], dim=-1).unsqueeze(-2)

        # 旋转
        cos, sin = torch.cos(yaw), torch.sin(yaw)
        rot = torch.stack([
            torch.stack([cos, -sin], dim=-1),
            torch.stack([sin,  cos], dim=-1),
        ], dim=-2)  # [..., 2, 2]

        corners_rot = torch.matmul(corners_local, rot.transpose(-2, -1))  # [..., 4, 2]

        # 平移
        corners = corners_rot + torch.stack([cx, cy], dim=-1).unsqueeze(-2)

        return corners

    if box_a.dim() == 1:
        box_a = box_a.unsqueeze(0)
    if box_b.dim() == 1:
        box_b = box_b.unsqueeze(0)

    corners_a = _corners(box_a)  # [N, 4, 2]
    corners_b = _corners(box_b)  # [M, 4, 2]

    N, M = corners_a.shape[0], corners_b.shape[0]

    # 计算每对框的 IoU
    iou = torch.zeros(N, M, device=box_a.device)

    for i in range(N):
        for j in range(M):
            ca = corners_a[i]  # [4, 2]
            cb = corners_b[j]  # [4, 2]

            # 使用 shapely 风格的多边形交集面积计算
            # 简化：使用轴对齐近似
            min_a = ca.min(dim=0)[0]
            max_a = ca.max(dim=0)[0]
            min_b = cb.min(dim=0)[0]
            max_b = cb.max(dim=0)[0]

            area_a = (max_a[0] - min_a[0]) * (max_a[1] - min_a[1])
            area_b = (max_b[0] - min_b[0]) * (max_b[1] - min_b[1])

            inter_min = torch.max(min_a, min_b)
            inter_max = torch.min(max_a, max_b)
            inter_area = torch.clamp(inter_max[0] - inter_min[0], min=0) * \
                         torch.clamp(inter_max[1] - inter_min[1], min=0)

            union = area_a + area_b - inter_area
            iou[i, j] = inter_area / (union + 1e-6)

    return iou


def compute_ap(
    recalls: np.ndarray,
    precisions: np.ndarray,
) -> float:
    """
    计算 Average Precision (AP)
    —— 使用 101-point 插值法
    """
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 101.0
    return ap


def evaluate_detection(
    predictions: List[Dict],
    ground_truths: List[Dict],
    num_classes: int = 10,
    iou_thresholds: List[float] = None,
) -> Dict:
    """
    评估 3D 检测性能

    Args:
        predictions: 预测结果列表
        ground_truths: 真值列表
        num_classes: 类别数
        iou_thresholds: IoU 阈值列表
    Returns:
        mAP, per_class_AP, etc.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5]

    results = {}
    all_aps = []

    for iou_thr in iou_thresholds:
        class_aps = []

        for cls_id in range(num_classes):
            # 收集该类别的所有预测和真值
            all_scores = []
            all_matched = []
            num_gt = 0

            for preds, gts in zip(predictions, ground_truths):
                # 过滤当前类别
                cls_mask_pred = preds['labels'] == cls_id
                cls_mask_gt = gts['labels'] == cls_id

                pred_boxes = preds['boxes_3d'][cls_mask_pred]
                pred_scores = preds['scores'][cls_mask_pred]
                gt_boxes = gts['boxes_3d'][cls_mask_gt]

                num_gt += gt_boxes.shape[0]

                if pred_boxes.shape[0] == 0:
                    continue

                # 按置信度排序
                sorted_idx = torch.argsort(pred_scores, descending=True)
                pred_boxes = pred_boxes[sorted_idx]
                pred_scores = pred_scores[sorted_idx]

                if gt_boxes.shape[0] > 0:
                    iou = compute_iou_3d(pred_boxes, gt_boxes)

                    matched_gt = set()
                    for p_idx in range(pred_boxes.shape[0]):
                        best_iou, best_gt = iou[p_idx].max(dim=0)
                        if best_iou >= iou_thr and best_gt.item() not in matched_gt:
                            all_matched.append(True)
                            matched_gt.add(best_gt.item())
                        else:
                            all_matched.append(False)
                        all_scores.append(pred_scores[p_idx].item())
                else:
                    for s in pred_scores:
                        all_scores.append(s.item())
                        all_matched.append(False)

            if len(all_scores) == 0:
                class_aps.append(0.0)
                continue

            # 按分数排序
            sorted_idx = np.argsort(all_scores)[::-1]
            all_matched = np.array(all_matched)[sorted_idx]

            tp = np.cumsum(all_matched)
            fp = np.cumsum(~all_matched)

            recalls = tp / max(num_gt, 1)
            precisions = tp / np.maximum(tp + fp, 1)

            ap = compute_ap(recalls, precisions)
            class_aps.append(ap)

        mAP = np.mean(class_aps)
        all_aps.append(mAP)
        results[f'mAP@{iou_thr}'] = mAP

    results['mAP'] = np.mean(all_aps)
    results['class_AP'] = class_aps

    return results


# ============================================================================
# 分割指标
# ============================================================================

class SegmentationMetrics:
    """
    分割评估指标计算器
    —— mIoU, Pixel Accuracy, Dice coefficient
    """

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Args:
            pred: [B, H, W] 或 [B, C, H, W] 预测
            target: [B, H, W] 目标标签
        """
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()

        # 过滤忽略索引
        mask = target != self.ignore_index
        pred = pred[mask]
        target = target[mask]

        for p, t in zip(pred, target):
            self.confusion_matrix[t, p] += 1

    def compute(self) -> Dict[str, float]:
        """计算各项指标"""
        cm = self.confusion_matrix

        # IoU per class
        intersection = np.diag(cm)
        union = cm.sum(axis=0) + cm.sum(axis=1) - intersection

        iou_per_class = intersection / (union + 1e-6)
        miou = np.mean(iou_per_class)

        # Pixel Accuracy
        pixel_acc = intersection.sum() / (cm.sum() + 1e-6)

        # Dice per class
        dice_per_class = 2 * intersection / (cm.sum(axis=0) + cm.sum(axis=1) + 1e-6)

        return {
            'mIoU': float(miou),
            'pixel_accuracy': float(pixel_acc),
            'iou_per_class': iou_per_class.tolist(),
            'dice_per_class': dice_per_class.tolist(),
        }


def evaluate_segmentation(
    predictions: List[Dict],
    ground_truths: List[Dict],
    num_drivable_classes: int = 2,
    num_lane_classes: int = 2,
) -> Dict:
    """
    评估分割性能

    Returns:
        drivable_mIoU, lane_mIoU, etc.
    """
    drivable_metrics = SegmentationMetrics(num_drivable_classes)
    lane_metrics = SegmentationMetrics(num_lane_classes)

    for preds, gts in zip(predictions, ground_truths):
        drivable_metrics.update(preds['drivable'], gts['drivable_mask'])
        lane_metrics.update(preds['lane'], gts['lane_mask'])

    drivable_results = drivable_metrics.compute()
    lane_results = lane_metrics.compute()

    return {
        'drivable_mIoU': drivable_results['mIoU'],
        'drivable_pixel_acc': drivable_results['pixel_accuracy'],
        'lane_mIoU': lane_results['mIoU'],
        'lane_pixel_acc': lane_results['pixel_accuracy'],
    }
