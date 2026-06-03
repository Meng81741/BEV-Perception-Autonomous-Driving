"""
可视化工具
==========
BEV 感知结果的可视化，包括：
- BEV 空间中的 3D 检测框
- 可行驶区域热力图
- 车道线分割图
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from typing import Dict, List, Optional, Tuple
import cv2


# 类别颜色 (NuScenes 风格)
CLASS_COLORS = {
    'car':                    (0, 0, 255),      # 红色
    'truck':                  (255, 0, 0),      # 蓝色
    'bus':                    (255, 255, 0),    # 青色
    'trailer':                (0, 255, 255),    # 黄色
    'construction_vehicle':   (255, 0, 255),    # 紫色
    'pedestrian':             (0, 255, 0),      # 绿色
    'motorcycle':             (128, 128, 0),    # 橄榄绿
    'bicycle':                (0, 128, 128),    # 深青
    'traffic_cone':           (128, 0, 128),    # 紫色
    'barrier':                (128, 128, 128),  # 灰色
}

CLASS_NAMES = list(CLASS_COLORS.keys())


def draw_boxes_bev(
    ax: plt.Axes,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
    bev_range: Tuple[float, float] = (-51.2, 51.2),
    linewidth: float = 1.5,
    show_velocity: bool = False,
):
    """
    在 BEV 视图上绘制 3D 检测框

    Args:
        ax: matplotlib axes
        boxes: [N, 9] (cx, cy, cz, l, w, h, yaw, vx, vy)
        labels: [N] 类别索引
        scores: [N] 置信度（可选）
        bev_range: (x_min, x_max)
    """
    for i in range(boxes.shape[0]):
        cx, cy = boxes[i, 0].item(), boxes[i, 1].item()
        l, w = boxes[i, 3].item(), boxes[i, 4].item()
        yaw = boxes[i, 6].item()

        cls_id = int(labels[i].item()) if labels[i].numel() == 1 else int(labels[i])
        color = np.array(list(CLASS_COLORS.values())[cls_id % len(CLASS_COLORS)]) / 255.0

        # 绘制旋转矩形
        # 框的 4 个角点（以中心为原点，未旋转）
        corners = np.array([
            [-l/2, -w/2],
            [ l/2, -w/2],
            [ l/2,  w/2],
            [-l/2,  w/2],
        ])

        # 旋转矩阵
        cos, sin = np.cos(yaw), np.sin(yaw)
        rot = np.array([[cos, -sin], [sin, cos]])
        corners = corners @ rot.T
        corners[:, 0] += cx
        corners[:, 1] += cy

        # 绘制多边形
        polygon = patches.Polygon(
            corners, closed=True, fill=False,
            edgecolor=color, linewidth=linewidth,
            alpha=0.9,
        )
        ax.add_patch(polygon)

        # 绘制方向指示线
        front = np.array([l/2, 0]) @ rot.T + np.array([cx, cy])
        ax.arrow(cx, cy, front[0]-cx, front[1]-cy,
                head_width=0.3, head_length=0.5,
                fc=color, ec=color, alpha=0.7)

        # 标注
        if scores is not None:
            text = f"{CLASS_NAMES[cls_id]}: {scores[i].item():.2f}"
        else:
            text = CLASS_NAMES[cls_id]
        ax.text(cx, cy - w/2 - 0.5, text, fontsize=6, color=color,
                ha='center', va='bottom', alpha=0.9)

        # 速度箭头
        if show_velocity:
            vx, vy = boxes[i, 7].item(), boxes[i, 8].item()
            ax.arrow(cx, cy, vx, vy,
                    head_width=0.2, head_length=0.3,
                    fc='orange', ec='orange', alpha=0.6)


def visualize_bev(
    detections: List[Dict],
    drivable_map: Optional[torch.Tensor] = None,
    lane_map: Optional[torch.Tensor] = None,
    bev_range: Tuple[float, float] = (-51.2, 51.2),
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
):
    """
    可视化 BEV 感知结果

    Args:
        detections: 检测结果列表
        drivable_map: [H, W] 可行驶区域图
        lane_map: [H, W] 车道线图
        bev_range: BEV 空间范围
        save_path: 保存路径
        figsize: 图像尺寸
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # 背景
    ax.set_xlim(bev_range[1], bev_range[0])  # 交换使车辆朝上
    ax.set_ylim(bev_range[0], bev_range[1])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m)')
    ax.set_title('BEV Perception Results')

    # 绘制可行驶区域
    if drivable_map is not None:
        if isinstance(drivable_map, torch.Tensor):
            drivable_map = drivable_map.cpu().numpy()
        H, W = drivable_map.shape
        x_min, x_max = bev_range
        y_min, y_max = bev_range

        extent = [y_min, y_max, x_min, x_max]
        ax.imshow(drivable_map, extent=extent, origin='lower',
                  cmap='Greens', alpha=0.3, interpolation='bilinear')

    # 绘制车道线
    if lane_map is not None:
        if isinstance(lane_map, torch.Tensor):
            lane_map = lane_map.cpu().numpy()
        H, W = lane_map.shape
        extent = [bev_range[1], bev_range[0], bev_range[0], bev_range[1]]
        ax.imshow(lane_map, extent=extent, origin='lower',
                  cmap='Blues', alpha=0.3, interpolation='bilinear')

    # 绘制检测框
    for dets in detections:
        if dets['boxes_3d'].numel() > 0:
            draw_boxes_bev(
                ax, dets['boxes_3d'], dets['labels'],
                scores=dets.get('scores'), bev_range=bev_range
            )

    # 绘制自车
    ego_car = patches.Rectangle(
        (-1.0, -2.0), 2.0, 4.0,
        linewidth=2, edgecolor='black', facecolor='white', alpha=0.9
    )
    ax.add_patch(ego_car)
    ax.text(0, 0, 'EGO', ha='center', va='center', fontsize=10, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_multiview(
    images: torch.Tensor,
    detections: Optional[List[Dict]] = None,
    save_path: Optional[str] = None,
):
    """
    可视化多视图相机图像

    Args:
        images: [N_cam, 3, H, W] 图像张量
        detections: 检测结果
        save_path: 保存路径
    """
    N_cam = images.shape[0]
    cols = min(3, N_cam)
    rows = (N_cam + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))

    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    cam_names = ['Front', 'Front Left', 'Front Right',
                 'Rear', 'Rear Left', 'Rear Right']

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(N_cam):
        ax = axes[i]
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        # 反归一化
        img = img * std + mean
        img = np.clip(img, 0, 1)

        ax.imshow(img)
        ax.set_title(cam_names[i] if i < len(cam_names) else f'Cam {i}')
        ax.axis('off')

    # 隐藏多余的子图
    for i in range(N_cam, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_drivable_colormap() -> ListedColormap:
    """创建可行驶区域的颜色映射"""
    colors = ['black', 'green']
    return ListedColormap(colors)


def create_lane_colormap() -> ListedColormap:
    """创建车道线的颜色映射"""
    colors = ['black', 'blue']
    return ListedColormap(colors)
