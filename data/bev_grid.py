"""
BEV 网格工具
============
用于构建 BEV 空间的坐标变换、投影等功能。
"""

import torch
import numpy as np
from typing import Tuple, Optional


class BEVGrid:
    """
    BEV 网格管理器 —— 处理 BEV 空间与世界坐标之间的变换
    """

    def __init__(
        self,
        x_bound: Tuple[float, float, float] = (-51.2, 51.2, 0.8),
        y_bound: Tuple[float, float, float] = (-51.2, 51.2, 0.8),
    ):
        self.x_min, self.x_max, self.x_res = x_bound
        self.y_min, self.y_max, self.y_res = y_bound

        self.H = int((self.x_max - self.x_min) / self.x_res)
        self.W = int((self.y_max - self.y_min) / self.y_res)

    def world_to_bev(self, points: torch.Tensor) -> torch.Tensor:
        """
        世界坐标 → BEV 网格索引

        Args:
            points: [..., 2] (x, y) 世界坐标
        Returns:
            indices: [..., 2] (row, col) BEV 网格索引
        """
        col = (points[..., 1] - self.y_min) / self.y_res
        row = (points[..., 0] - self.x_min) / self.x_res
        return torch.stack([row, col], dim=-1)

    def bev_to_world(self, indices: torch.Tensor) -> torch.Tensor:
        """
        BEV 网格索引 → 世界坐标（网格中心）

        Args:
            indices: [..., 2] (row, col) BEV 网格索引
        Returns:
            points: [..., 2] (x, y) 世界坐标
        """
        x = indices[..., 0] * self.x_res + self.x_min + self.x_res / 2
        y = indices[..., 1] * self.y_res + self.y_min + self.y_res / 2
        return torch.stack([x, y], dim=-1)

    def get_coord_grid(self, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        获取 BEV 坐标网格

        Returns:
            grid: [H, W, 2] — 每个 BEV 格子的世界坐标 (x, y)
        """
        xs = torch.arange(self.H, device=device) * self.x_res + self.x_min + self.x_res / 2
        ys = torch.arange(self.W, device=device) * self.y_res + self.y_min + self.y_res / 2
        grid_y, grid_x = torch.meshgrid(xs, ys, indexing='ij')
        return torch.stack([grid_x, grid_y], dim=-1)


def generate_bev_reference_points(
    bev_h: int,
    bev_w: int,
    z_value: float = 0.0,
    num_points_in_pillar: int = 4,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    生成 BEV 参考点（用于可变形注意力）

    Args:
        bev_h, bev_w: BEV 网格尺寸
        z_value: Z 轴高度
        num_points_in_pillar: 每个柱体的采样点数
    Returns:
        ref_3d: [bev_h * bev_w, num_points_in_pillar, 3]
    """
    xs = torch.linspace(0.5, bev_h - 0.5, bev_h, device=device) / bev_h
    ys = torch.linspace(0.5, bev_w - 0.5, bev_w, device=device) / bev_w

    grid_y, grid_x = torch.meshgrid(xs, ys, indexing='ij')
    ref_2d = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # [N_bev, 2]

    # 沿 Z 轴采样
    zs = torch.linspace(0.5, 1.0, num_points_in_pillar, device=device) - 0.5

    ref_3d = []
    for z in zs:
        ref_3d.append(
            torch.cat([ref_2d, torch.full_like(ref_2d[:, :1], z)], dim=-1)
        )
    ref_3d = torch.stack(ref_3d, dim=1)  # [N_bev, num_points_in_pillar, 3]

    return ref_3d
