"""
时序融合模块
============
利用历史帧 BEV 特征增强当前帧感知能力。
- 自注意力时序融合
- 支持多帧历史信息聚合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class TemporalSelfAttention(nn.Module):
    """
    时序自注意力 —— 当前帧 BEV 查询关注历史帧 BEV 特征
    """

    def __init__(self, embed_dims: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads

        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.out_proj = nn.Linear(embed_dims, embed_dims)

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dims * 4, embed_dims),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        curr_bev: torch.Tensor,
        hist_bev: torch.Tensor,
        hist_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            curr_bev: 当前帧 BEV 特征 [B, N_bev, C]
            hist_bev: 历史帧 BEV 特征 [B, T, N_bev, C]
            hist_mask: 历史帧有效掩码 [B, T]
        Returns:
            融合后的 BEV 特征 [B, N_bev, C]
        """
        B, N_bev, C = curr_bev.shape
        T = hist_bev.shape[1]

        # 当前帧作为 query
        q = self.q_proj(self.norm1(curr_bev)).reshape(B, N_bev, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # [B, num_heads, N_bev, head_dim]

        # 历史帧作为 key/value
        hist_flat = hist_bev.reshape(B, T * N_bev, C)
        k = self.k_proj(self.norm1(hist_flat)).reshape(B, T * N_bev, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)
        v = self.v_proj(self.norm1(hist_flat)).reshape(B, T * N_bev, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3)

        # 缩放点积注意力
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # [B, num_heads, N_bev, head_dim]
        out = out.transpose(1, 2).reshape(B, N_bev, C)
        out = self.out_proj(out)
        out = curr_bev + out  # 残差

        # FFN
        out = out + self.ffn(self.norm2(out))

        return out


class TemporalFusion(nn.Module):
    """
    时序融合模块
    —— 管理历史 BEV 特征队列，进行时序融合
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_history: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_history = num_history
        self.temporal_attn = TemporalSelfAttention(embed_dims, num_heads, dropout)

        # 历史特征队列（训练时使用滑动窗口）
        self.register_buffer('history_queue', None, persistent=False)

    def forward(
        self,
        bev_features: torch.Tensor,
        can_bus: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            bev_features: 当前帧 BEV 特征 [B, N_bev, C]
            can_bus: CAN 总线数据用于时序对齐 [B, T, ...]
        Returns:
            时序增强的 BEV 特征
        """
        B, N_bev, C = bev_features.shape

        # 如果没有历史帧，直接返回
        if self.history_queue is None:
            self._update_queue(bev_features)
            return bev_features

        # 时序融合
        hist_bev = self.history_queue[:self.num_history]  # [T, B, N_bev, C]
        hist_bev = hist_bev.permute(1, 0, 2, 3)  # [B, T, N_bev, C]

        fused = self.temporal_attn(bev_features, hist_bev)

        # 更新队列
        self._update_queue(bev_features)

        return fused

    def _update_queue(self, bev: torch.Tensor):
        """更新历史队列"""
        bev_detached = bev.detach()
        if self.history_queue is None:
            self.history_queue = bev_detached.unsqueeze(0)
        else:
            self.history_queue = torch.cat([bev_detached.unsqueeze(0), self.history_queue], dim=0)
            if self.history_queue.shape[0] > self.num_history:
                self.history_queue = self.history_queue[:self.num_history]

    def reset_queue(self):
        """重置历史队列（用于推理开始或场景切换）"""
        self.history_queue = None
