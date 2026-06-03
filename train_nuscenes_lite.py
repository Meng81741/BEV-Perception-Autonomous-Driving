"""
NuScenes 训练脚本（零依赖版）
=============================
使用零依赖 nuScenes 加载器 + 轻量 BEV 模型完成端到端训练。

用法:
    python train_nuscenes_lite.py --data_root datasat/nuscenes --epochs 5
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.bevformer_config import lightweight_config
from models.bev_perception import BEVPerception
from data.nuscenes_loader import NuScenesLoader, NUSCENES_CLASS_NAMES
from data.transforms import NormalizeImage, generate_heatmap
from losses.segmentation_loss import MultiTaskSegmentationLoss

# AMP 兼容
try:
    from torch.amp import GradScaler, autocast
    _NEW_AMP = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _NEW_AMP = False


def parse_args():
    p = argparse.ArgumentParser(description='NuScenes 零依赖训练')
    p.add_argument('--data_root', type=str, default='datasat/nuscenes')
    p.add_argument('--version', type=str, default='v1.0-mini')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--output_dir', type=str, default='./output_nuscenes_lite')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--num_workers', type=int, default=0)
    return p.parse_args()


def build_heatmap_target(boxes_3d: np.ndarray, labels: np.ndarray,
                         bev_h: int, bev_w: int,
                         bev_range=(-51.2, 51.2, 1.6)) -> np.ndarray:
    """从 3D 框生成 BEV 热力图"""
    num_classes = 10
    x_min, x_max, x_res = bev_range
    y_min, y_max, y_res = bev_range
    heatmap = np.zeros((num_classes, bev_h, bev_w), dtype=np.float32)

    for box, cls_id in zip(boxes_3d, labels):
        cx, cy = box[0], box[1]
        h_idx = int((cx - x_min) / x_res)
        w_idx = int((cy - y_min) / y_res)
        if 0 <= h_idx < bev_h and 0 <= w_idx < bev_w:
            radius = 1
            for dh in range(-radius, radius + 1):
                for dw in range(-radius, radius + 1):
                    hh, ww = h_idx + dh, w_idx + dw
                    if 0 <= hh < bev_h and 0 <= ww < bev_w:
                        val = np.exp(-(dh**2 + dw**2) / 2.0)
                        heatmap[int(cls_id), hh, ww] = max(
                            heatmap[int(cls_id), hh, ww], val
                        )
    return heatmap


def train():
    args = parse_args()
    
    # 轻量配置
    cfg = lightweight_config
    bev_h, bev_w = cfg.bev_grid.bev_h, cfg.bev_grid.bev_w
    print(f"BEV grid: {bev_h}×{bev_w}  Image: {cfg.image_size}")
    print(f"Backbone: {cfg.backbone.backbone_type}  Embed: {cfg.bev_encoder.embed_dims}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 数据 ----
    print("\nLoading nuScenes...")
    loader = NuScenesLoader(args.data_root, args.version, image_size=cfg.image_size)
    normalize = NormalizeImage()

    # ---- 模型 ----
    print("Building model...")
    model = BEVPerception(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,}")

    # ---- 损失 & 优化器 ----
    seg_loss_fn = MultiTaskSegmentationLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda', enabled=True) if _NEW_AMP else GradScaler(enabled=True)

    # ---- 训练 ----
    print(f"\n{'='*50}\nTraining {args.epochs} epochs\n{'='*50}")

    for epoch in range(args.epochs):
        model.train()
        model.reset_temporal()
        epoch_loss = 0.0
        total_det = 0.0
        total_seg = 0.0

        # 随机打乱样本索引
        indices = np.random.permutation(len(loader))
        pbar = tqdm(indices[:len(indices)], desc=f'Epoch {epoch+1}/{args.epochs}')

        for idx in pbar:
            sample = loader[idx]

            # ---- 预处理图像 ----
            imgs = sample['images'].astype(np.float32) / 255.0
            imgs = (imgs - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            imgs = torch.from_numpy(imgs).float().permute(0, 3, 1, 2)  # [6, 3, H, W]
            imgs = imgs.unsqueeze(0).to(device)  # [1, 6, 3, H, W]

            # ---- 目标 ----
            boxes = sample['boxes_3d']
            labels = sample['labels']

            # BEV 热力图 (检测目标)
            heatmap_gt = build_heatmap_target(boxes, labels, bev_h, bev_w)
            heatmap_gt = torch.from_numpy(heatmap_gt).float().unsqueeze(0).to(device)

            # 分割目标 (nuScenes 没有可行驶区域/车道线标注，用零占位)
            drivable_gt = torch.zeros(1, bev_h, bev_w, dtype=torch.long, device=device)
            lane_gt = torch.zeros(1, bev_h, bev_w, dtype=torch.long, device=device)

            # ---- 前向 ----
            amp_kw = {'device_type': 'cuda', 'enabled': True} if _NEW_AMP else {'enabled': True}
            with autocast(**amp_kw):
                outputs = model(imgs)

                # 检测损失 (简化: 只用 FocalLoss on heatmap)
                pred_hm = torch.sigmoid(outputs['detections']['heatmap'])
                # Focal Loss
                alpha, beta = 2.0, 4.0
                pos = (heatmap_gt == 1).float()
                neg = (heatmap_gt < 1).float()
                det_loss = -(pos * torch.log(pred_hm + 1e-6) * (1 - pred_hm) ** alpha
                           + neg * torch.log(1 - pred_hm + 1e-6) * pred_hm ** alpha
                           * (1 - heatmap_gt) ** beta).mean()

                # 分割损失
                seg_losses = seg_loss_fn(
                    outputs['segmentation'],
                    {'drivable_mask': drivable_gt, 'lane_mask': lane_gt},
                )
                seg_loss = seg_losses['seg_loss']

                total_loss = det_loss + seg_loss

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()
            total_det += det_loss.item()
            total_seg += seg_loss.item()

            pbar.set_postfix({
                'loss': f'{total_loss.item():.3f}',
                'det': f'{det_loss.item():.3f}',
                'seg': f'{seg_loss.item():.3f}',
            })

        scheduler.step()
        n = len(indices)
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {epoch_loss/n:.4f} "
              f"(det: {total_det/n:.4f}, seg: {total_seg/n:.4f})")

        # 保存
        if (epoch + 1) % 5 == 0:
            ckpt = os.path.join(args.output_dir, f'epoch_{epoch+1}.pth')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, ckpt)
            print(f"  Saved: {ckpt}")

    # 最终保存
    final = os.path.join(args.output_dir, 'final.pth')
    torch.save({'model_state_dict': model.state_dict(), 'config': cfg}, final)
    print(f"\nDone! Model: {final}")


if __name__ == '__main__':
    train()
