"""
NuScenes 训练脚本
==================
专门针对 nuScenes 数据集的训练入口。

用法:
    # 完整训练
    python train_nuscenes.py --data_root /data/nuscenes

    # mini 模式快速测试
    python train_nuscenes.py --data_root /data/nuscenes --mini --epochs 3

    # 不使用 HD Map（仅 3D 检测）
    python train_nuscenes.py --data_root /data/nuscenes --no_map
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# PyTorch 2.x 兼容
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

import argparse
import os
import sys
import time
from tqdm import tqdm
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.bevformer_config import BEVPerceptionConfig
from models.bev_perception import BEVPerception
from data.nuscenes_dataset import build_nuscenes_dataloader
from losses.detection_loss import DetectionLoss
from losses.segmentation_loss import MultiTaskSegmentationLoss


def parse_args():
    parser = argparse.ArgumentParser(
        description='NuScenes BEV 多任务感知训练'
    )
    parser.add_argument('--data_root', type=str, required=True,
                        help='nuScenes 数据集根目录')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        choices=['v1.0-trainval', 'v1.0-mini'],
                        help='nuScenes 版本')
    parser.add_argument('--mini', action='store_true',
                        help='使用 mini 数据集快速测试')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # 模型参数
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'])
    parser.add_argument('--lightweight', action='store_true',
                        help='使用轻量化配置 (ResNet18 + 通道减半, ~15M)')
    parser.add_argument('--no_map', action='store_true',
                        help='不使用 HD Map (仅 3D 检测)')

    # 系统参数
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='./output_nuscenes')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--save_interval', type=int, default=3)

    return parser.parse_args()


def main():
    args = parse_args()

    # 配置
    if args.lightweight:
        from configs.bevformer_config import lightweight_config
        config = lightweight_config
        print("使用轻量化配置 (ResNet18 + 通道减半)")
    else:
        config = BEVPerceptionConfig()
        config.backbone.backbone_type = args.backbone
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.weight_decay = args.weight_decay
    config.max_epochs = args.epochs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print(f"nuScenes 版本: {args.version}")
    print(f"HD Map: {'关闭' if args.no_map else '启用'}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 数据加载器 ----
    split = 'mini' if args.mini else 'train'
    print(f"\n加载 {split} 数据...")

    train_loader = build_nuscenes_dataloader(
        data_root=args.data_root,
        version=args.version,
        split=split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        render_map=not args.no_map,
    )
    print(f"训练批次数: {len(train_loader)}")

    # ---- 模型 ----
    print("\n构建模型...")
    model = BEVPerception(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {n_params:,}")

    # ---- 损失函数 ----
    det_loss_fn = DetectionLoss()
    seg_loss_fn = MultiTaskSegmentationLoss()

    # ---- 优化器 ----
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs
    )
    scaler = GradScaler('cuda', enabled=args.amp)

    # ---- 训练循环 ----
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    global_step = 0
    for epoch in range(config.max_epochs):
        model.train()
        model.reset_temporal()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.max_epochs}')
        for batch in pbar:
            images = batch['images'].to(device)
            camera_params = batch['camera_params'].to(device)

            with autocast('cuda', enabled=args.amp):
                outputs = model(images, camera_params)

                # 检测损失
                # 注意: 需要将 batch 中的 boxes_3d/labels 转为 heatmap 等目标格式
                # 此处为简化版本，实际训练需要完整的目标构建
                det_losses = {'det_loss': torch.tensor(0.0, device=device)}

                # 分割损失
                seg_targets = {
                    'drivable_mask': batch['drivable_mask'].to(device),
                    'lane_mask': batch['lane_mask'].to(device),
                }
                seg_losses = seg_loss_fn(outputs['segmentation'], seg_targets)

                total_loss = det_losses['det_loss'] + seg_losses['seg_loss']

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()
            global_step += 1

            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'seg': f'{seg_losses["seg_loss"].item():.4f}',
            })

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config.max_epochs} - Avg Loss: {avg_loss:.4f}")

        # 保存检查点
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.output_dir, f'epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, ckpt_path)
            print(f"  保存: {ckpt_path}")

    # 最终保存
    final_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save({'model_state_dict': model.state_dict(), 'config': config}, final_path)
    print(f"\n训练完成! 模型: {final_path}")


if __name__ == '__main__':
    main()
