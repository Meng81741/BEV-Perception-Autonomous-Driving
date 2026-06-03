"""
训练脚本
========
BEV 多任务自动驾驶感知模型训练入口。

用法:
    python train.py --config configs/bevformer_config.py --data_root /path/to/data
    python train.py --data_root /path/to/nuscenes --use_nuscenes --batch_size 2 --epochs 24
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# PyTorch 2.x 兼容: 新版 torch.amp, 旧版 torch.cuda.amp
try:
    from torch.amp import GradScaler, autocast
    _NEW_AMP_API = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _NEW_AMP_API = False

import argparse
import os
import sys
import time
from tqdm import tqdm
from typing import Dict, Optional

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.bevformer_config import BEVPerceptionConfig, default_config
from models.bev_perception import BEVPerception, build_bev_perception
from data.dataset import build_dataloader
from losses.detection_loss import DetectionLoss
from losses.segmentation_loss import MultiTaskSegmentationLoss
from utils.metrics import evaluate_detection, evaluate_segmentation, SegmentationMetrics
from utils.visualization import visualize_bev, visualize_multiview


def parse_args():
    parser = argparse.ArgumentParser(description='BEV 多任务感知模型训练')

    # 数据参数
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据集根目录')
    parser.add_argument('--use_nuscenes', action='store_true',
                        help='使用 NuScenes 数据集')
    parser.add_argument('--ann_file', type=str, default=None,
                        help='自定义标注文件路径')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=24,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                        help='预热轮数')

    # 模型参数
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'],
                        help='骨干网络类型')
    parser.add_argument('--lightweight', action='store_true',
                        help='使用轻量化配置 (ResNet18 + 通道减半, ~15M 参数)')
    parser.add_argument('--no_temporal', action='store_true',
                        help='禁用时序融合')

    # 系统参数
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='使用混合精度训练')
    parser.add_argument('--val_interval', type=int, default=2,
                        help='验证间隔（轮）')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='保存检查点间隔（轮）')
    parser.add_argument('--log_interval', type=int, default=20,
                        help='日志打印间隔（批次）')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式（少量数据快速测试）')

    return parser.parse_args()


def build_config(args) -> BEVPerceptionConfig:
    """根据命令行参数构建配置"""
    if args.lightweight:
        from configs.bevformer_config import lightweight_config
        config = lightweight_config
        print("使用轻量化配置 (ResNet18 + 通道减半)")
        return config

    config = BEVPerceptionConfig()

    # 更新骨干配置
    config.backbone.backbone_type = args.backbone

    # 更新训练参数
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.weight_decay = args.weight_decay
    config.max_epochs = args.epochs
    config.warmup_epochs = args.warmup_epochs

    # 更新时序配置
    if args.no_temporal:
        config.temporal.use_temporal = False

    return config


class Trainer:
    """BEV 多任务感知训练器"""

    def __init__(self, config: BEVPerceptionConfig, args):
        self.config = config
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))

        # 构建模型
        self.model = BEVPerception(config).to(self.device)
        print(f"模型参数量: {self._count_parameters():,}")

        # 构建数据加载器
        self._build_dataloaders()

        # 构建损失函数
        self.det_loss_fn = DetectionLoss()
        self.seg_loss_fn = MultiTaskSegmentationLoss()

        # 优化器和调度器
        self._build_optimizer()

        # 混合精度
        # 混合精度
        if _NEW_AMP_API:
            self.scaler = GradScaler('cuda', enabled=args.amp)
        else:
            self.scaler = GradScaler(enabled=args.amp)

        # 恢复训练
        self.start_epoch = 0
        self.global_step = 0
        if args.resume:
            self._resume(args.resume)

    def _count_parameters(self) -> int:
        """统计可训练参数"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _build_dataloaders(self):
        """构建训练和验证数据加载器"""
        common_kwargs = dict(
            data_root=self.args.data_root,
            image_size=self.config.image_size,
            bev_h=self.config.bev_grid.bev_h,
            bev_w=self.config.bev_grid.bev_w,
            num_cams=self.config.bev_encoder.num_cams,
            use_nuscenes=self.args.use_nuscenes,
            ann_file=self.args.ann_file,
        )

        self.train_loader = build_dataloader(
            split='train', batch_size=self.config.batch_size,
            num_workers=self.args.num_workers, shuffle=True, **common_kwargs,
        )

        self.val_loader = build_dataloader(
            split='val', batch_size=1,
            num_workers=self.args.num_workers, shuffle=False, **common_kwargs,
        )

    def _build_optimizer(self):
        """构建优化器和学习率调度器"""
        # 分组权重衰减
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bn' in name or 'bias' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]

        self.optimizer = optim.AdamW(param_groups, lr=self.config.learning_rate)

        # 余弦退火调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_epochs - self.config.warmup_epochs,
            eta_min=self.config.learning_rate * 0.01,
        )

        # 线性预热
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config.warmup_epochs * len(self.train_loader),
        )

    def _resume(self, checkpoint_path: str):
        """从检查点恢复"""
        print(f"从 {checkpoint_path} 恢复训练")
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.start_epoch = ckpt['epoch'] + 1
        self.global_step = ckpt['global_step']
        print(f"恢复到 Epoch {self.start_epoch}, Step {self.global_step}")

    def _prepare_targets(self, batch: Dict, detections: Dict = None) -> Dict:
        """准备训练目标。若无检测标注，用零张量替代（仅计算分割损失）。"""
        targets = {
            'drivable_mask': batch['drivable_mask'].to(self.device),
            'lane_mask': batch['lane_mask'].to(self.device),
        }

        # 检测目标 — 如果数据集中不存在则生成占位零张量
        det_missing = batch.get('heatmap') is None and detections is not None
        targets['det_missing'] = det_missing

        if det_missing and detections is not None:
            hm = detections['heatmap']  # [B, n_cls, H, W]
            B, _, H, W = hm.shape
            dev = hm.device
            targets['heatmap'] = torch.zeros_like(hm)
            targets['size'] = torch.zeros(B, 3, H, W, device=dev)
            targets['offset'] = torch.zeros(B, 2, H, W, device=dev)
            targets['rotation'] = torch.zeros(B, 2, H, W, device=dev)
            targets['velocity'] = torch.zeros(B, 2, H, W, device=dev)
            targets['z'] = torch.zeros(B, 1, H, W, device=dev)
            targets['reg_mask'] = torch.zeros(B, 1, H, W, device=dev, dtype=torch.bool)
        else:
            targets['heatmap'] = batch.get('heatmap')
            targets['size'] = batch.get('size_target')
            targets['offset'] = batch.get('offset_target')
            targets['rotation'] = batch.get('rotation_target')
            targets['velocity'] = batch.get('velocity_target')
            targets['z'] = batch.get('z_target')
            targets['reg_mask'] = batch.get('reg_mask')

        return targets

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        self.model.reset_temporal()

        epoch_losses = {}
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.max_epochs}')
        for batch_idx, batch in enumerate(pbar):
            # 数据移动到设备
            images = batch['images'].to(self.device)

            # 前向传播
            amp_kwargs = {'device_type': 'cuda', 'enabled': self.args.amp} if _NEW_AMP_API else {'enabled': self.args.amp}
            with autocast(**amp_kwargs):
                outputs = self.model(images)
                targets = self._prepare_targets(batch, outputs['detections'])
                det_losses = self.det_loss_fn(outputs['detections'], targets)
                seg_losses = self.seg_loss_fn(outputs['segmentation'], targets)

                total_loss = (
                    self.config.loss_det_weight * det_losses['det_loss'] +
                    self.config.loss_seg_drivable_weight * seg_losses['seg_drivable_total'] +
                    self.config.loss_seg_lane_weight * seg_losses['seg_lane_total']
                )

            # 反向传播
            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()

            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 预热阶段更新学习率
            if epoch < self.config.warmup_epochs:
                self.warmup_scheduler.step()

            # 记录损失
            losses = {
                'total': total_loss.item(),
                **{k: v.item() for k, v in det_losses.items()},
                **{k: v.item() for k, v in seg_losses.items()},
            }
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v

            # TensorBoard
            self.global_step += 1
            if batch_idx % self.args.log_interval == 0:
                self.writer.add_scalar('train/total_loss', losses['total'], self.global_step)
                self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.global_step)

            pbar.set_postfix({'loss': f"{losses['total']:.4f}"})

        # 平均损失
        for k in epoch_losses:
            epoch_losses[k] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        self.model.reset_temporal()

        all_det_preds = []
        all_det_gts = []
        drivable_metrics = SegmentationMetrics(
            self.config.segmentation_head.num_drivable_classes
        )
        lane_metrics = SegmentationMetrics(
            self.config.segmentation_head.num_lane_classes
        )

        pbar = tqdm(self.val_loader, desc='Validation')
        for batch in pbar:
            images = batch['images'].to(self.device)

            outputs = self.model(images)

            # 收集检测结果
            all_det_preds.append({
                'boxes_3d': outputs['detections']['...'],  # 需要解码
                'scores': None,   # 需要从 heatmap 获取
                'labels': None,
            })

            # 更新分割指标
            drivable_pred = outputs['segmentation']['drivable'].argmax(dim=1)
            lane_pred = outputs['segmentation']['lane'].argmax(dim=1)

            drivable_metrics.update(drivable_pred, batch['drivable_mask'].to(self.device))
            lane_metrics.update(lane_pred, batch['lane_mask'].to(self.device))

        # 计算指标
        drivable_results = drivable_metrics.compute()
        lane_results = lane_metrics.compute()

        return {
            'drivable_mIoU': drivable_results['mIoU'],
            'lane_mIoU': lane_results['mIoU'],
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        ckpt = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
        }

        filename = f'checkpoint_epoch_{epoch+1}.pth'
        path = os.path.join(self.args.output_dir, filename)
        torch.save(ckpt, path)
        print(f"保存检查点: {path}")

        if is_best:
            best_path = os.path.join(self.args.output_dir, 'best_model.pth')
            torch.save(ckpt, best_path)
            print(f"保存最佳模型: {best_path}")

    def train(self):
        """完整训练流程"""
        print("=" * 60)
        print("BEV 多任务感知模型训练")
        print("=" * 60)

        best_val_miou = 0.0

        for epoch in range(self.start_epoch, self.config.max_epochs):
            # 训练
            train_losses = self.train_epoch(epoch)

            # 调整学习率
            if epoch >= self.config.warmup_epochs:
                self.scheduler.step()

            # 打印
            print(f"\nEpoch {epoch+1}/{self.config.max_epochs} - "
                  f"Train Loss: {train_losses['total']:.4f}")

            # 验证
            if (epoch + 1) % self.args.val_interval == 0:
                val_metrics = self.validate()
                print(f"Validation - Drivable mIoU: {val_metrics['drivable_mIoU']:.4f}, "
                      f"Lane mIoU: {val_metrics['lane_mIoU']:.4f}")

                for k, v in val_metrics.items():
                    self.writer.add_scalar(f'val/{k}', v, epoch)

                # 最佳模型判断
                current_miou = (val_metrics['drivable_mIoU'] + val_metrics['lane_mIoU']) / 2

                is_best = current_miou > best_val_miou
                if is_best:
                    best_val_miou = current_miou

                self.save_checkpoint(epoch, is_best=is_best)
            elif (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint(epoch)

        # 训练完成
        final_path = os.path.join(self.args.output_dir, 'final_model.pth')
        torch.save({'model_state_dict': self.model.state_dict(), 'config': self.config}, final_path)
        print(f"\n训练完成! 最终模型保存至: {final_path}")
        self.writer.close()


def main():
    args = parse_args()
    config = build_config(args)

    trainer = Trainer(config, args)
    trainer.train()


if __name__ == '__main__':
    main()
