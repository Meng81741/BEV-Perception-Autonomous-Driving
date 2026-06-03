"""
推理脚本
========
BEV 多任务感知模型推理与可视化。

用法:
    python inference.py --checkpoint output/best_model.pth --input /path/to/images/
    python inference.py --checkpoint output/best_model.pth --input demo_data/ --visualize
"""

import torch
import numpy as np
import argparse
import os
import sys
import json
import cv2
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.bevformer_config import BEVPerceptionConfig
from models.bev_perception import BEVPerception
from utils.visualization import visualize_bev, visualize_multiview
from utils.metrics import decode_detections


def parse_args():
    parser = argparse.ArgumentParser(description='BEV 多任务感知模型推理')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--input', type=str, required=True,
                        help='输入图像目录或单个样本目录')
    parser.add_argument('--output', type=str, default='./inference_output',
                        help='推理输出目录')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='生成可视化结果')
    parser.add_argument('--score_threshold', type=float, default=0.1,
                        help='检测置信度阈值')
    parser.add_argument('--top_k', type=int, default=100,
                        help='每帧最多保留的检测数')
    parser.add_argument('--device', type=str, default='cuda',
                        help='推理设备')

    return parser.parse_args()


class BEVInference:
    """BEV 感知模型推理器"""

    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"推理设备: {self.device}")

        # 加载检查点
        self._load_model(checkpoint_path)

        # 图像预处理参数
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _load_model(self, checkpoint_path: str):
        """加载模型和配置"""
        print(f"加载模型: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        if 'config' in ckpt:
            config = ckpt['config']
        else:
            config = BEVPerceptionConfig()

        self.config = config
        self.model = BEVPerception(config).to(self.device)

        if 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'])
        else:
            self.model.load_state_dict(ckpt)

        self.model.eval()
        print("模型加载完成")

    def preprocess_images(self, images: List[np.ndarray],
                          target_size: Tuple[int, int] = (900, 1600)) -> torch.Tensor:
        """
        预处理多视图图像

        Args:
            images: 图像列表，每个 [H, W, 3] (RGB)
            target_size: 目标尺寸 (H, W)
        Returns:
            Tensor [1, N_cam, 3, H, W]
        """
        processed = []
        for img in images:
            # 缩放
            if img.shape[:2] != target_size:
                img = cv2.resize(img, (target_size[1], target_size[0]))

            # 归一化
            img = img.astype(np.float32) / 255.0
            img = (img - self.mean) / self.std

            # HWC -> CHW
            img = img.transpose(2, 0, 1)
            processed.append(img)

        # [N_cam, 3, H, W] -> [1, N_cam, 3, H, W]
        tensor = torch.from_numpy(np.stack(processed, axis=0)).float()
        tensor = tensor.unsqueeze(0).to(self.device)

        return tensor

    def load_sample_images(self, input_path: str) -> List[np.ndarray]:
        """
        加载样本图像

        支持的输入格式:
        - 单个目录: 包含 cam_front.jpg, cam_front_left.jpg, ...
        - 图像列表: 逗号分隔的图像路径
        """
        if os.path.isdir(input_path):
            # 搜索相机图像
            cam_patterns = [
                'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
            ]
            images = []
            for pattern in cam_patterns:
                found = False
                for fname in os.listdir(input_path):
                    if pattern in fname.upper() and fname.endswith(('.jpg', '.png', '.jpeg')):
                        img = cv2.imread(os.path.join(input_path, fname))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img)
                        found = True
                        break
                if not found:
                    # 使用占位符（黑色图像）
                    print(f"  警告: 未找到相机 {pattern} 的图像，使用占位符")
                    images.append(np.zeros((900, 1600, 3), dtype=np.uint8))

            return images
        elif os.path.isfile(input_path):
            # 单张图像（复制到所有相机）
            img = cv2.imread(input_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return [img] * 6
        else:
            raise FileNotFoundError(f"输入路径不存在: {input_path}")

    @torch.no_grad()
    def inference(self, input_path: str, score_threshold: float = 0.1,
                  top_k: int = 100) -> Dict:
        """
        执行推理

        Args:
            input_path: 输入路径（目录或图像文件）
            score_threshold: 检测置信度阈值
            top_k: 最多保留的检测数
        Returns:
            推理结果字典
        """
        # 加载图像
        images = self.load_sample_images(input_path)
        print(f"加载 {len(images)} 张图像")

        # 预处理
        input_tensor = self.preprocess_images(images)
        print(f"输入形状: {input_tensor.shape}")

        # 推理
        self.model.reset_temporal()
        outputs = self.model(input_tensor)

        # 后处理
        results = self._postprocess(outputs, score_threshold, top_k)

        return results

    def _postprocess(self, outputs: Dict, score_threshold: float,
                     top_k: int) -> Dict:
        """后处理模型输出"""
        # 检测结果
        detections = outputs['detections']

        # 解码 3D 框
        heatmap = torch.sigmoid(detections['heatmap'])
        B, num_classes, H, W = heatmap.shape

        all_boxes = []
        all_scores = []
        all_labels = []

        for cls_id in range(num_classes):
            hm = heatmap[0, cls_id]
            scores_flat = hm.flatten()
            top_scores, top_indices = torch.topk(
                scores_flat, min(top_k, scores_flat.numel())
            )

            for score, idx in zip(top_scores, top_indices):
                if score < score_threshold:
                    continue

                h_idx = idx // W
                w_idx = idx % W

                # 世界坐标
                cx = (h_idx.float() + detections['offset'][0, 0, h_idx, w_idx]) * 0.8 - 51.2
                cy = (w_idx.float() + detections['offset'][0, 1, h_idx, w_idx]) * 0.8 - 51.2

                size = detections['size'][0, :, h_idx, w_idx]
                l, w, h_box = size[0].item(), size[1].item(), size[2].item()

                rot_sin = detections['rotation'][0, 0, h_idx, w_idx].item()
                rot_cos = detections['rotation'][0, 1, h_idx, w_idx].item()
                yaw = np.arctan2(rot_sin, rot_cos)

                z = detections['z'][0, 0, h_idx, w_idx].item()
                vx = detections['velocity'][0, 0, h_idx, w_idx].item()
                vy = detections['velocity'][0, 1, h_idx, w_idx].item()

                box = [cx.item(), cy.item(), z, l, w, h_box, yaw, vx, vy]
                all_boxes.append(box)
                all_scores.append(score.item())
                all_labels.append(cls_id)

        # 分割结果
        drivable_map = torch.softmax(outputs['segmentation']['drivable'][0], dim=0)
        drivable_map = drivable_map[1].cpu().numpy()  # 取可行驶区域通道

        lane_map = torch.softmax(outputs['segmentation']['lane'][0], dim=0)
        lane_map = lane_map[1].cpu().numpy()

        results = {
            'boxes_3d': np.array(all_boxes) if all_boxes else np.zeros((0, 9)),
            'scores': np.array(all_scores) if all_scores else np.zeros(0),
            'labels': np.array(all_labels) if all_labels else np.zeros(0, dtype=np.int32),
            'drivable_map': drivable_map,
            'lane_map': lane_map,
            'num_detections': len(all_boxes),
        }

        print(f"\n检测到 {len(all_boxes)} 个目标")
        if len(all_boxes) > 0:
            for i, (box, score, label) in enumerate(
                zip(results['boxes_3d'][:5], results['scores'][:5], results['labels'][:5])
            ):
                print(f"  [{i}] 类别={label} 位置=({box[0]:.1f}, {box[1]:.1f}) "
                      f"尺寸=({box[3]:.1f}, {box[4]:.1f}, {box[5]:.1f}) 置信度={score:.3f}")

        return results


def main():
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 初始化推理器
    inferencer = BEVInference(args.checkpoint, args.device)

    # 执行推理
    results = inferencer.inference(
        args.input,
        score_threshold=args.score_threshold,
        top_k=args.top_k,
    )

    # 保存结果
    result_path = os.path.join(args.output, 'results.json')
    serializable = {
        'num_detections': int(results['num_detections']),
        'boxes_3d': results['boxes_3d'].tolist() if len(results['boxes_3d']) > 0 else [],
        'scores': results['scores'].tolist(),
        'labels': results['labels'].tolist(),
    }
    with open(result_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"结果已保存: {result_path}")

    # 保存分割图
    if results['drivable_map'] is not None:
        drivable_path = os.path.join(args.output, 'drivable_map.png')
        cv2.imwrite(drivable_path,
                    (results['drivable_map'] * 255).astype(np.uint8))
        print(f"可行驶区域图: {drivable_path}")

    if results['lane_map'] is not None:
        lane_path = os.path.join(args.output, 'lane_map.png')
        cv2.imwrite(lane_path,
                    (results['lane_map'] * 255).astype(np.uint8))
        print(f"车道线图: {lane_path}")

    # 可视化
    if args.visualize:
        # 准备检测结果（与 visualize_bev 兼容的格式）
        det_result = {
            'boxes_3d': torch.from_numpy(results['boxes_3d']).float(),
            'scores': torch.from_numpy(results['scores']).float(),
            'labels': torch.from_numpy(results['labels']).long(),
        }

        visualize_bev(
            detections=[det_result],
            drivable_map=results['drivable_map'],
            lane_map=results['lane_map'],
            save_path=os.path.join(args.output, 'bev_visualization.png'),
        )

        print(f"可视化结果: {os.path.join(args.output, 'bev_visualization.png')}")


if __name__ == '__main__':
    main()
