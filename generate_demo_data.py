"""
生成演示用虚拟数据集
====================
快速生成一组随机样本用于验证模型训练流程是否正常运行。
"""

import numpy as np
import json
import os
import cv2


def generate_demo_dataset(output_dir: str, num_samples: int = 50):
    """
    生成演示数据集

    目录结构:
      demo_data/
      ├── annotations.json      # 标注文件
      ├── cam_front_*.jpg       # 前置相机图像
      ├── cam_front_left_*.jpg
      ├── cam_front_right_*.jpg
      ├── cam_back_*.jpg
      ├── cam_back_left_*.jpg
      ├── cam_back_right_*.jpg
      ├── drivable_*.png        # 可行驶区域掩码
      └── lane_*.png            # 车道线掩码
    """
    os.makedirs(output_dir, exist_ok=True)

    cam_names = [
        'cam_front', 'cam_front_left', 'cam_front_right',
        'cam_back', 'cam_back_left', 'cam_back_right',
    ]
    img_h, img_w = 900, 1600
    bev_h, bev_w = 128, 128

    samples = []
    for i in range(num_samples):
        sample = {
            'token': f'demo_{i:04d}',
            'images': [],
            'camera_params': [],
            'boxes_3d': [],
            'labels': [],
            'drivable_mask': '',
            'lane_mask': '',
        }

        for cam in cam_names:
            img_name = f'{cam}_{i:04d}.jpg'
            img_path = os.path.join(output_dir, img_name)

            # 生成随机彩色图像（模拟道路场景）
            img = np.random.randint(50, 200, (img_h, img_w, 3), dtype=np.uint8)
            # 添加一些简单的"道路"结构
            cv2.rectangle(img, (0, img_h // 2), (img_w, img_h), (100, 100, 100), -1)
            cv2.imwrite(img_path, img)

            sample['images'].append(img_name)
            # 随机相机参数 [fx, fy, cx, cy, qx, qy, qz, qw, tx, ty, tz, ...]
            sample['camera_params'].append(
                np.random.randn(16).tolist()
            )

        # 生成 3D 框标注（随机 5~15 个目标）
        num_boxes = np.random.randint(5, 15)
        for _ in range(num_boxes):
            box = [
                np.random.uniform(-40, 40),   # cx
                np.random.uniform(-40, 40),   # cy
                np.random.uniform(-2, 0),     # cz
                np.random.uniform(3, 8),      # length
                np.random.uniform(1.5, 3),    # width
                np.random.uniform(1.5, 3),    # height
                np.random.uniform(-np.pi, np.pi),  # yaw
                np.random.uniform(-5, 5),     # vx
                np.random.uniform(-5, 5),     # vy
            ]
            sample['boxes_3d'].append(box)
            sample['labels'].append(np.random.randint(0, 10))

        # 生成可行驶区域掩码 (BEV 128×128)
        drivable_name = f'drivable_{i:04d}.png'
        drivable_path = os.path.join(output_dir, drivable_name)
        drivable_mask = np.zeros((bev_h, bev_w), dtype=np.uint8)
        # 中心区域为可行驶区域
        drivable_mask[20:bev_h - 20, 30:bev_w - 30] = 1
        cv2.imwrite(drivable_path, drivable_mask * 255)
        sample['drivable_mask'] = drivable_name

        # 生成车道线掩码
        lane_name = f'lane_{i:04d}.png'
        lane_path = os.path.join(output_dir, lane_name)
        lane_mask = np.zeros((bev_h, bev_w), dtype=np.uint8)
        # 几条竖线模拟车道
        for x in range(40, bev_w, 20):
            lane_mask[10:bev_h - 10, x - 1:x + 2] = 1
        cv2.imwrite(lane_path, lane_mask * 255)
        sample['lane_mask'] = lane_name

        samples.append(sample)
        print(f"  生成样本 {i + 1}/{num_samples}: {sample['token']} "
              f"({num_boxes} boxes)")

    # 保存标注文件
    ann_path = os.path.join(output_dir, 'annotations.json')
    with open(ann_path, 'w') as f:
        json.dump({'samples': samples}, f, indent=2)

    print(f"\n数据集生成完成!")
    print(f"  目录: {output_dir}")
    print(f"  标注: {ann_path}")
    print(f"  样本数: {num_samples}")
    print(f"  图像尺寸: {img_h}×{img_w}")
    print(f"  BEV 尺寸: {bev_h}×{bev_w}")

    return output_dir


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./demo_data')
    parser.add_argument('--num_samples', type=int, default=50)
    args = parser.parse_args()

    generate_demo_dataset(args.output_dir, args.num_samples)
