"""
快速启动脚本
============
一键生成演示数据并开始训练。

用法:
    python quickstart.py                     # 生成 50 个演示样本并训练 5 轮
    python quickstart.py --epochs 2 --debug  # 快速冒烟测试（2 轮）
"""

import subprocess
import sys
import os
import argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    parser = argparse.ArgumentParser(description='BEV 多任务感知快速启动')
    parser.add_argument('--demo_dir', type=str, default='./demo_data')
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lightweight', action='store_true', default=True,
                        help='使用轻量化配置 (默认开启)')
    parser.add_argument('--full', action='store_true',
                        help='使用完整配置 (ResNet50, ~50M 参数)')
    parser.add_argument('--debug', action='store_true', help='仅 2 轮快速测试')
    args = parser.parse_args()

    demo_dir = os.path.abspath(args.demo_dir)
    epochs = 2 if args.debug else args.epochs

    # ---- Step 1: 生成演示数据 ----
    if not os.path.exists(os.path.join(demo_dir, 'annotations.json')):
        print("=" * 60)
        print("Step 1: 生成演示数据集...")
        print("=" * 60)
        result = subprocess.run(
            [sys.executable, 'generate_demo_data.py',
             '--output_dir', demo_dir,
             '--num_samples', str(args.num_samples)],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if result.returncode != 0:
            print("数据生成失败!")
            sys.exit(1)
    else:
        print(f"数据集已存在: {demo_dir}")

    # ---- Step 2: 开始训练 ----
    print("\n" + "=" * 60)
    print(f"Step 2: 开始训练 ({epochs} epochs)...")
    print("=" * 60)

    # 设置 CUDA 内存配置（减少碎片）
    env = os.environ.copy()
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    train_cmd = [
        sys.executable, 'train.py',
        '--data_root', demo_dir,
        '--epochs', str(epochs),
        '--batch_size', str(args.batch_size),
        '--output_dir', './output',
        '--val_interval', '99',
        '--save_interval', str(epochs + 1),
        '--log_interval', '5',
    ]

    if not args.full:
        train_cmd.append('--lightweight')

    result = subprocess.run(
        train_cmd,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    if result.returncode == 0:
        print("\n✅ 训练完成!")
    else:
        print(f"\n❌ 训练异常退出 (code {result.returncode})")


if __name__ == '__main__':
    main()
