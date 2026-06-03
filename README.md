# BEV 视角下多任务自动驾驶感知算法研究

> **深度学习模型开发 | 端到端 BEV 感知系统**

基于 BEVFormer 框架，融合自主改进的双分支 ResNet + 注意力机制骨干网络与多尺度注意力模块，
实现从环视相机图像到 BEV 空间的端到端感知，同时输出 **3D 检测框**、**可行驶区域** 与 **车道线**。

---

## 🏗️ 项目架构

```
环视相机图像 [B, N_cam, 3, H, W]
        │
┌───────▼────────────────────────────────┐
│  双分支 ResNet + 注意力骨干网络          │  ← 借鉴遥感变化检测
│  ├─ 空间细节分支 (标准 ResNet)          │
│  ├─ 语义分支 (空洞卷积 + CBAM)          │
│  ├─ 跨分支门控融合                     │
│  └─ FPN 多尺度输出                     │
└───────┬────────────────────────────────┘
        │  多尺度图像特征
┌───────▼────────────────────────────────┐
│  BEV 特征变换编码器 (BEVFormer 风格)    │
│  ├─ 空间交叉注意力 (可变形注意力)       │
│  └─ 相机感知位置编码                   │
└───────┬────────────────────────────────┘
        │  BEV 特征 [B, H_bev*W_bev, C]
┌───────▼────────────────────────────────┐
│  时序融合模块 (可选)                    │
│  └─ 时序自注意力 → 历史帧融合          │
└───────┬────────────────────────────────┘
        │
┌───────▼────────────────────────────────┐
│  多尺度注意力模块 (自主改进)             │
│  ├─ BiFPN 金字塔融合                   │
│  ├─ 跨尺度自注意力                     │
│  └─ 自适应尺度聚合                     │
└───────┬────────────────────────────────┘
        │  增强 BEV 特征
┌───────┴──────────────┬─────────────────┐
│                      │                 │
▼                      ▼                 ▼
┌──────────┐  ┌──────────────┐  ┌──────────────┐
│ 3D 检测头 │  │可行驶区域分割头│  │ 车道线分割头  │
│CenterPoint│  │  FCN 解码器   │  │  FCN 解码器   │
└─────┬────┘  └──────┬───────┘  └──────┬───────┘
      │              │                 │
      ▼              ▼                 ▼
  3D 检测框      可行驶区域图        车道线图
  (x,y,z,l,w,h,θ)  (H×W)            (H×W)
```

---

## 📁 项目结构

```
driver_detection/
├── configs/
│   ├── __init__.py
│   └── bevformer_config.py       # 模型与训练配置
├── models/
│   ├── __init__.py                # 模块导出
│   ├── backbone.py                # 双分支 ResNet + CBAM 注意力骨干
│   ├── bev_encoder.py             # BEV 特征变换编码器 (spatial cross-attention)
│   ├── multi_scale_attention.py   # 多尺度注意力模块 (BiFPN + cross-scale attention)
│   ├── temporal_fusion.py         # 时序融合 (temporal self-attention)
│   ├── detection_head.py          # 3D 目标检测头 (CenterPoint 范式)
│   ├── segmentation_head.py       # 分割头 (可行驶区域 + 车道线)
│   └── bev_perception.py          # 完整模型组装
├── data/
│   ├── __init__.py
│   ├── dataset.py                 # 数据集定义与 DataLoader (通用)
│   ├── nuscenes_dataset.py        # NuScenes 完整数据加载器 ★
│   ├── transforms.py              # 数据预处理与增强
│   └── bev_grid.py                # BEV 网格坐标变换工具
├── losses/
│   ├── __init__.py
│   ├── detection_loss.py          # 检测损失 (Focal Loss + L1)
│   └── segmentation_loss.py       # 分割损失 (CE + Dice)
├── utils/
│   ├── __init__.py
│   ├── metrics.py                 # 评估指标 (mAP, mIoU, NDS)
│   └── visualization.py           # BEV 可视化
├── train.py                       # 训练入口脚本
├── inference.py                   # 推理入口脚本
├── requirements.txt               # 依赖包
└── README.md                      # 项目文档
```

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3+ (推荐用于 GPU 训练)

### 安装依赖

```bash
cd driver_detection
pip install -r requirements.txt
```

### 方式一：演示数据快速测试（无需下载任何数据集）

```bash
# 一键启动：自动生成演示数据 + 训练 5 轮
python quickstart.py

# 或分步：
python generate_demo_data.py --output_dir ./demo_data --num_samples 50
python train.py --data_root ./demo_data --epochs 5
```

### 方式二：使用 NuScenes 真实数据集（推荐）

#### 1. 下载数据

从 [nuScenes 官网](https://www.nuscenes.org/download) 注册并下载：

| 文件 | 大小 | 说明 |
|------|------|------|
| `v1.0-trainval_meta.tgz` | ~5 MB | 标注元数据（必须） |
| `v1.0-trainval01_blobs.tgz` ~ `10_blobs.tgz` | ~340 GB | 相机图像（10 个分卷） |
| `v1.0-mini.tgz` | ~4 GB | **Mini 版（快速调试，推荐先下这个）** |

```bash
# 解压后目录结构应如下：
/path/to/nuscenes/
├── maps/                    # HD Map (自动下载)
├── samples/
│   ├── CAM_FRONT/          # 前视相机
│   ├── CAM_FRONT_LEFT/     # 前左
│   ├── CAM_FRONT_RIGHT/    # 前右
│   ├── CAM_BACK/           # 后视
│   ├── CAM_BACK_LEFT/      # 后左
│   └── CAM_BACK_RIGHT/     # 后右
└── v1.0-mini/ 或 v1.0-trainval/   # 标注 JSON
```

#### 2. 安装 nuScenes 开发工具

```bash
pip install nuscenes-devkit
```

#### 3. 测试数据加载器

```bash
# 验证 nuScenes 数据能正常加载
python data/nuscenes_dataset.py --data_root /path/to/nuscenes --version v1.0-mini
```

预期输出：
```
[NuScenes] 加载完成: 404 个 mini 样本
Batch 0:
  images:       torch.Size([1, 6, 3, 900, 1600])
  camera_params: torch.Size([1, 6, 16])
  boxes_3d:     torch.Size([1, N, 9])
  drivable:     torch.Size([1, 128, 128])
  lane:         torch.Size([1, 128, 128])
```

#### 4. 启动训练

```bash
# Mini 模式快速验证（10 场景，~400 帧）
python train_nuscenes.py --data_root /path/to/nuscenes --mini --epochs 5

# 完整训练（700 场景，~28k 帧）
python train_nuscenes.py --data_root /path/to/nuscenes --epochs 24

# 不使用 HD Map（仅 3D 检测，训练更快）
python train_nuscenes.py --data_root /path/to/nuscenes --no_map

# 使用 ResNet101 骨干
python train_nuscenes.py --data_root /path/to/nuscenes --backbone resnet101
```

### 方式三：使用其他公开数据集

| 数据集 | 相机数 | 3D框 | 可行驶区域 | 车道线 | 接入方式 |
|--------|--------|------|-----------|--------|---------|
| **Waymo Open** | 5 | ✅ | ❌ | ❌ | 参考 `nuscenes_dataset.py` 模式编写 `waymo_dataset.py`，需安装 `waymo-open-dataset-tf` |
| **Argoverse 2** | 7 | ✅ | ✅ | ✅ | 参考 `nuscenes_dataset.py`，需安装 `av2` 包 |
| **KITTI-360** | 4 | ✅ | ❌ | ❌ | 数据量小，适合调试；自定义 JSON 标注接入即可 |

接入新数据集的步骤：
1. 在 `data/` 下新建 `xxx_dataset.py`
2. 实现 `__len__` 和 `__getitem__`，返回与 `collate_fn` 兼容的字典
3. 在 `data/__init__.py` 中导出

### 推理

```bash
# 单帧推理
python inference.py --checkpoint output/best_model.pth --input /path/to/images/

# 可视化输出
python inference.py --checkpoint output/best_model.pth \
    --input demo_data/ --visualize --output ./results/
```

---

## 🔬 核心创新点

### 1. 双分支 ResNet + 注意力机制骨干网络

借鉴遥感变化检测中的双分支架构（如 ChangeNet、BIT），设计适用于自动驾驶环视感知的骨干网络：

| 分支 | 特点 | 作用 |
|------|------|------|
| **空间细节分支** | 标准 ResNet 残差块，保留下采样节奏 | 保留高分辨率空间定位信息 |
| **语义分支** | 空洞卷积替换步长下采样 + CBAM 注意力 | 扩大感受野，增强语义理解 |
| **跨分支融合** | 门控机制自适应学习分支权重 | 软特征选择，最优融合 |

### 2. 轻量 BEV 特征变换模块

基于 BEVFormer 的空间交叉注意力，采用轻量化设计：

- 可变形注意力：每个 BEV 查询只关注 4 个关键采样点
- 减少 Transformer 层数（6→3）
- 相机感知位置编码：显式编码相机内外参

### 3. 自主改进的多尺度注意力模块

在标准 FPN 基础上引入自注意力机制，使不同尺度的 BEV 特征可以相互关注：

- **BiFPN**：带可学习权重的双向特征金字塔
- **跨尺度自注意力**：不同尺度特征互相查询，学习全局上下文与局部细节的最优融合
- **自适应尺度聚合**：软权重学习各尺度重要性

### 4. 多任务联合学习

单一 BEV 特征同时支持三项任务：

| 任务 | 方法 | 损失函数 |
|------|------|----------|
| 3D 目标检测 | CenterPoint 范式热力图 | Focal Loss + Smooth L1 |
| 可行驶区域 | FCN 语义分割 | CrossEntropy + Dice Loss |
| 车道线 | FCN 语义分割（任务间交叉引导） | CrossEntropy + Dice Loss |

---

## 📊 模型配置

所有配置集中于 `configs/bevformer_config.py`，主要参数：

```python
@dataclass
class BEVPerceptionConfig:
    # BEV 网格
    bev_grid: BEVGridConfig       # 范围: ±51.2m, 分辨率: 0.8m

    # 骨干网络
    backbone: BackboneConfig      # ResNet50/101, 双分支, CBAM

    # BEV 编码器
    bev_encoder: BEVEncoderConfig # 3层空间交叉注意力, 256维

    # 多尺度注意力
    multi_scale_attn: MultiScaleAttentionConfig  # BiFPN + 跨尺度自注意力

    # 训练参数
    image_size: (900, 1600)
    batch_size: 1
    learning_rate: 2e-4
    max_epochs: 24
```

---

## 📈 评估指标

- **3D 检测**: mAP (mean Average Precision) @ IoU 0.5/0.7, NDS
- **可行驶区域**: mIoU (mean Intersection over Union)
- **车道线**: mIoU

---

## 📚 参考工作

- **BEVFormer**: Li et al., "BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers", ECCV 2022
- **CenterPoint**: Yin et al., "Center-based 3D Object Detection and Tracking", CVPR 2021
- **CBAM**: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
- **BiFPN**: Tan et al., "EfficientDet: Scalable and Efficient Object Detection", CVPR 2020
- **遥感变化检测双分支架构**: ChangeNet, BIT (Remote Sensing)

---

## 📝 许可证

本项目仅用于学术研究与学习目的。
