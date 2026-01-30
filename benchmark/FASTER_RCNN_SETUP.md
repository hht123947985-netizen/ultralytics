# Faster R-CNN 训练指南

## 为什么需要单独处理?

Faster R-CNN 是 **两阶段检测器**,与 Ultralytics 框架中的单阶段检测器(YOLO, RT-DETR)架构不同,需要使用不同的框架。

## 安装 Detectron2

### 方法 1: 从源码安装 (推荐)
```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### 方法 2: 使用预编译版本
针对不同的 CUDA 版本:

**CUDA 11.8 + PyTorch 2.0:**
```bash
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
```

**CUDA 12.1 + PyTorch 2.1:**
```bash
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.1/index.html
```

### 验证安装
```bash
python -c "import detectron2; print(detectron2.__version__)"
```

## 使用方法

### 1. 单独训练 Faster R-CNN
```bash
cd /home/aiuser/work/ultralytics/benchmark/scripts
python train_faster_rcnn.py
```

### 2. 单独评估
```bash
python eval_faster_rcnn.py
```

### 3. 与其他模型一起对比
运行主评估脚本时会自动包含 Faster R-CNN 结果(如果存在):
```bash
python eval_all.py
```

## 输出位置

训练结果将保存在:
```
/home/aiuser/work/ultralytics/runs/detect/benchmark/faster_rcnn/
├── model_final.pth          # 最终模型
├── model_info.json          # 模型元信息
├── eval_results.json        # 评估结果
└── metrics.json             # 训练过程指标
```

## 配置说明

Faster R-CNN 的配置在 `benchmark_config.yaml` 中:

```yaml
models:
  faster_rcnn:
    framework: detectron2
    backbone: ResNet50-FPN
    # 其他参数自动从 train 部分继承
```

## 关键差异

| 特性 | YOLO/RT-DETR | Faster R-CNN |
|------|--------------|--------------|
| 框架 | Ultralytics | Detectron2 |
| 阶段 | 单阶段 | 两阶段 (RPN + ROI Head) |
| 训练方式 | 端到端 | 先RPN后ROI |
| 推理速度 | 快 | 较慢 |
| 精度 | 高 | 更高 (通常) |

## 常见问题

### Q: 为什么不能用 Ultralytics 训练 Faster R-CNN?
A: Ultralytics 专注于单阶段检测器,不支持两阶段架构。

### Q: 可以混合评估吗?
A: 可以!`eval_all.py` 会自动检测并包含 Faster R-CNN 结果。

### Q: 显存不够怎么办?
A: 降低 batch size:
```python
cfg.SOLVER.IMS_PER_BATCH = 1  # 默认是 2
```

## 技术细节

**模型架构**: Faster R-CNN (ResNet50-FPN-3x)
- Backbone: ResNet50 with Feature Pyramid Network
- RPN: Region Proposal Network
- ROI Head: Region of Interest classification + bbox regression
- Input size: 960x960 (与其他模型一致)

**预训练**: COCO dataset (80 classes)
**微调**: Wind turbine blade defect detection (2 classes)
