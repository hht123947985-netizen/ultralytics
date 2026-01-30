# YOLO11 Benchmark Project

## 目标
对比评估以下目标检测模型:
- **YOLO11s**: 最新YOLO架构
- **YOLOv8s**: 行业基线
- **RT-DETR-l**: Transformer架构
- **Faster R-CNN**: 两阶段检测器 (需单独训练)

## 快速开始

### 1. 环境准备
```bash
cd /home/aiuser/work/ultralytics
pip install -r requirements.txt
```

### 2. 配置数据集
编辑 `benchmark/configs/benchmark_config.yaml`，确保数据集路径正确:
```yaml
dataset:
  path: ../datasets/Defect detection of wind turbine blades
```

### 3. 训练所有模型
```bash
cd benchmark/scripts
python train_all.py
```

### 4. 评估并对比
```bash
python eval_all.py
```

## 目录结构
```
benchmark/
├── configs/          # 配置文件
├── scripts/          # 训练/评估脚本
├── experiments/      # 实验输出
└── results/          # 对比结果
```

## 评估指标
- mAP@0.5
- mAP@0.5:0.95
- Precision & Recall
- FPS (推理速度)
- 参数量
- GFLOPs

## Faster R-CNN 训练

由于 Faster R-CNN 使用不同的框架(Detectron2),需要单独训练:

```bash
# 1. 安装 Detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# 2. 训练 Faster R-CNN
python scripts/train_faster_rcnn.py

# 3. 评估
python scripts/eval_faster_rcnn.py
```

详细说明见: [FASTER_RCNN_SETUP.md](FASTER_RCNN_SETUP.md)

## 后续工作
- [x] 添加Faster R-CNN支持
- [ ] 可视化对比图表
- [ ] 消融实验
- [ ] 不同输入尺寸测试
