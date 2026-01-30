# Benchmark 快速开始指南

## 📋 项目结构

```
benchmark/
├── configs/
│   └── benchmark_config.yaml      # 统一配置文件
├── scripts/
│   ├── train_all.py              # 训练 YOLO/RT-DETR 模型
│   ├── train_faster_rcnn.py      # 训练 Faster R-CNN (单独)
│   ├── eval_all.py               # 统一评估所有模型
│   └── eval_faster_rcnn.py       # 评估 Faster R-CNN (可选)
├── experiments/                   # 实验输出
└── results/                       # 评估结果
```

## 🚀 完整工作流程

### 方案 A: 只测试 YOLO 和 RT-DETR (推荐,最简单)

```bash
cd /home/aiuser/work/ultralytics/benchmark/scripts

# 1. 训练所有 Ultralytics 模型
python train_all.py

# 2. 评估对比
python eval_all.py
```

### 方案 B: 包含 Faster R-CNN 的完整对比

```bash
cd /home/aiuser/work/ultralytics/benchmark/scripts

# 1. 训练 YOLO 和 RT-DETR
python train_all.py

# 2. 安装 Detectron2 (首次需要)
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# 3. 训练 Faster R-CNN
python train_faster_rcnn.py

# 4. 评估 Faster R-CNN (可选,eval_all.py 会自动加载结果)
python eval_faster_rcnn.py

# 5. 统一评估所有模型
python eval_all.py
```

## 📊 预期输出

### 1. 训练输出
```
runs/detect/benchmark/
├── yolo11s/
│   └── weights/best.pt
├── yolov8s/
│   └── weights/best.pt
├── rtdetr_exp/
│   └── weights/best.pt
└── faster_rcnn/          # 如果训练了
    └── model_final.pth
```

### 2. 评估结果
```
benchmark/results/benchmark_comparison.csv
```

示例输出:
```
Model          mAP@0.5  mAP@0.5:0.95  Precision  Recall  Speed(ms)  FPS    Params(M)
YOLO11s        0.8234   0.6781        0.8456     0.7823  12.5       80.0   9.4
YOLOv8s        0.8156   0.6723        0.8312     0.7756  11.8       84.7   11.2
RT-DETR-l      0.8345   0.6892        0.8567     0.7945  45.2       22.1   32.0
Faster R-CNN   0.8412   0.7023        0.8634     N/A     N/A        N/A    41.5
```

## 🔧 配置调整

编辑 `benchmark/configs/benchmark_config.yaml`:

```yaml
# 通用训练配置
train:
  epochs: 200        # 训练轮次
  imgsz: 960        # 图像尺寸
  batch: 16         # 批次大小
  device: '0'       # GPU 设备
  workers: 0        # 数据加载线程(0 可节省显存)

# 模型配置
models:
  yolo11s:
    model: yolo11s.pt
    lr0: 0.01

  yolov8s:
    model: yolov8s.pt
    lr0: 0.01

  rtdetr_l:
    model: rtdetr-l.pt
    lr0: 0.0001
    batch: 4        # RT-DETR 需要更小的 batch
```

## ⚠️ 常见问题

### Q1: 显存不足怎么办?
**解决方案:**
```yaml
train:
  batch: 8          # 降低 batch size
  workers: 0        # 关闭多进程加载
```

对于 Faster R-CNN:
```python
# 在 train_faster_rcnn.py 中修改:
cfg.SOLVER.IMS_PER_BATCH = 1  # 降低到 1
```

### Q2: 为什么 Faster R-CNN 要单独训练?
**原因:**
- Ultralytics 只支持单阶段检测器(YOLO, RT-DETR)
- Faster R-CNN 是两阶段检测器,需要不同框架(Detectron2)
- 使用独立脚本可以保持架构解耦

### Q3: 能不能跳过某个模型?
**可以!** 在配置文件中注释掉不需要的模型:
```yaml
models:
  yolo11s:
    ...
  # yolov8s:  # 注释掉就不会训练
  #   ...
```

### Q4: 已有训练结果,不想重新训练?
**配置复用:**
```yaml
models:
  yolov8s:
    model: yolov8s.pt
    existing_results: /path/to/previous/training
```

### Q5: Detectron2 安装失败?
**尝试预编译版本:**
```bash
# CUDA 11.8
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html

# CUDA 12.1
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.1/index.html
```

## 🎯 使用建议

### 快速验证 (< 1小时)
```yaml
train:
  epochs: 50
  imgsz: 640
```

### 完整对比 (推荐)
```yaml
train:
  epochs: 200
  imgsz: 960
```

### 论文实验 (最佳精度)
```yaml
train:
  epochs: 300
  imgsz: 1280
  patience: 100
```

## 📈 结果分析

运行 `eval_all.py` 后,你会得到:

1. **CSV 文件**: `results/benchmark_comparison.csv`
2. **控制台输出**: 包含性能排名
3. **模型排序**: 按 mAP 和 FPS 分别排序

示例分析:
```
🏆 性能排名:

  按mAP@0.5排序:
  Faster R-CNN    - 0.8412  👈 最高精度,但速度慢
  RT-DETR-l       - 0.8345  👈 Transformer架构,精度高
  YOLO11s         - 0.8234  👈 最新架构,平衡好
  YOLOv8s         - 0.8156  👈 基线模型

  按FPS排序:
  YOLOv8s         - 84.7 FPS  👈 最快
  YOLO11s         - 80.0 FPS  👈 新架构,稍慢
  RT-DETR-l       - 22.1 FPS  👈 Transformer较慢
  Faster R-CNN    - N/A       👈 两阶段最慢
```

## 🔬 进阶使用

### 单独评估某个模型
```python
from ultralytics import YOLO

model = YOLO('runs/detect/benchmark/yolov8s/weights/best.pt')
metrics = model.val(data='datasets/.../data.yaml')
print(f"mAP50: {metrics.box.map50}")
```

### 导出模型用于部署
```python
model = YOLO('runs/detect/benchmark/yolov8s/weights/best.pt')
model.export(format='onnx')  # 或 'tensorrt', 'coreml' 等
```

### 推理测试
```python
model = YOLO('runs/detect/benchmark/yolov8s/weights/best.pt')
results = model.predict('test_image.jpg', save=True)
```

## 📚 参考资料

- [Ultralytics 文档](https://docs.ultralytics.com)
- [Detectron2 文档](https://detectron2.readthedocs.io)
- [FASTER_RCNN_SETUP.md](FASTER_RCNN_SETUP.md) - Faster R-CNN 详细说明
