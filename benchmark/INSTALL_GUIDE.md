# 依赖安装指南

## 当前环境状态

✅ **已安装的依赖**:
- opencv-python (4.13.0.90)
- pandas (3.0.0)
- pycocotools
- fvcore
- iopath
- omegaconf
- hydra-core
- ultralytics (YOLO 框架)

❌ **待安装**:
- detectron2 (用于 Faster R-CNN)

## 安装 Detectron2 的方法

### 方法 1: 直接从本地文件安装 (推荐)

如果你有 detectron2 的离线安装包:

```bash
# 假设你已经下载了 detectron2 源码包
cd /path/to/detectron2
uv pip install -e .
```

### 方法 2: 使用代理

如果你有代理或 VPN:

```bash
# 设置代理
export http_proxy=http://your-proxy:port
export https_proxy=http://your-proxy:port

# 克隆并安装
cd /tmp
git clone https://github.com/facebookresearch/detectron2.git --depth 1
cd detectron2
uv pip install -e .
```

### 方法 3: 手动下载安装包

1. 在能访问 GitHub 的机器上下载:
   ```bash
   wget https://github.com/facebookresearch/detectron2/archive/refs/heads/main.zip
   ```

2. 传输到当前服务器

3. 解压并安装:
   ```bash
   unzip main.zip
   cd detectron2-main
   uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .
   ```

### 方法 4: 跳过 Faster R-CNN (临时方案)

如果暂时无法安装 detectron2，可以先运行其他模型:

```bash
cd /home/aiuser/work/ultralytics/benchmark/scripts

# 只训练 YOLO 和 RT-DETR
python train_all.py

# 只评估 YOLO 和 RT-DETR
python eval_all.py
```

这样仍然可以对比 3 个模型:
- YOLO11s
- YOLOv8s
- RT-DETR-l

## 验证安装

安装完成后验证:

```bash
python -c "import detectron2; print(f'Detectron2 version: {detectron2.__version__}')"
```

## 当前可用的训练脚本

### 1. 训练 YOLO 和 RT-DETR (无需 detectron2)
```bash
cd /home/aiuser/work/ultralytics/benchmark/scripts
python train_all.py
```

### 2. 训练 Faster R-CNN (需要 detectron2)
```bash
cd /home/aiuser/work/ultralytics/benchmark/scripts
python train_faster_rcnn.py
```

### 3. 评估所有模型
```bash
cd /home/aiuser/work/ultralytics/benchmark/scripts
python eval_all.py
```

`eval_all.py` 会自动检测:
- 如果 detectron2 已安装且 Faster R-CNN 已训练,会包含其结果
- 如果没有,会跳过并只评估 YOLO/RT-DETR

## 镜像源配置

为 uv 配置永久镜像源:

```bash
# 创建或编辑配置文件
mkdir -p ~/.config/uv
cat > ~/.config/uv/uv.toml << EOF
[pip]
index-url = "https://pypi.tuna.tsinghua.edu.cn/simple"
extra-index-url = [
    "https://mirrors.aliyun.com/pypi/simple/",
]
EOF
```

或使用环境变量:

```bash
export UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export UV_EXTRA_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
```

## 常用镜像源

- 清华: https://pypi.tuna.tsinghua.edu.cn/simple
- 阿里云: https://mirrors.aliyun.com/pypi/simple/
- 中科大: https://pypi.mirrors.ustc.edu.cn/simple/
- 豆瓣: https://pypi.douban.com/simple/

## 下一步

1. **如果成功安装 detectron2**: 运行 `python train_faster_rcnn.py`
2. **如果暂时无法安装**: 运行 `python train_all.py` 训练其他模型
3. **评估模型**: 运行 `python eval_all.py` 生成对比报告
