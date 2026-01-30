"""
Faster R-CNN 训练脚本 (基于 Detectron2)
独立于 Ultralytics 框架运行

显存优化配置 (针对12GB GPU):
- Batch size: 1 (Faster R-CNN 显存占用远大于 YOLO)
- ROI batch size: 64 (默认512, 大幅降低)
- 梯度累积: 4 steps (等效 batch_size=4)
- Image size: 960x960 (与其他模型一致)

对比其他模型配置:
- YOLO11s/YOLOv8s: batch=16, imgsz=960
- RT-DETR-L: batch=4, imgsz=960
- Faster R-CNN: batch=1 (实际等效4), imgsz=960
"""
import os
import yaml
import torch
import json
from pathlib import Path
from datetime import datetime

def check_dependencies():
    """检查并安装必要依赖"""
    try:
        import detectron2
        print("✅ Detectron2 已安装")
    except ImportError:
        print("❌ 未检测到 Detectron2")
        print("\n请运行以下命令安装:")
        print("pip install 'git+https://github.com/facebookresearch/detectron2.git'")
        print("\n或使用预编译版本:")
        print("python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html")
        return False

    try:
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultTrainer
        from detectron2.data import DatasetCatalog, MetadataCatalog
        from detectron2 import model_zoo
        print("✅ Detectron2 导入成功")
        return True
    except Exception as e:
        print(f"❌ Detectron2 导入失败: {e}")
        return False

def convert_yolo_to_coco(dataset_yaml_path):
    """
    将 YOLO 格式数据集转换为 COCO 格式

    Args:
        dataset_yaml_path: YOLO 数据集的 yaml 文件路径

    Returns:
        train_json, val_json: COCO 格式的标注文件路径
    """
    from detectron2.structures import BoxMode
    from detectron2.data import DatasetCatalog, MetadataCatalog
    import cv2

    # 读取 YOLO 数据集配置
    with open(dataset_yaml_path, 'r') as f:
        dataset_config = yaml.safe_load(f)

    dataset_root = Path(dataset_yaml_path).parent
    train_img_dir = dataset_root / dataset_config.get('train', 'images/train')
    val_img_dir = dataset_root / dataset_config.get('val', 'images/val')

    train_label_dir = str(train_img_dir).replace('images', 'labels')
    val_label_dir = str(val_img_dir).replace('images', 'labels')

    def yolo_to_coco_dict(img_dir, label_dir, class_names):
        """转换单个数据集分割"""
        dataset_dicts = []
        img_dir = Path(img_dir)
        label_dir = Path(label_dir)

        for idx, img_path in enumerate(img_dir.glob('*.jpg')):
            record = {}

            # 读取图像尺寸
            img = cv2.imread(str(img_path))
            height, width = img.shape[:2]

            record["file_name"] = str(img_path)
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width

            # 读取标注
            label_path = label_dir / f"{img_path.stem}.txt"
            objs = []

            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue

                        class_id, x_center, y_center, w, h = map(float, parts)

                        # YOLO 格式 (归一化) 转 COCO 格式 (绝对坐标)
                        x_center *= width
                        y_center *= height
                        w *= width
                        h *= height

                        x_min = x_center - w / 2
                        y_min = y_center - h / 2

                        obj = {
                            "bbox": [x_min, y_min, w, h],
                            "bbox_mode": BoxMode.XYWH_ABS,
                            "category_id": int(class_id),
                        }
                        objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)

        return dataset_dicts

    # 注册数据集
    class_names = dataset_config.get('names', [])

    train_dicts = yolo_to_coco_dict(train_img_dir, train_label_dir, class_names)
    val_dicts = yolo_to_coco_dict(val_img_dir, val_label_dir, class_names)

    # 注册到 Detectron2
    DatasetCatalog.register("wind_turbine_train", lambda: train_dicts)
    MetadataCatalog.get("wind_turbine_train").set(thing_classes=class_names)

    DatasetCatalog.register("wind_turbine_val", lambda: val_dicts)
    MetadataCatalog.get("wind_turbine_val").set(thing_classes=class_names)

    print(f"✅ 数据集转换完成:")
    print(f"   训练集: {len(train_dicts)} 张图像")
    print(f"   验证集: {len(val_dicts)} 张图像")
    print(f"   类别数: {len(class_names)} - {class_names}")

    return "wind_turbine_train", "wind_turbine_val", len(class_names)

def setup_faster_rcnn_config(num_classes, output_dir, config_yaml):
    """配置 Faster R-CNN"""
    from detectron2.config import get_cfg
    from detectron2 import model_zoo

    cfg = get_cfg()

    # 基础模型配置
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    ))

    # 数据集配置
    cfg.DATASETS.TRAIN = ("wind_turbine_train",)
    cfg.DATASETS.TEST = ("wind_turbine_val",)
    cfg.DATALOADER.NUM_WORKERS = config_yaml['train'].get('workers', 0)

    # 模型配置
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    # 降低 ROI batch size 以减少显存占用 (默认512, 降至64)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.DEVICE = f"cuda:{config_yaml['train'].get('device', '0')}" if torch.cuda.is_available() else "cpu"

    # 训练配置 - 针对12GB显存优化
    # Faster R-CNN 显存占用大，batch size 设为 1
    cfg.SOLVER.IMS_PER_BATCH = 1
    # 调整学习率以适应小 batch size (原始 0.00025 对应 batch=2)
    cfg.SOLVER.BASE_LR = 0.000125
    # 计算总迭代次数：epochs * (数据集大小 / batch_size)
    # 粗略估算: 200 epochs * 150 images / 1 = 30000 iterations
    cfg.SOLVER.MAX_ITER = 30000
    cfg.SOLVER.STEPS = []  # 不使用学习率衰减
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000
    # 梯度累积，模拟更大的 batch size
    cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS = 4  # 等效 batch_size=4

    # 输入图像尺寸 - 与其他模型保持一致
    cfg.INPUT.MIN_SIZE_TRAIN = (config_yaml['train'].get('imgsz', 960),)
    cfg.INPUT.MAX_SIZE_TRAIN = config_yaml['train'].get('imgsz', 960)
    cfg.INPUT.MIN_SIZE_TEST = config_yaml['train'].get('imgsz', 960)
    cfg.INPUT.MAX_SIZE_TEST = config_yaml['train'].get('imgsz', 960)

    # 输出配置
    cfg.OUTPUT_DIR = str(output_dir)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # 打印配置摘要
    print("\n" + "="*60)
    print("📋 Faster R-CNN 配置摘要")
    print("="*60)
    print(f"图像尺寸: {cfg.INPUT.MIN_SIZE_TRAIN[0]}x{cfg.INPUT.MAX_SIZE_TRAIN}")
    print(f"Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"梯度累积: {cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS} steps (等效 batch={cfg.SOLVER.IMS_PER_BATCH * cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS})")
    print(f"ROI batch size: {cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE}")
    print(f"学习率: {cfg.SOLVER.BASE_LR}")
    print(f"最大迭代: {cfg.SOLVER.MAX_ITER}")
    print(f"类别数: {num_classes}")
    print(f"设备: {cfg.MODEL.DEVICE}")
    print("="*60 + "\n")

    return cfg

def train_faster_rcnn():
    """主训练函数"""
    from detectron2.engine import DefaultTrainer
    from detectron2.evaluation import COCOEvaluator

    print("="*60)
    print("🚀 Faster R-CNN 训练")
    print("="*60)

    # 0. 清理 GPU 显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🔧 GPU 显存清理完成")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # 1. 检查依赖
    if not check_dependencies():
        return

    # 2. 加载配置
    config_path = Path(__file__).parent.parent / 'configs' / 'benchmark_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 3. 转换数据集
    dataset_yaml = Path('/home/aiuser/work/ultralytics') / config['dataset']['path']
    train_name, val_name, num_classes = convert_yolo_to_coco(dataset_yaml)

    # 4. 设置输出目录
    output_dir = Path('/home/aiuser/work/ultralytics/runs/detect/benchmark/faster_rcnn')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 5. 配置模型
    cfg = setup_faster_rcnn_config(num_classes, output_dir, config)

    # 6. 训练
    print("\n" + "="*60)
    print("开始训练...")
    print("="*60)

    class CustomTrainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name):
            return COCOEvaluator(dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR)

    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    print("\n" + "="*60)
    print("✅ Faster R-CNN 训练完成!")
    print(f"📂 模型保存在: {output_dir}")
    print("="*60)

    # 保存配置信息
    info = {
        'model': 'Faster R-CNN (ResNet50-FPN)',
        'framework': 'Detectron2',
        'num_classes': num_classes,
        'output_dir': str(output_dir),
        'trained_at': datetime.now().isoformat()
    }

    with open(output_dir / 'model_info.json', 'w') as f:
        json.dump(info, f, indent=2)

if __name__ == '__main__':
    train_faster_rcnn()
