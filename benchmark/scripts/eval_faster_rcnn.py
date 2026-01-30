"""
Faster R-CNN 评估脚本
用于与其他模型进行对比
"""
import json
import yaml
from pathlib import Path
import torch

def check_dependencies():
    """检查依赖"""
    try:
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset
        from detectron2.data import build_detection_test_loader
        return True
    except ImportError:
        print("❌ Detectron2 未安装,请先运行: pip install detectron2")
        return False

def evaluate_faster_rcnn():
    """评估 Faster R-CNN"""
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader
    from detectron2 import model_zoo

    print("="*60)
    print("📊 Faster R-CNN 评估")
    print("="*60)

    if not check_dependencies():
        return None

    # 1. 加载训练配置
    model_dir = Path('/home/aiuser/work/ultralytics/runs/detect/benchmark/faster_rcnn')
    if not model_dir.exists():
        print(f"❌ 未找到训练结果: {model_dir}")
        print("请先运行 train_faster_rcnn.py")
        return None

    info_path = model_dir / 'model_info.json'
    if not info_path.exists():
        print(f"❌ 未找到模型信息: {info_path}")
        return None

    with open(info_path, 'r') as f:
        model_info = json.load(f)

    # 2. 重建配置
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    ))

    # 加载最佳权重
    best_model = model_dir / 'model_final.pth'
    if not best_model.exists():
        print(f"❌ 未找到最终模型: {best_model}")
        return None

    cfg.MODEL.WEIGHTS = str(best_model)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = model_info['num_classes']
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.DATASETS.TEST = ("wind_turbine_val",)

    # 3. 评估
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("wind_turbine_val", cfg, False, output_dir=str(model_dir))
    val_loader = build_detection_test_loader(cfg, "wind_turbine_val")

    print("\n开始评估...")
    results = inference_on_dataset(predictor.model, val_loader, evaluator)

    # 4. 提取关键指标
    metrics = {
        'model': 'Faster R-CNN',
        'mAP50': results['bbox']['AP50'] / 100,
        'mAP50-95': results['bbox']['AP'] / 100,
        'precision': results['bbox'].get('AP75', 0) / 100,  # 近似
        'params': sum(p.numel() for p in predictor.model.parameters()) / 1e6,
    }

    # 保存结果
    results_file = model_dir / 'eval_results.json'
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*60)
    print("📊 评估结果:")
    print("="*60)
    print(f"mAP@0.5:      {metrics['mAP50']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
    print(f"参数量:       {metrics['params']:.2f}M")
    print(f"\n结果已保存到: {results_file}")

    return metrics

if __name__ == '__main__':
    evaluate_faster_rcnn()
