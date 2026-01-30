"""
批量评估所有训练好的模型（优化版）
- 自动从配置读取模型路径
- 生成详细对比报告
"""
import yaml
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import time
import torch
import json

def load_faster_rcnn_results():
    """加载 Faster R-CNN 评估结果 (如果存在)"""
    results_path = Path('/home/aiuser/work/ultralytics/runs/detect/benchmark/faster_rcnn/eval_results.json')
    if results_path.exists():
        with open(results_path, 'r') as f:
            data = json.load(f)
            return {
                'Model': 'Faster R-CNN',
                'mAP@0.5': f"{data['mAP50']:.4f}",
                'mAP@0.5:0.95': f"{data['mAP50-95']:.4f}",
                'Precision': f"{data.get('precision', 0):.4f}",
                'Recall': "N/A",  # Detectron2 不直接提供
                'Speed(ms)': "N/A",  # 需要单独测试
                'FPS': "N/A",
                'Params(M)': f"{data['params']:.2f}",
                'GFLOPs': "N/A"
            }
    return None

def load_config(config_path='../configs/benchmark_config.yaml'):
    """加载配置文件"""
    config_path = Path(__file__).parent / config_path
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def find_best_weights(model_name, model_config, common_config):
    """查找模型的最佳权重文件"""
    # 1. 检查是否有existing_results
    if 'existing_results' in model_config:
        weight_path = Path(model_config['existing_results']) / 'weights' / 'best.pt'
        if weight_path.exists():
            return weight_path

    # 2. 在benchmark目录下查找
    project_dir = Path(common_config['project'])
    exp_name = model_config.get('name', model_name)
    weight_path = project_dir / exp_name / 'weights' / 'best.pt'
    if weight_path.exists():
        return weight_path

    return None

def evaluate_model(model_name, weight_path, dataset_config):
    """评估单个模型"""
    print(f"\n{'='*60}")
    print(f"评估模型: {model_name}")
    print(f"权重文件: {weight_path}")
    print(f"{'='*60}")

    if not weight_path or not weight_path.exists():
        print(f"❌ 未找到权重文件")
        return None

    # 加载模型
    model = YOLO(str(weight_path))

    # 验证集评估
    print("📊 开始验证集评估...")
    metrics = model.val(
        data=dataset_config['path'],
        split='val',
        imgsz=640,
        batch=16,
        verbose=False
    )

    # 推理速度测试
    print("⚡ 测试推理速度...")
    # 使用验证集图片测试
    val_dir = Path(dataset_config['path']).parent / 'valid' / 'images'
    if not val_dir.exists():
        # 尝试另一个常见路径
        val_dir = Path(dataset_config['path']).parent / 'val' / 'images'

    if val_dir.exists():
        # 预热
        _ = model.predict(
            source=str(val_dir),
            imgsz=640,
            save=False,
            verbose=False,
            stream=True,
            max_det=300
        )

        # 正式测速
        results_list = list(model.predict(
            source=str(val_dir),
            imgsz=640,
            save=False,
            verbose=False,
            stream=True,
            max_det=300
        ))

        # 计算平均速度
        total_time = sum(r.speed['inference'] for r in results_list)
        avg_speed = total_time / len(results_list) if results_list else 0
    else:
        avg_speed = metrics.speed['inference']

    # 计算FPS
    fps = 1000.0 / avg_speed if avg_speed > 0 else 0

    # 模型参数统计
    params = sum(p.numel() for p in model.model.parameters()) / 1e6

    # 计算GFLOPs
    try:
        # YOLO模型通常有flops属性
        gflops = model.model.flops / 1e9 if hasattr(model.model, 'flops') else 0
        if gflops == 0:
            # 备用方法：手动计算
            from ultralytics.utils.ops import Profile
            # 这里简化处理，实际可以更精确
            gflops = 0  # 暂时设为0
    except:
        gflops = 0

    # 收集结果
    results = {
        'Model': model_name,
        'mAP@0.5': f"{metrics.box.map50:.4f}",
        'mAP@0.5:0.95': f"{metrics.box.map:.4f}",
        'Precision': f"{metrics.box.mp:.4f}",
        'Recall': f"{metrics.box.mr:.4f}",
        'Speed(ms)': f"{avg_speed:.2f}",
        'FPS': f"{fps:.1f}",
        'Params(M)': f"{params:.2f}",
        'GFLOPs': f"{gflops:.2f}" if gflops > 0 else "N/A"
    }

    print(f"\n✅ {model_name} 评估完成")
    print(f"   mAP@0.5: {results['mAP@0.5']}")
    print(f"   FPS: {results['FPS']}")

    return results

def main():
    """主函数"""
    print("\n" + "="*60)
    print("🎯 YOLO Benchmark 评估系统")
    print("="*60)

    config = load_config()

    all_results = []

    # 评估所有 YOLO/RT-DETR 模型
    for model_name, model_config in config['models'].items():
        # 跳过 Faster R-CNN (单独处理)
        if 'framework' in model_config and model_config['framework'] == 'detectron2':
            continue

        weight_path = find_best_weights(
            model_name=model_name,
            model_config=model_config,
            common_config=config['train']
        )

        result = evaluate_model(
            model_name=model_name,
            weight_path=weight_path,
            dataset_config=config['dataset']
        )

        if result:
            all_results.append(result)

    # 尝试加载 Faster R-CNN 结果
    faster_rcnn_result = load_faster_rcnn_results()
    if faster_rcnn_result:
        print("\n✅ 找到 Faster R-CNN 评估结果")
        all_results.append(faster_rcnn_result)
    else:
        print("\n⚠️  未找到 Faster R-CNN 结果 (运行 train_faster_rcnn.py 和 eval_faster_rcnn.py 来训练和评估)")

    # 保存和显示结果
    if all_results:
        df = pd.DataFrame(all_results)

        # 保存CSV
        output_dir = Path(__file__).parent.parent / 'results'
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / 'benchmark_comparison.csv'
        df.to_csv(csv_path, index=False)

        # 显示结果
        print("\n" + "="*80)
        print("📊 Benchmark结果对比")
        print("="*80)
        print(df.to_string(index=False))
        print("\n" + "="*80)
        print(f"✅ 结果已保存到: {csv_path}")
        print("="*80)

        # 打印性能排名
        print("\n🏆 性能排名:")
        df_sorted = df.copy()
        df_sorted['mAP@0.5'] = df_sorted['mAP@0.5'].astype(float)
        df_sorted['FPS'] = df_sorted['FPS'].astype(float)

        print("\n   按mAP@0.5排序:")
        for idx, row in df_sorted.sort_values('mAP@0.5', ascending=False).iterrows():
            print(f"   {row['Model']:15s} - {row['mAP@0.5']}")

        print("\n   按FPS排序:")
        for idx, row in df_sorted.sort_values('FPS', ascending=False).iterrows():
            print(f"   {row['Model']:15s} - {row['FPS']} FPS")
    else:
        print("\n❌ 没有找到可评估的模型")

if __name__ == '__main__':
    main()
