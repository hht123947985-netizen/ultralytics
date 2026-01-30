"""
批量训练所有benchmark模型（优化版）
- 跳过已有训练结果
- 统一使用runs/detect/benchmark目录
"""
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch

# 解决显存不足的关键配置
torch.backends.cudnn.benchmark = False  # 禁用自动寻找最佳算法，防止显存峰值溢出
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True  # 允许 TF32，能加速且省显存

def load_config(config_path='../configs/benchmark_config.yaml'):
    """加载配置文件"""
    config_path = Path(__file__).parent / config_path
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_model(model_name, model_config, config, dataset_config):
    """训练单个模型"""
    print(f"\n{'='*60}")
    print(f"模型: {model_name}")
    print(f"{'='*60}")

    # 检查是否已有训练结果
    if 'existing_results' in model_config:
        existing_path = Path(model_config['existing_results'])
        if existing_path.exists():
            print(f"✅ 发现已有训练结果: {existing_path}")
            print(f"⏭️  跳过训练，将使用现有结果进行评估")
            return {'status': 'existing', 'path': existing_path}

    # 初始化模型
    print(f"📦 加载模型: {model_config['model']}")
    model = YOLO(model_config['model'])

    # 设置训练参数 - 单独读取每个参数
    train_args = {
        'data': dataset_config['path'],
        'name': model_config.get('name', model_name),
        'optimizer': model_config.get('optimizer', 'auto'),
        'lr0': model_config.get('lr0', 0.01),
        'epochs': model_config.get('epochs', config.get('train', {}).get('epochs', 100)),
        'imgsz': model_config.get('imgsz', config.get('train', {}).get('imgsz', 640)),
        'batch': model_config.get('batch', config.get('train', {}).get('batch', 16)),
        'device': model_config.get('device', config.get('train', {}).get('device', '0')),
        'workers': model_config.get('workers', config.get('train', {}).get('workers', 8)),
        'patience': model_config.get('patience', config.get('train', {}).get('patience', 50)),
        'project': model_config.get('project', config.get('train', {}).get('project', 'runs/detect')),
        'save': model_config.get('save', config.get('train', {}).get('save', True)),
        'plots': model_config.get('plots', config.get('train', {}).get('plots', True)),
    }

    # 开始训练
    try:
        print(f"🚀 开始训练...")
        results = model.train(**train_args)
        save_dir = Path(results.save_dir)
        print(f"\n✅ {model_name} 训练完成!")
        print(f"📂 结果保存在: {save_dir}")
        return {'status': 'trained', 'path': save_dir, 'results': results}
    except Exception as e:
        print(f"\n❌ {model_name} 训练失败: {str(e)}")
        return {'status': 'failed', 'error': str(e)}

def main():
    """主函数"""
    # 加载配置
    config = load_config()

    if not config:
        print("❌ 配置文件加载失败")
        return

    # 检查GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"🖥️  使用设备: {device}")
    if device == 'cuda':
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}")

    # 更新设备配置
    if 'train' not in config:
        config['train'] = {}
    config['train']['device'] = device

    # 训练所有模型
    results = {}
    for model_name, model_config in config['models'].items():
        result = train_model(
            model_name=model_name,
            model_config=model_config,
            config=config,
            dataset_config=config['dataset']
        )
        results[model_name] = result

    # 打印汇总
    print("\n" + "="*60)
    print("📊 训练汇总")
    print("="*60)

    for model_name, result in results.items():
        status = result['status']
        if status == 'existing':
            print(f"✅ {model_name:15s} - 使用现有结果")
        elif status == 'trained':
            print(f"🎉 {model_name:15s} - 新训练完成")
        elif status == 'failed':
            print(f"❌ {model_name:15s} - 失败: {result['error']}")

    print("\n下一步: 运行 'python eval_all.py' 进行评估对比")

if __name__ == '__main__':
    main()
