from ultralytics import YOLO

import torch

# 解决 CUDNN_STATUS_EXECUTION_FAILED 的关键配置
torch.backends.cudnn.benchmark = False  # 禁用自动寻找最佳算法，防止显存峰值溢出
torch.backends.cudnn.deterministic = False 
torch.backends.cuda.matmul.allow_tf32 = True # 允许 TF32，能加速且省显存

def train_detect():
    # Upgrade to Medium model for better feature extraction on 4090
    model = YOLO("yolo11s.pt") 
    
    results = model.train(
        data="datasets/Defect detection of wind turbine blades/data.yaml",
        device=0,
        batch=16,
        workers=0,
        epochs=200,
        patience=50,
        save=True,
        project="wind-turbine-blade",
        name="detect_v3",
        plots=True,
        
        # High resolution to capture small cracks/defects
        imgsz=960,          
        
        # Optimization settings
        lr0=0.01,
        lrf=0.01,
        cos_lr=True,       # Use cosine learning rate scheduler
        
        # Augmentation for complex surface defects
        mosaic=1.0,
        mixup=0.2,          
        copy_paste=0.2,     
        
        # Loss gain adjustments to improve Recall
        box=7.5,
        cls=1.5,           # Increased class gain to reduce false negatives
        dfl=1.5,
    )

    # Evaluation
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")

if __name__ == '__main__':
    train_detect()
