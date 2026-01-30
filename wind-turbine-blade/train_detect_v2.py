from ultralytics import YOLO


def train_detect():
    model = YOLO("yolo11s.pt")
    results = model.train(
        data="datasets/Defect detection of wind turbine blades/data.yaml",
        device=0,
        batch=16,
        workers=8,
        epochs=150,
        patience=20,
        save=True,
        save_period=10,
        project="wind-turbine-blade",
        name="detect_v2",
        plots=True,
        # Optimization parameters
        imgsz=640,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        # Loss weights - increase cls_loss for better classification
        box=7.5,
        cls=1.0,
        dfl=1.5,
    )

    metrics = model.val()
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")


if __name__ == '__main__':
    train_detect()
