from ultralytics import YOLO

def train_detect():

    model = YOLO("yolo11s.pt")
    results = model.train(
        data="datasets/Defect detection of wind turbine blades/data.yaml",
        device=0,
        batch=16,
        workers=8,
        epochs=100,
        patience=10,
        save=True,
        save_period=10,
        project="wind-turbine-blade",
        name="detect",
        plots=True,
    )

if __name__ == '__main__':
    train_detect()