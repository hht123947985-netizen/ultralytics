"""Convert YOLO detection dataset to classification dataset format."""

from pathlib import Path

import cv2


def convert_detection_to_classification(
    dataset_path: str,
    output_path: str = None,
    class_names: list = None,
    min_crop_size: int = 32,
    padding_ratio: float = 0.1,
):
    """
    Convert detection dataset to classification dataset.

    Args:
        dataset_path: Source dataset path
        output_path: Output path, defaults to dataset_path/classification
        class_names: List of class names
        min_crop_size: Minimum crop size, objects smaller than this will be skipped
        padding_ratio: Padding ratio for cropping
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path) if output_path else dataset_path / "classification"

    if class_names is None:
        class_names = read_class_names(dataset_path)
    print(f"Classes: {class_names}")

    splits = {"train": "train", "valid": "val", "test": "test"}
    stats = {split: {name: 0 for name in class_names} for split in splits.values()}

    for src_split, dst_split in splits.items():
        src_images_dir = dataset_path / src_split / "images"
        src_labels_dir = dataset_path / src_split / "labels"

        if not src_images_dir.exists():
            print(f"Skipping {src_split}: directory not found")
            continue

        print(f"\nProcessing {src_split} dataset...")

        for class_name in class_names:
            (output_path / dst_split / class_name).mkdir(parents=True, exist_ok=True)

        image_files = list(src_images_dir.glob("*.[jJ][pP][gG]")) + list(
            src_images_dir.glob("*.[pP][nN][gG]")
        )

        for img_path in image_files:
            label_path = src_labels_dir / (img_path.stem + ".txt")

            if not label_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  Warning: Cannot read image {img_path}")
                continue

            img_h, img_w = img.shape[:2]

            with open(label_path, "r") as f:
                lines = f.readlines()

            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                if class_id >= len(class_names):
                    print(f"  Warning: Invalid class ID {class_id}")
                    continue

                x_center = float(parts[1]) * img_w
                y_center = float(parts[2]) * img_h
                bbox_w = float(parts[3]) * img_w
                bbox_h = float(parts[4]) * img_h

                pad_w = bbox_w * padding_ratio
                pad_h = bbox_h * padding_ratio

                x1 = int(max(0, x_center - bbox_w / 2 - pad_w))
                y1 = int(max(0, y_center - bbox_h / 2 - pad_h))
                x2 = int(min(img_w, x_center + bbox_w / 2 + pad_w))
                y2 = int(min(img_h, y_center + bbox_h / 2 + pad_h))

                crop_w = x2 - x1
                crop_h = y2 - y1
                if crop_w < min_crop_size or crop_h < min_crop_size:
                    continue

                crop_img = img[y1:y2, x1:x2]

                class_name = class_names[class_id]
                output_filename = f"{img_path.stem}_{idx}.jpg"
                output_file = output_path / dst_split / class_name / output_filename

                cv2.imwrite(str(output_file), crop_img)
                stats[dst_split][class_name] += 1

    print("\n" + "=" * 50)
    print("Conversion complete! Statistics:")
    print("=" * 50)

    for split, class_stats in stats.items():
        total = sum(class_stats.values())
        if total > 0:
            print(f"\n{split}:")
            for class_name, count in class_stats.items():
                print(f"  {class_name}: {count}")
            print(f"  Total: {total}")

    print(f"\nOutput directory: {output_path}")

    create_classification_yaml(output_path, class_names)

    return output_path


def read_class_names(dataset_path: Path) -> list:
    """Read class names from data.yaml or classes.txt."""
    import re

    yaml_path = dataset_path / "data.yaml"

    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            content = f.read()

        match = re.search(r"names:\s*\[(.*?)\]", content)
        if match:
            names_str = match.group(1)
            names = [n.strip().strip("'\"") for n in names_str.split(",")]
            return names

    classes_path = dataset_path / "classes.txt"
    if classes_path.exists():
        with open(classes_path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
            names = [n for n in names if n.lower() != "null"]
            return names

    raise ValueError("Cannot find class names, please specify class_names parameter")


def create_classification_yaml(output_path: Path, class_names: list):
    """Create classification dataset config file."""
    yaml_content = f"""path: {output_path.absolute()}
train: train
val: val
test: test

nc: {len(class_names)}

names:
"""
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"

    yaml_path = output_path / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"Config file created: {yaml_path}")


if __name__ == "__main__":
    dataset_path = Path(__file__).parent.parent / "datasets" / "Defect detection of wind turbine blades"
    class_names = ["crack", "damage", "dirt", "peeled_paint"]

    convert_detection_to_classification(
        dataset_path=dataset_path,
        output_path=dataset_path / "classification",
        class_names=class_names,
        min_crop_size=32,
        padding_ratio=0.1,
    )
