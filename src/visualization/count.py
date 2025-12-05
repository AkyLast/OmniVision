import yaml
import os
import json
from pathlib import Path
from collections import defaultdict

with open("config.json", "r", encoding="utf-8") as file:
    CONFIG = json.load(file)

DATASET = f"{CONFIG['model_config']['dataset_path']}/data.yaml"

def analyze_labels(label_path, num_classes):
    label_path = Path(label_path)
    if not label_path.exists():
        return {"images": 0, "instances": [0]*num_classes}

    counts = [0] * num_classes
    image_count = 0
    
    for file in label_path.glob("*.txt"):
        image_count += 1

        with open(file, "r") as f:
            for line in f.readlines():
                cls_id = int(line.strip().split()[0])
                if 0 <= cls_id < num_classes:
                    counts[cls_id] += 1

    return {
        "images": image_count,
        "instances": counts
    }


def analyze_yolo_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    class_names = data.get("names", [])
    total_classes = len(class_names)

    # Caminhos das labels
    train_labels = str(Path(data.get("train")).parent / "labels")
    val_labels   = str(Path(data.get("val")).parent / "labels")
    test_labels  = str(Path(data.get("test")).parent / "labels")

    train = analyze_labels(train_labels, total_classes)
    val   = analyze_labels(val_labels, total_classes)
    test  = analyze_labels(test_labels, total_classes)

    return {
        "total_classes": total_classes,
        "classes": class_names,
        "train": train,
        "val": val,
        "test": test,
    }


result = analyze_yolo_yaml(DATASET)

print("\n===== RESULTADO DO DATASET =====")
print(f"Total de classes: {result['total_classes']}")
print(f"Classes: {result['classes']}")

print("\n--- Train ---")
print(f"Imagens: {result['train']['images']}")
print(f"Instâncias por classe: {result['train']['instances']}")

print("\n--- Validação ---")
print(f"Imagens: {result['val']['images']}")
print(f"Instâncias por classe: {result['val']['instances']}")

print("\n--- Teste ---")
print(f"Imagens: {result['test']['images']}")
print(f"Instâncias por classe: {result['test']['instances']}")
