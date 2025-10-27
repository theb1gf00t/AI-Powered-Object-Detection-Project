import shutil
from pathlib import Path


def create_filtered_dataset():
    print("\n" + "="*60)
    print("STEP 1: Creating Filtered Dataset")
    print("="*60)
    
    base_path = Path('/mnt/34B471F7B471BBC4/CSO_project/datasets')
    filtered_train_path = base_path / 'filtered_train2017'
    filtered_val_path = base_path / 'filtered_val2017'
    
    if filtered_train_path.exists():
        shutil.rmtree(filtered_train_path)
    if filtered_val_path.exists():
        shutil.rmtree(filtered_val_path)
    
    filtered_train_path.mkdir(exist_ok=True)
    filtered_val_path.mkdir(exist_ok=True)
    (filtered_train_path / 'labels').mkdir(exist_ok=True)
    (filtered_val_path / 'labels').mkdir(exist_ok=True)
    
    original_train_labels = base_path / 'labels' / 'train2017'
    original_train_images = base_path / 'train2017'
    train_count = 0
    
    print("Filtering training set...")
    for label_file in original_train_labels.glob('*.txt'):
        image_file = original_train_images / f"{label_file.stem}.jpg"
        if image_file.exists():
            shutil.copy2(image_file, filtered_train_path / image_file.name)
            shutil.copy2(label_file, filtered_train_path / 'labels' / label_file.name)
            train_count += 1
    
    original_val_labels = base_path / 'labels' / 'val2017'
    original_val_images = base_path / 'validation' / 'val2017'
    val_count = 0
    
    print("Filtering validation set...")
    for label_file in original_val_labels.glob('*.txt'):
        image_file = original_val_images / f"{label_file.stem}.jpg"
        if image_file.exists():
            shutil.copy2(image_file, filtered_val_path / image_file.name)
            shutil.copy2(label_file, filtered_val_path / 'labels' / label_file.name)
            val_count += 1
    
    print(f"Training images with labels: {train_count}")
    print(f"Validation images with labels: {val_count}")
    
    return train_count, val_count


def create_yolo_structure():
    print("\n" + "="*60)
    print("STEP 2: Creating YOLO Directory Structure")
    print("="*60)
    
    base_path = Path('/mnt/34B471F7B471BBC4/CSO_project/datasets')
    
    yolo_train = base_path / 'images' / 'train'
    yolo_val = base_path / 'images' / 'val'
    yolo_train_labels = base_path / 'labels' / 'train'
    yolo_val_labels = base_path / 'labels' / 'val'
    
    for dir_path in [yolo_train, yolo_val, yolo_train_labels, yolo_val_labels]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("Organizing training data...")
    filtered_train = base_path / 'filtered_train2017'
    train_count = 0
    for image_file in filtered_train.glob('*.jpg'):
        shutil.copy2(image_file, yolo_train / image_file.name)
        label_file = filtered_train / 'labels' / f"{image_file.stem}.txt"
        if label_file.exists():
            shutil.copy2(label_file, yolo_train_labels / f"{image_file.stem}.txt")
            train_count += 1
    
    print("Organizing validation data...")
    filtered_val = base_path / 'filtered_val2017'
    val_count = 0
    for image_file in filtered_val.glob('*.jpg'):
        shutil.copy2(image_file, yolo_val / image_file.name)
        label_file = filtered_val / 'labels' / f"{image_file.stem}.txt"
        if label_file.exists():
            shutil.copy2(label_file, yolo_val_labels / f"{image_file.stem}.txt")
            val_count += 1
    
    print(f"YOLO training images: {train_count}")
    print(f"YOLO validation images: {val_count}")
    
    return train_count, val_count


def create_yaml_config(train_count, val_count):
    print("\n" + "="*60)
    print("STEP 3: Creating YAML Configuration")
    print("="*60)
    
    yaml_content = f"""path: /mnt/34B471F7B471BBC4/CSO_project/datasets
train: images/train
val: images/val

nc: 30

names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'backpack', 'umbrella', 'handbag', 'tie',
        'skis', 'snowboard', 'sports ball', 'kite',
        'banana', 'apple', 'sandwich']
"""
    
    with open('coco_yolo_exact.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("Created: coco_yolo_exact.yaml")


def verify_structure():
    print("\n" + "="*60)
    print("STEP 4: Verifying Dataset Structure")
    print("="*60)
    
    base_path = Path('/mnt/34B471F7B471BBC4/CSO_project/datasets')
    
    required_dirs = [
        'images/train',
        'images/val',
        'labels/train',
        'labels/val'
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = base_path / dir_name
        exists = dir_path.exists()
        count = len(list(dir_path.glob('*'))) if exists else 0
        print(f"{dir_name}: {count} files - {'OK' if exists else 'MISSING'}")
        all_exist = all_exist and exists
    
    yaml_exists = Path('coco_yolo_exact.yaml').exists()
    print(f"coco_yolo_exact.yaml - {'OK' if yaml_exists else 'MISSING'}")
    
    return all_exist and yaml_exists


def setup_complete_dataset():
    print("\n" + "="*60)
    print("YOLO DATASET SETUP")
    print("="*60)
    
    train_count, val_count = create_filtered_dataset()
    train_count, val_count = create_yolo_structure()
    create_yaml_config(train_count, val_count)
    success = verify_structure()
    
    print("\n" + "="*60)
    if success:
        print("DATASET SETUP COMPLETE")
        print("="*60)
        print(f"Training images: {train_count}")
        print(f"Validation images: {val_count}")
        print("Config file: coco_yolo_exact.yaml")
        print("\nReady for training! Run: python3 train_final_yolo.py")
    else:
        print("SETUP FAILED - Check errors above")
    print("="*60)


if __name__ == "__main__":
    setup_complete_dataset()
