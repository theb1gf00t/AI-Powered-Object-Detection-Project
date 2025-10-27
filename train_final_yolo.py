from ultralytics import YOLO
import torch
import time
from pathlib import Path

def clear_all_cache():
    cache_patterns = [
        '/mnt/34B471F7B471BBC4/CSO_project/datasets/*.cache',
        '/mnt/34B471F7B471BBC4/CSO_project/datasets/images/*.cache',
        '/mnt/34B471F7B471BBC4/CSO_project/datasets/labels/*.cache',
    ]
    for pattern in cache_patterns:
        for cache_file in Path('/').glob(pattern[1:]):
            try:
                cache_file.unlink()
            except:
                pass

def train_final_yolo():
    clear_all_cache()
    model = YOLO('yolov8s.pt')
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    start_time = time.time()
    results = model.train(
        data='coco_yolo_exact.yaml',
        epochs=30,
        imgsz=416,
        batch=16,
        workers=8,
        device=0,
        lr0=0.01,
        save=True,
        amp=True,
        optimizer='SGD',
        val=True,
        patience=10,
        verbose=True
    )
    end_time = time.time()
    training_duration = (end_time - start_time) / 60

if __name__ == "__main__":
    exec(open('final_fix_yolo_structure.py').read())
    train_final_yolo()
