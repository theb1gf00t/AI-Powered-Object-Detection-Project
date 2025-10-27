# AI-Powered Object Detection Project

A full-featured pipeline for training, evaluating, and deploying YOLOv8-based object detection models on a filtered COCO dataset (30 classes). Includes dataset setup, model training, an interactive Streamlit dashboard, and real-time inference.

---

## Project Structure and Files

| File                    | Purpose                                                                                                     |
|-------------------------|-------------------------------------------------------------------------------------------------------------|
| `dataset_setup.py`      | Prepares dataset: filters/copies images, creates YOLO structure, generates `coco_yolo_exact.yaml`.          |
| `coco_yolo_exact.yaml`  | Dataset config for YOLO: paths, classes, labels (auto-generated). Needed for all training/inference.        |
| `train_final_yolo.py`   | Trains YOLOv8 model using above config. Handles cache, launches training, saves trained weights.            |
| `app.py`                | Streamlit dashboard: image upload/detection, benchmarking (CPU vs GPU), class stats, and visualization.     |
| `real_time.py`          | Runs real-time object detection with webcam and trained YOLO weights; shows live annotated results.         |

---

## Workflow

1. **Prepare Dataset**
    - Clean, filter, and structure data for YOLO. Auto-generates needed YAML config.
    - Run:
      ```
      python3 dataset_setup.py
      ```

2. **Train the Model**
    - Launches training using YOLOv8 and the prepared dataset/config.
    - Run:
      ```
      python3 train_final_yolo.py
      ```

3. **Interactive Dashboard**
    - Upload images for detection, view benchmarking and class analyses through Streamlit.
    - Run:
      ```
      streamlit run app.py
      ```

4. **Real-Time Detection**
    - Detect objects live from webcam using trained model.
    - Run:
      ```
      python3 real_time.py
      ```

---



## Notes

- Ensure all dataset paths in scripts and YAML match your local directory structure.
- The YOLOv8 weights after training will be loaded by both the dashboard and real-time scripts.
- Project is modular: each file can be run independently given prerequisites (e.g., trained model, dataset setup).

---

## Requirements

- Python 3.8+
- `ultralytics`, `streamlit`, `opencv-python`, `pandas`, `plotly`, `psutil`, etc.
- Nvidia GPU and CUDA (recommended for training/benchmarking but CPU fallback available).

---

## Credits

Developed as an end-to-end demonstration for modern object detection pipelines using YOLOv8.
