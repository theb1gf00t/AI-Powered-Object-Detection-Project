# AI-Powered-Object-Detection-Project

A Python-based object detection system built with YOLO and COCO dataset.  
This project enables training, evaluation and real-time inference of object detection models.

## Table of Contents
- [About](#about)  
- [Features](#features)  
- [Repository Structure](#repository-structure)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
  - [Usage](#usage)  
    - [Dataset setup](#dataset-setup)  
    - [Training the model](#training-the-model)  
    - [Real-time detection](#real-time-detection)  
- [Configuration](#configuration)  
- [Tips & Notes](#tips-and-notes)  
- [Contributing](#contributing)  
- [License](#license)  

## About  
This project uses the YOLO (You Only Look Once) architecture for object detection and the COCO (“Common Objects in Context”) dataset configuration. It supports:  
- Preparing/customizing a dataset for object detection.  
- Training YOLO models.  
- Running real-time detection via webcam or video feed.  
- Configurable via YAML for model/data settings.

## Features  
- Dataset preparation script: `dataset_setup.py`  
- Training script: `train_final_yolo.py`  
- Real-time inference script: `real_time.py`  
- Flask or web-based front end: `app.py` (for demo or deployment)  
- Configuration file: `coco_yolo_exact.yaml` (detailing dataset/model settings)  

## Repository Structure  
```
├── app.py                    # Web/app interface for inference  
├── coco_yolo_exact.yaml      # Configuration for dataset/model  
├── dataset_setup.py          # Script to prepare or convert dataset  
├── real_time.py              # Script for realtime object detection  
├── train_final_yolo.py       # Script to train the YOLO model  
└── README.md                 # This documentation  
```

## Getting Started  

### Prerequisites  
- Python 3.7+ (or your preferred 3.x version)  
- Git  
- GPU recommended (for training)  
- Install dependencies (see next step)  

### Installation  
```bash
git clone https://github.com/theb1gf00t/AI-Powered-Object-Detection-Project.git  
cd AI-Powered-Object-Detection-Project  
pip install -r requirements.txt   # (if you create one)  
```

### Usage  

#### Dataset setup  
Edit `coco_yolo_exact.yaml` to point to your dataset(s). For conversion/preparation run:  
```bash
python dataset_setup.py --config coco_yolo_exact.yaml  
```
This will prepare the training/validation splits and annotations.

#### Training the model  
Once dataset is prepared, start training using:  
```bash
python train_final_yolo.py --config coco_yolo_exact.yaml  
```
You can monitor loss, metrics and save checkpoints.

#### Real-time detection  
After model is trained (or use a pre-trained model), run the real-time script:  
```bash
python real_time.py --weights path/to/model_weights.pt --config coco_yolo_exact.yaml  
```
This will start detection via webcam or video feed and display bounding-boxes, object labels, and confidence scores.

You may also start the web/app interface via:  
```bash
python app.py  
```
Then open your browser at `http://localhost:5000` (or appropriate port) to use the model from a UI.

## Configuration  
All key settings (dataset paths, classes, model architecture, hyperparameters) are in `coco_yolo_exact.yaml`. Adjust:  
- `train` / `val` dataset paths  
- `classes` (object categories)  
- `model` (YOLO version, layers/anchors)  
- `lr`, `epochs`, `batch_size`  
- `device` (cpu / cuda)  

## Tips & Notes  
- For best performance, use a modern GPU (e.g., NVIDIA RTX series) and ensure CUDA/cudnn compatibility.  
- You can fine-tune from a pre-trained YOLO model for faster convergence.  
- Real-time performance depends on input resolution and GPU speed; consider resizing frames.  
- Ensure your dataset annotations match YOLO format (class, x_center, y_center, width, height).  
- If adding new classes, update both your dataset and YAML config accordingly.

## Contributing  
Contributions are welcome! Feel free to:  
- Submit issues (bugs/feature requests)  
- Submit pull requests (improvements, fixes, documentation)  
- Fork the repository and follow good commit practice  
Before major changes, please open an issue so we can discuss the proposed feature.

## License  
Specify your license here (e.g., MIT License) or “All rights reserved”.
