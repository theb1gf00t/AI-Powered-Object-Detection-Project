import streamlit as st
from ultralytics import YOLO
from PIL import Image
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import time
import numpy as np
from collections import Counter, deque
import psutil
import subprocess
import threading
from collections import defaultdict
import tempfile
import os
import cv2  # ✅ ADD THIS
import torch  # ✅ ADD THIS

st.set_page_config(page_title="YOLO Detection Dashboard", layout="wide")

@st.cache_resource
def load_model():
    return YOLO('runs/detect/train7/weights/best.pt')

model = load_model()

class DistanceEstimator:
    def __init__(self, focal_length=1200, real_height=1.5):
        self.focal_length = focal_length
        self.real_height = real_height
    
    def estimate_distance(self, bbox):
        bbox_height = bbox[3] - bbox[1]
        if bbox_height > 0:
            distance = (self.real_height * self.focal_length) / bbox_height
            return distance
        return None


class CollisionDetector:
    def __init__(self, fps=30):
        self.fps = fps
        self.track_history = defaultdict(list)
        self.CRITICAL_TTC = 1.5
        self.WARNING_TTC = 3.0
        self.CAUTION_TTC = 5.0
        
    def update_track(self, track_id, distance, timestamp):
        self.track_history[track_id].append({
            'distance': distance,
            'time': timestamp
        })
        if len(self.track_history[track_id]) > 10:
            self.track_history[track_id].pop(0)
    
    def calculate_speed(self, track_id):
        history = self.track_history[track_id]
        if len(history) < 2:
            return 0
        prev = history[-2]
        curr = history[-1]
        dist_change = prev['distance'] - curr['distance']
        time_elapsed = curr['time'] - prev['time']
        if time_elapsed > 0:
            return dist_change / time_elapsed
        return 0
    
    def calculate_ttc(self, distance, speed):
        if speed <= 0:
            return float('inf')
        return distance / speed
    
    def assess_danger(self, ttc):
        if ttc < self.CRITICAL_TTC:
            return "CRITICAL", (0, 0, 255)
        elif ttc < self.WARNING_TTC:
            return "WARNING", (0, 165, 255)
        elif ttc < self.CAUTION_TTC:
            return "CAUTION", (0, 255, 255)
        else:
            return "SAFE", (0, 255, 0)


def process_collision_video(video_path, model_path, focal_length=1200, progress_callback=None):
    """
    Comprehensive collision detection with all safety factors:
    1. Distance estimation (from bounding box)
    2. Relative speed calculation
    3. Time-to-collision (TTC)
    4. Low speed filter (too slow = no danger)
    5. Safe distance threshold (far away = no danger)
    6. Lane detection (other lane = no danger)
    7. Ego speed factor (your speed affects safe distance)
    """
    
    model = YOLO(model_path)
    distance_est = DistanceEstimator(focal_length=focal_length)
    collision_det = CollisionDetector()
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # ========== SAFETY PARAMETERS ==========
    
    # Lane boundaries (your lane = center 40%)
    YOUR_LANE_LEFT = int(width * 0.30)
    YOUR_LANE_RIGHT = int(width * 0.70)
    
    # Speed thresholds
    MIN_DANGER_SPEED = 3.0      # m/s (~11 km/h) - below this, no danger
    SLOW_SPEED_THRESHOLD = 5.0  # m/s (~18 km/h) - slow approach
    
    # Distance thresholds
    MIN_DETECTION_DISTANCE = 2.0   # m - too close, likely error
    MAX_DETECTION_DISTANCE = 100.0 # m - too far to care
    
    # Ego speed estimation (from average scene motion)
    ego_speed_history = deque(maxlen=30)  # Track your speed
    ASSUMED_EGO_SPEED = 50.0 / 3.6  # Default: 50 km/h = 13.9 m/s
    
    # ========== OUTPUT SETUP ==========
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    stats = {'warnings': 0, 'critical': 0, 'total_detections': 0, 'safe': 0}
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = frame_count / fps
        frame_count += 1
        
        if progress_callback and frame_count % 10 == 0:
            progress_callback(frame_count / total_frames)
        
        # Detect and track vehicles
        results = model.track(frame, persist=True, classes=[2, 3, 5, 7], 
                            verbose=False, conf=0.3)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            
            # Estimate ego speed from average vehicle speeds (simplified)
            frame_speeds = []
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                track_id_int = int(track_id)
                
                # ========== FACTOR 6: LANE DETECTION ==========
                box_center_x = (x1 + x2) / 2
                box_width = x2 - x1
                
                # Check if vehicle center is in your lane
                in_your_lane = YOUR_LANE_LEFT < box_center_x < YOUR_LANE_RIGHT
                
                # Also check if majority of vehicle is in your lane
                vehicle_overlap = (
                    min(x2, YOUR_LANE_RIGHT) - max(x1, YOUR_LANE_LEFT)
                ) / box_width
                
                in_your_lane = in_your_lane and vehicle_overlap > 0.5
                
                if not in_your_lane:
                    # Different lane - no danger, draw gray box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                (120, 120, 120), 1)
                    continue
                
                # ========== FACTOR 1: DISTANCE ESTIMATION ==========
                distance = distance_est.estimate_distance(box)
                
                # ========== FACTOR 5: DISTANCE RANGE CHECK ==========
                if not distance or distance < MIN_DETECTION_DISTANCE or distance > MAX_DETECTION_DISTANCE:
                    # Distance out of valid range
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                (0, 255, 0), 1)
                    continue
                
                # Track this vehicle
                collision_det.update_track(track_id_int, distance, current_time)
                
                # ========== FACTOR 2: RELATIVE SPEED ==========
                relative_speed = collision_det.calculate_speed(track_id_int)
                
                # Collect speeds for ego speed estimation
                if abs(relative_speed) > 0.5:
                    frame_speeds.append(abs(relative_speed))
                
                # ========== FACTOR 3: TIME TO COLLISION ==========
                ttc = collision_det.calculate_ttc(distance, abs(relative_speed))
                
                # ========== FACTOR 7: EGO SPEED FACTOR ==========
                # Estimate your speed from average vehicle motion
                if frame_speeds:
                    current_ego_speed = np.median(frame_speeds)
                    ego_speed_history.append(current_ego_speed)
                    ego_speed = np.mean(list(ego_speed_history))
                else:
                    ego_speed = ASSUMED_EGO_SPEED
                
                # Calculate safe distance based on your speed
                # Rule: 1 second per 10 km/h (simplified 2-second rule)
                ego_speed_kmh = ego_speed * 3.6
                safe_distance = max(10.0, ego_speed_kmh / 3.6 * 2.0)  # 2-second rule
                
                # Adjust TTC thresholds based on ego speed
                if ego_speed_kmh > 80:  # Highway speeds
                    critical_ttc = 2.0
                    warning_ttc = 4.0
                elif ego_speed_kmh > 50:  # City fast
                    critical_ttc = 1.5
                    warning_ttc = 3.0
                else:  # City slow
                    critical_ttc = 1.0
                    warning_ttc = 2.0
                
                # ========== FACTOR 4: LOW SPEED FILTER ==========
                # If approaching too slowly, always safe
                if relative_speed < MIN_DANGER_SPEED:
                    status_text = "SAFE"
                    text_color = (0, 255, 0)
                    box_color = (0, 255, 0)
                    stats['safe'] += 1
                
                # ========== COMBINED DANGER ASSESSMENT ==========
                elif relative_speed >= MIN_DANGER_SPEED:
                    # Vehicle approaching fast enough to matter
                    
                    # Check if distance is too small regardless of TTC
                    if distance < safe_distance * 0.5:
                        # Very close - critical
                        status_text = "DANGER"
                        text_color = (0, 0, 255)
                        box_color = (0, 0, 255)
                        stats['critical'] += 1
                    
                    elif ttc < critical_ttc:
                        # Critical TTC
                        status_text = "DANGER"
                        text_color = (0, 0, 255)
                        box_color = (0, 0, 255)
                        stats['critical'] += 1
                    
                    elif ttc < warning_ttc or distance < safe_distance:
                        # Warning TTC or below safe distance
                        status_text = "WARNING"
                        text_color = (0, 165, 255)
                        box_color = (0, 165, 255)
                        stats['warnings'] += 1
                    
                    elif relative_speed < SLOW_SPEED_THRESHOLD:
                        # Approaching slowly
                        status_text = "CAUTION"
                        text_color = (0, 255, 255)
                        box_color = (0, 255, 255)
                        stats['warnings'] += 1
                    
                    else:
                        # Safe
                        status_text = "SAFE"
                        text_color = (0, 255, 0)
                        box_color = (0, 255, 0)
                        stats['safe'] += 1
                
                else:
                    # Not approaching
                    status_text = "SAFE"
                    text_color = (0, 255, 0)
                    box_color = (0, 255, 0)
                    stats['safe'] += 1
                
                stats['total_detections'] += 1
                
                # ========== VISUALIZATION ==========
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                            box_color, 3)
                
                # Status label
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                
                (text_width, text_height), _ = cv2.getTextSize(
                    status_text, font, font_scale, thickness
                )
                
                y_pos = int(y1) - 15
                
                # Black background
                cv2.rectangle(
                    frame,
                    (int(x1), y_pos - text_height - 10),
                    (int(x1) + text_width + 10, y_pos + 5),
                    (0, 0, 0),
                    -1
                )
                
                # Status text
                cv2.putText(
                    frame, status_text, (int(x1) + 5, y_pos),
                    font, font_scale, text_color, thickness
                )
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    return output_path, stats


class SystemMonitor:
    def __init__(self):
        self.cpu_readings = deque()
        self.ram_readings = deque()
        self.gpu_readings = deque()
        self.monitoring = False

    def get_gpu_usage(self):
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_data = result.stdout.strip().split(', ')
                if len(gpu_data) >= 3:
                    return {
                        'gpu_utilization': float(gpu_data[0]),
                        'memory_used': float(gpu_data[1]),
                        'memory_total': float(gpu_data[2]),
                        'memory_percent': (float(gpu_data[1]) / float(gpu_data[2])) * 100,
                        'temperature': float(gpu_data[3]) if len(gpu_data) >= 4 else 0
                    }
        except:
            pass
        return None

    def get_system_usage(self):
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'ram_percent': psutil.virtual_memory().percent,
            'ram_used_gb': psutil.virtual_memory().used / 1e9,
            'ram_total_gb': psutil.virtual_memory().total / 1e9
        }

    def monitor_continuously(self, duration=300):
        self.cpu_readings.clear()
        self.ram_readings.clear()
        self.gpu_readings.clear()
        self.monitoring = True
        start_time = time.time()
        while self.monitoring and (time.time() - start_time) < duration:
            system_usage = self.get_system_usage()
            self.cpu_readings.append(system_usage['cpu_percent'])
            self.ram_readings.append(system_usage['ram_percent'])
            gpu_usage = self.get_gpu_usage()
            if gpu_usage:
                self.gpu_readings.append(gpu_usage['gpu_utilization'])
            time.sleep(1)

    def get_average_usage(self):
        cpu_avg = np.mean(list(self.cpu_readings)) if self.cpu_readings else 0
        ram_avg = np.mean(list(self.ram_readings)) if self.ram_readings else 0
        gpu_avg = np.mean(list(self.gpu_readings)) if self.gpu_readings else 0
        return {
            'cpu_avg': cpu_avg,
            'ram_avg': ram_avg,
            'gpu_avg': gpu_avg,
            'cpu_max': max(self.cpu_readings) if self.cpu_readings else 0,
            'ram_max': max(self.ram_readings) if self.ram_readings else 0,
            'gpu_max': max(self.gpu_readings) if self.gpu_readings else 0,
            'samples': len(self.cpu_readings)
        }

st.title("🔍 YOLO Object Detection Dashboard")
st.markdown("**Fine-tuned YOLOv8s** on 30 COCO classes")

tabs = st.tabs(["📷 Detection", "📊 Model Performance", "⚡ CPU vs GPU", "📈 Class Distribution", "🚀 Benchmark", "Road Safety"])

with tabs[0]:
    st.header("Image Upload & Detection")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Upload images (single or multiple)", 
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True
        )
        
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.markdown(f"### Processing: {uploaded_file.name}")
            
            image = Image.open(uploaded_file)
            
            col_orig, col_pred = st.columns(2)
            
            with col_orig:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            results = model(image, conf=conf_threshold)
            annotated_img = results[0].plot()
            boxes = results[0].boxes
            
            with col_pred:
                st.subheader("Detected Objects")
                st.image(annotated_img, use_container_width=True)
            
            st.success(f"✅ Found {len(boxes)} objects")
            
            if len(boxes) > 0:
                st.markdown("**Detections:**")
                detection_data = []
                for i, box in enumerate(boxes, 1):
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    detection_data.append(f"{i}. **{model.names[class_id]}** - {confidence:.1%}")
                
                st.markdown("<br>".join(detection_data), unsafe_allow_html=True)
            
            st.markdown("---")

with tabs[1]:
    st.header("Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("mAP50", "57.5%", "+184%")
    with col2:
        st.metric("mAP50-95", "40.0%", "+171%")
    with col3:
        st.metric("Precision", "64.4%")
    with col4:
        st.metric("Recall", "53.3%")
    
    st.markdown("---")
    
    st.subheader("Baseline vs Fine-tuned Comparison")
    
    metrics = ['mAP50', 'mAP50-95', 'Precision', 'Recall']
    baseline = [0.2025, 0.1479, 0.45, 0.38]
    finetuned = [0.5751, 0.4002, 0.6442, 0.5331]
    
    fig = go.Figure(data=[
        go.Bar(name='Pretrained YOLOv8s', x=metrics, y=baseline, marker_color='#FF6B6B'),
        go.Bar(name='Fine-tuned YOLOv8s', x=metrics, y=finetuned, marker_color='#4ECDC4')
    ])
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Metric",
        yaxis_title="Score",
        barmode='group',
        height=400,
        yaxis_range=[0, 0.7]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Improvement Breakdown")
    
    improvements = {
        'mAP50': '+184.0%',
        'mAP50-95': '+170.5%',
        'Precision': '+43.1%',
        'Recall': '+40.3%'
    }
    
    for metric, improvement in improvements.items():
        st.markdown(f"**{metric}:** {improvement}")
    
    st.markdown("---")
    
    st.subheader("Supported Classes (30 total)")
    
    classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'backpack', 'umbrella', 'handbag', 'tie',
        'skis', 'snowboard', 'sports ball', 'kite',
        'banana', 'apple', 'sandwich'
    ]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**People & Transport:**")
        for c in classes[:8]:
            st.markdown(f"• {c}")
    
    with col2:
        st.markdown("**Street & Animals:**")
        for c in classes[8:19]:
            st.markdown(f"• {c}")
    
    with col3:
        st.markdown("**Accessories & Sports:**")
        for c in classes[19:]: 
            st.markdown(f"• {c}")

with tabs[2]:
    st.header("CPU vs GPU Performance Comparison")
    
    st.info("Run performance test on sample images")
    
    num_images = st.slider("Number of test images", 10, 1000, 500, 50)
    
    if st.button("🚀 Run Performance Test"):
        test_images_path = Path('/mnt/34B471F7B471BBC4/CSO_project/datasets/test_dataset/test2017')
        all_images = list(test_images_path.glob('*.jpg'))[:num_images]
        
        if all_images:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            import torch
            torch.cuda.empty_cache()
            status_text.text("Loading GPU model...")
            gpu_model = YOLO('runs/detect/train7/weights/best.pt')
            gpu_model.to('cuda')
            
            status_text.text("Warming up GPU...")
            for _ in range(3):
                for img_path in all_images[:10]:
                    _ = gpu_model(str(img_path), device='cuda', verbose=False)
        
            status_text.text("Testing on GPU...")
            gpu_times = []
            for i, img_path in enumerate(all_images):
                start = time.time()
                results = gpu_model(str(img_path), device='cuda', verbose=False)
                gpu_times.append(time.time() - start)
                progress_bar.progress((i + 1) / (2 * len(all_images)))
            
            del gpu_model
            torch.cuda.empty_cache()
            
            status_text.text("Loading CPU model...")
            cpu_model = YOLO('runs/detect/train7/weights/best.pt')
            cpu_model.to('cpu')
            
            status_text.text("Warming up CPU...")
            for _ in range(2):
                for img_path in all_images[:10]:
                    _ = cpu_model(str(img_path), device='cpu', verbose=False)
            
            status_text.text("Testing on CPU...")
            cpu_times = []
            for i, img_path in enumerate(all_images):
                start = time.time()
                results = cpu_model(str(img_path), device='cpu', verbose=False)
                cpu_times.append(time.time() - start)
                progress_bar.progress((len(all_images) + i + 1) / (2 * len(all_images)))
            
            del cpu_model
            torch.cuda.empty_cache()
            
            progress_bar.empty()
            status_text.empty()
            
            gpu_avg = np.mean(gpu_times)
            cpu_avg = np.mean(cpu_times)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("GPU Avg Time", f"{gpu_avg:.3f}s")
                st.metric("GPU FPS", f"{1/gpu_avg:.1f}")
            
            with col2:
                st.metric("CPU Avg Time", f"{cpu_avg:.3f}s")
                st.metric("CPU FPS", f"{1/cpu_avg:.1f}")
            
            if gpu_avg < cpu_avg:
                speedup = cpu_avg / gpu_avg
                st.success(f"🚀 GPU is **{speedup:.2f}x faster** than CPU!")
            else:
                slowdown = gpu_avg / cpu_avg
                st.warning(f"⚠️ GPU is **{slowdown:.2f}x slower** than CPU")
                st.info(f"💡 GPU: {gpu_avg*1000:.1f}ms | CPU: {cpu_avg*1000:.1f}ms | Try increasing image count for better GPU utilization")
            
            fig = go.Figure()
            fig.add_trace(go.Box(y=gpu_times, name='GPU', marker_color='#4ECDC4'))
            fig.add_trace(go.Box(y=cpu_times, name='CPU', marker_color='#FF6B6B'))
            fig.update_layout(
                title="Inference Time Distribution",
                yaxis_title="Time (seconds)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Performance Details")
            st.markdown(f"- **GPU Min**: {min(gpu_times)*1000:.1f}ms | **Max**: {max(gpu_times)*1000:.1f}ms | **Std**: {np.std(gpu_times)*1000:.1f}ms")
            st.markdown(f"- **CPU Min**: {min(cpu_times)*1000:.1f}ms | **Max**: {max(cpu_times)*1000:.1f}ms | **Std**: {np.std(cpu_times)*1000:.1f}ms")
        else:
            st.error("No test images found!")

with tabs[3]:
    st.header("Class Detection Distribution")
    
    st.info("Analyze detection patterns across test dataset")
    
    analysis_images = st.slider("Number of images to analyze", 50, 1000, 500, 50)
    
    if st.button("📊 Analyze Test Dataset"):
        test_images_path = Path('/mnt/34B471F7B471BBC4/CSO_project/datasets/test_dataset/test2017')
        all_images = list(test_images_path.glob('*.jpg'))[:analysis_images]
        
        if all_images:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            class_counter = Counter()
            confidence_scores = {}
            
            for i, img_path in enumerate(all_images):
                status_text.text(f"Processing {i+1}/{len(all_images)} images...")
                results = model(str(img_path), conf=0.3, verbose=False)
                boxes = results[0].boxes
                
                for box in boxes:
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    class_name = model.names[class_id]
                    class_counter[class_name] += 1
                    
                    if class_name not in confidence_scores:
                        confidence_scores[class_name] = []
                    confidence_scores[class_name].append(confidence)
                
                progress_bar.progress((i + 1) / len(all_images))
            
            progress_bar.empty()
            status_text.empty()
            
            st.subheader("Detection Statistics")
            
            total_detections = sum(class_counter.values())
            st.metric("Total Detections", total_detections)
            st.metric("Total Images Processed", len(all_images))
            
            sorted_classes = sorted(class_counter.items(), key=lambda x: x[1], reverse=True)[:10]
            classes_names = [x[0] for x in sorted_classes]
            classes_counts = [x[1] for x in sorted_classes]
            
            fig = go.Figure(data=[
                go.Bar(x=classes_names, y=classes_counts, marker_color='#4ECDC4')
            ])
            fig.update_layout(
                title="Top 10 Most Detected Classes",
                xaxis_title="Class",
                yaxis_title="Count",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Average Confidence by Class")
            
            avg_confidence = {k: np.mean(v) for k, v in confidence_scores.items() if v}
            sorted_conf = sorted(avg_confidence.items(), key=lambda x: x[1], reverse=True)[:10]
            
            conf_names = [x[0] for x in sorted_conf]
            conf_values = [x[1] for x in sorted_conf]
            
            fig2 = go.Figure(data=[
                go.Bar(x=conf_names, y=conf_values, marker_color='#FF6B6B')
            ])
            fig2.update_layout(
                title="Top 10 Classes by Confidence",
                xaxis_title="Class",
                yaxis_title="Average Confidence",
                height=400,
                yaxis_range=[0, 1]
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            st.subheader("Detailed Statistics")
            
            df_data = []
            for class_name, count in sorted_classes:
                avg_conf = np.mean(confidence_scores[class_name]) if class_name in confidence_scores else 0
                percentage = (count / total_detections) * 100
                df_data.append({
                    'Class': class_name,
                    'Detections': count,
                    'Avg Confidence': f"{avg_conf:.1%}",
                    'Percentage': f"{percentage:.1f}%"
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.error("No test images found!")

with tabs[4]:
    st.header("🚀 Performance Benchmark")
    
    st.info("Comprehensive GPU performance test with real-time system monitoring")
    
    benchmark_images = st.slider("Number of images for benchmark", 1000, 5000, 4000, 500)
    
    if st.button("🏃 Run Benchmark"):
        test_images_path = Path('/mnt/34B471F7B471BBC4/CSO_project/datasets/test_dataset/test2017')
        all_images = list(test_images_path.glob('*.jpg'))[:benchmark_images]
        
        if all_images:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            monitor = SystemMonitor()
            
            status_text.text("Warming up GPU...")
            for img_path in all_images[:10]:
                _ = model(str(img_path), device='cuda', verbose=False)
            
            times = []
            estimated_duration = len(all_images) * 0.1
            monitor_thread = threading.Thread(target=monitor.monitor_continuously, args=(estimated_duration * 2,))
            monitor_thread.start()
            
            start_total = time.time()
            
            for i, img_path in enumerate(all_images):
                start_time = time.time()
                results = model(str(img_path), device='cuda', verbose=False)
                end_time = time.time()
                inference_time = (end_time - start_time) * 1000
                times.append(inference_time)
                
                if (i + 1) % 100 == 0 or (i + 1) == len(all_images):
                    elapsed = time.time() - start_total
                    images_remaining = len(all_images) - (i + 1)
                    avg_time_per_image = elapsed / (i + 1)
                    eta_seconds = images_remaining * avg_time_per_image
                    eta_minutes = eta_seconds / 60
                    avg_inference_ms = np.mean(times)
                    current_fps = 1000 / avg_inference_ms if avg_inference_ms > 0 else 0
                    
                    status_text.text(f"Progress: {i+1}/{len(all_images)} | Avg: {avg_inference_ms:.1f}ms | FPS: {current_fps:.1f} | ETA: {eta_minutes:.1f}min")
                    progress_bar.progress((i + 1) / len(all_images))
            
            total_time = time.time() - start_total
            monitor.monitoring = False
            monitor_thread.join()
            system_stats = monitor.get_average_usage()
            
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"✅ Benchmark Complete! Processed {len(all_images)} images in {total_time/60:.2f} minutes")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Time", f"{np.mean(times):.2f}ms")
                st.metric("Min Time", f"{np.min(times):.2f}ms")
            
            with col2:
                st.metric("FPS", f"{len(all_images)/total_time:.2f}")
                st.metric("Max Time", f"{np.max(times):.2f}ms")
            
            with col3:
                st.metric("Total Time", f"{total_time/60:.2f} min")
                st.metric("Std Dev", f"{np.std(times):.2f}ms")
            
            st.markdown("---")
            
            st.subheader("System Resource Usage")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Avg CPU Usage", f"{system_stats['cpu_avg']:.1f}%")
                st.metric("Max CPU Usage", f"{system_stats['cpu_max']:.1f}%")
            
            with col2:
                st.metric("Avg RAM Usage", f"{system_stats['ram_avg']:.1f}%")
                st.metric("Max RAM Usage", f"{system_stats['ram_max']:.1f}%")
            
            with col3:
                st.metric("Avg GPU Usage", f"{system_stats['gpu_avg']:.1f}%")
                st.metric("Max GPU Usage", f"{system_stats['gpu_max']:.1f}%")
            
            st.markdown("---")
            
            fig1 = go.Figure()
            fig1.add_trace(go.Histogram(x=times, nbinsx=50, marker_color='#4ECDC4'))
            fig1.update_layout(
                title="Inference Time Distribution",
                xaxis_title="Time (ms)",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=['CPU', 'RAM', 'GPU'],
                y=[system_stats['cpu_avg'], system_stats['ram_avg'], system_stats['gpu_avg']],
                marker_color=['#FF6B6B', '#4ECDC4', '#FFD166']
            ))
            fig2.update_layout(
                title="Average System Resource Usage",
                yaxis_title="Usage (%)",
                height=400,
                yaxis_range=[0, 100]
            )
            st.plotly_chart(fig2, use_container_width=True)
            
        else:
            st.error("No test images found!")

with tabs[5]:  # Collision Detection Tab
    st.header("🚗 Road Safety System")
    st.markdown("""
    Upload a dashcam video to detect vehicles and assess collision risks.
    The system will:
    - Detect and track vehicles (cars, trucks, buses, motorcycles)
    - Estimate distance to each vehicle
    - Calculate approaching speed
    - Compute Time-to-Collision (TTC)
    - Provide color-coded warnings:
      - 🟢 **Green**: Safe
      - 🟡 **Yellow**: Caution
      - 🟠 **Orange**: Warning (TTC < 3s)
      - 🔴 **Red**: Critical (TTC < 1.5s)
    """)
    
    st.divider()
    
    # File uploader
    uploaded_video = st.file_uploader(
        "Upload Dashcam Video",
        type=['mp4', 'mov', 'avi'],
        help="Upload a video from your dashcam or driving footage"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        focal_length = st.slider(
            "Camera Focal Length (pixels)",
            min_value=200,
            max_value=2400,
            value=1200,
            step=50,
            help="Adjust based on your camera calibration"
        )
    
    with col2:
        process_button = st.button(
            "🚀 Process Video",
            type="primary",
            disabled=uploaded_video is None
        )
    
    if uploaded_video and process_button:
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.read())
            temp_video_path = tmp_file.name
        
        st.info("🔄 Processing video... This may take a few minutes.")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress):
            progress_bar.progress(progress)
            status_text.text(f"Processing: {int(progress*100)}%")
        
        try:
            # Process video
            output_path, stats = process_collision_video(
                temp_video_path,
                'runs/detect/train7/weights/best.pt',
                progress_callback=update_progress
            )
            
            progress_bar.progress(1.0)
            status_text.text("Processing complete! ✅")
            
            st.success("✅ Video processed successfully!")
            
            # Display statistics
            st.subheader("📊 Detection Statistics")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("Total Detections", stats['total_detections'])
            with metric_col2:
                st.metric("Warnings Issued", stats['warnings'], 
                         delta="Caution/Warning")
            with metric_col3:
                st.metric("Critical Alerts", stats['critical'], 
                         delta="Danger!", delta_color="inverse")
            
            st.divider()
            
            # Display output video
            st.subheader("🎥 Processed Video")
            
            with open(output_path, 'rb') as video_file:
                video_bytes = video_file.read()
                st.video(video_bytes)
            
            # Download button
            st.download_button(
                label="💾 Download Processed Video",
                data=video_bytes,
                file_name="collision_detection_output.mp4",
                mime="video/mp4"
            )
            
            # Cleanup
            os.unlink(temp_video_path)
            os.unlink(output_path)
            
        except Exception as e:
            st.error(f"❌ Error processing video: {str(e)}")
            os.unlink(temp_video_path)
    
    elif not uploaded_video:
        st.info("👆 Upload a dashcam video to get started")
        
        # Show example/demo
        st.subheader("📖 How it Works")
        st.markdown("""
        **Distance Estimation:**
        - Uses bounding box height and camera focal length
        - Formula: `Distance = (Real Height × Focal Length) / Pixel Height`
        
        **Speed Calculation:**
        - Tracks distance change over time
        - Positive speed = vehicle approaching
        
        **Time-to-Collision (TTC):**
        - Formula: `TTC = Distance / Speed`
        - Warns when TTC is dangerously low
        
        **Color Coding:**
        - Safe (Green): TTC > 5s or not approaching
        - Caution (Yellow): 3s < TTC < 5s
        - Warning (Orange): 1.5s < TTC < 3s
        - Critical (Red): TTC < 1.5s
        """)
