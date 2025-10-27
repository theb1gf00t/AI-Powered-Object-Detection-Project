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

st.set_page_config(page_title="YOLO Detection Dashboard", layout="wide")

@st.cache_resource
def load_model():
    return YOLO('runs/detect/train7/weights/best.pt')

model = load_model()

# System Monitor Class
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

tabs = st.tabs(["📷 Detection", "📊 Model Performance", "⚡ CPU vs GPU", "📈 Class Distribution", "🚀 Benchmark"])

# TAB 1: Image Detection
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

# TAB 2: Model Performance
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

# TAB 3: CPU vs GPU Performance
# TAB 3: CPU vs GPU Performance (FIXED VERSION)
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
            
            # IMPORTANT: Load fresh model for GPU
            import torch
            torch.cuda.empty_cache()
            status_text.text("Loading GPU model...")
            gpu_model = YOLO('runs/detect/train7/weights/best.pt')
            gpu_model.to('cuda')
            
            # GPU Warmup (more iterations)
            status_text.text("Warming up GPU...")
            for _ in range(3):
                for img_path in all_images[:10]:
                    _ = gpu_model(str(img_path), device='cuda', verbose=False)
            
            # GPU Test
            status_text.text("Testing on GPU...")
            gpu_times = []
            for i, img_path in enumerate(all_images):
                start = time.time()
                results = gpu_model(str(img_path), device='cuda', verbose=False)
                gpu_times.append(time.time() - start)
                progress_bar.progress((i + 1) / (2 * len(all_images)))
            
            # Clean up GPU model
            del gpu_model
            torch.cuda.empty_cache()
            
            # Load fresh model for CPU
            status_text.text("Loading CPU model...")
            cpu_model = YOLO('runs/detect/train7/weights/best.pt')
            cpu_model.to('cpu')
            
            # CPU Warmup
            status_text.text("Warming up CPU...")
            for _ in range(2):
                for img_path in all_images[:10]:
                    _ = cpu_model(str(img_path), device='cpu', verbose=False)
            
            # CPU Test
            status_text.text("Testing on CPU...")
            cpu_times = []
            for i, img_path in enumerate(all_images):
                start = time.time()
                results = cpu_model(str(img_path), device='cpu', verbose=False)
                cpu_times.append(time.time() - start)
                progress_bar.progress((len(all_images) + i + 1) / (2 * len(all_images)))
            
            # Clean up CPU model
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
            
            # Fixed speedup calculation
            if gpu_avg < cpu_avg:
                speedup = cpu_avg / gpu_avg
                st.success(f"🚀 GPU is **{speedup:.2f}x faster** than CPU!")
            else:
                slowdown = gpu_avg / cpu_avg
                st.warning(f"⚠️ GPU is **{slowdown:.2f}x slower** than CPU")
                st.info(f"💡 GPU: {gpu_avg*1000:.1f}ms | CPU: {cpu_avg*1000:.1f}ms | Try increasing image count for better GPU utilization")
            
            # Chart
            fig = go.Figure()
            fig.add_trace(go.Box(y=gpu_times, name='GPU', marker_color='#4ECDC4'))
            fig.add_trace(go.Box(y=cpu_times, name='CPU', marker_color='#FF6B6B'))
            fig.update_layout(
                title="Inference Time Distribution",
                yaxis_title="Time (seconds)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional debugging info
            st.markdown("### Performance Details")
            st.markdown(f"- **GPU Min**: {min(gpu_times)*1000:.1f}ms | **Max**: {max(gpu_times)*1000:.1f}ms | **Std**: {np.std(gpu_times)*1000:.1f}ms")
            st.markdown(f"- **CPU Min**: {min(cpu_times)*1000:.1f}ms | **Max**: {max(cpu_times)*1000:.1f}ms | **Std**: {np.std(cpu_times)*1000:.1f}ms")
        else:
            st.error("No test images found!")


# TAB 4: Class Distribution Analysis
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

# TAB 5: 4K Benchmark with System Monitoring
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
