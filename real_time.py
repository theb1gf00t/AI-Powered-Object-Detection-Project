from ultralytics import YOLO
import cv2
import time


def realtime_detection(camera_id=0, conf_threshold=0.3):
    model = YOLO('runs/detect/train7/weights/best.pt')
    cap = cv2.VideoCapture(camera_id)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n" + "="*80)
    print("Real-Time Object Detection with YOLO")
    print("="*80)
    print(f"Camera ID: {camera_id}")
    print(f"Confidence Threshold: {conf_threshold}")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("="*80 + "\n")
    
    fps_list = []
    frame_count = 0
    
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        frame_count += 1
        
        results = model(frame, conf=conf_threshold, verbose=False)
        annotated_frame = results[0].plot()
        
        boxes = results[0].boxes
        num_detections = len(boxes)
        
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        fps_list.append(fps)
        avg_fps = sum(fps_list[-30:]) / len(fps_list[-30:])
        
        cv2.putText(annotated_frame, f'FPS: {avg_fps:.1f}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(annotated_frame, f'Detections: {num_detections}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(annotated_frame, f'Frame: {frame_count}', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow('YOLO Real-Time Detection', annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f'screenshot_{frame_count}.jpg'
            cv2.imwrite(filename, annotated_frame)
            print(f"Saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*80)
    print("Session Summary")
    print("="*80)
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {sum(fps_list)/len(fps_list):.2f}")
    print(f"Max FPS: {max(fps_list):.2f}")
    print(f"Min FPS: {min(fps_list):.2f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    realtime_detection(camera_id=0, conf_threshold=0.3)
