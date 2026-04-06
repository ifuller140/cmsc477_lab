import os
from ultralytics import YOLO, settings
import sys, traceback

# Ensure ultralytics datasets_dir points to the relative 'datasets' folder in this directory
current_dir = os.path.dirname(os.path.abspath(__file__))
settings.update({'datasets_dir': os.path.join(current_dir, 'datasets')})

try:
    print("Loading Yolov8s...", flush=True)
    model = YOLO("yolov8s.pt")
    
    print("Training Lego data on GPU...", flush=True)
    model.train(data="lego_data.yaml", epochs=50, imgsz=640, device=0, name="lego_gpu_train")
    
    print("Training Robomaster data on GPU...", flush=True)
    model = YOLO("yolov8s.pt") # reload base model for second dataset
    model.train(data="robomaster_data.yaml", epochs=50, imgsz=640, device=0, name="robomaster_gpu_train")
    
    print("Training complete.", flush=True)
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
