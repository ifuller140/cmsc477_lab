import os
from ultralytics import YOLO, settings
import sys, traceback

# Ensure ultralytics datasets_dir points to the relative 'datasets' folder in this directory
current_dir = os.path.dirname(os.path.abspath(__file__))
settings.update({'datasets_dir': os.path.join(current_dir, 'datasets')})

try:
    print("Loading Yolov8s...", flush=True)
    # Start with YOLOv8 small model
    model = YOLO("yolov8s.pt")
    
    print("Training Lab3 data on GPU...", flush=True)
    model.train(data="lab3_data.yaml", epochs=50, imgsz=640, device=0, name="lab3_gpu_train")
    
    print("Training complete. Running predictions on validation images...", flush=True)
    # Run predictions on the validation images
    model.predict(source="datasets/lab3/images/val", save=True, name="lab3_gpu_val")
    
    print("Validation complete.", flush=True)
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
