import os
import cv2
import torch
from tqdm import tqdm
from glob import glob
from ultralytics import YOLO

# ===============================
# 1. Cek apakah GPU tersedia
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("GPU tidak tersedia, menggunakan CPU.")

# ===============================
# 3. Training YOLOv8
# ===============================
model = YOLO("D:\PythonProject\Project MobileNetV3\Dataset Fix Resize\yolov8nMobilenetv3.yaml")  # Load model YOLOv8 Nano

# Jalankan training
model.train(
    data="D:\PythonProject\Project MobileNetV3\Dataset Fix Resize\data.yaml",  # Path ke dataset yang sudah di-resize
    epochs=50,       # Jumlah epoch
    imgsz=320,       # Ukuran gambar setelah resize
    batch=16,        # Batch size
    optimizer="AdamW",  # Optimizer yang digunakan
    lr0=0.0075,      # Learning rate
    device="cuda"    # Pakai GPU jika tersedia
)

print("Training selesai!")