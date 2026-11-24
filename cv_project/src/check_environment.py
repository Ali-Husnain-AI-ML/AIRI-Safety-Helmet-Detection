import sys
import torch
import cv2
import ultralytics

print("=" * 45)
print("AIRI PITB - COMPUTER VISION PROJECT")
print("=" * 45)

print("Python Version:", sys.version)
print("YOLO Version:", ultralytics.__version__)
print("OpenCV Version:", cv2.__version__)
print("PyTorch Version:", torch.__version__)

print("CUDA Available:", torch.cuda.is_available())

print("\nEnvironment Setup Successful!")