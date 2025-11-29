#!/usr/bin/env python3
"""CUDA環境チェックスクリプト"""
import torch

print("=" * 50)
print("CUDA Environment Check")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"Total memory: {props.total_memory / (1024**3):.2f} GB")
    print(f"Compute capability: {props.major}.{props.minor}")
else:
    print("CUDA is not available")

# Tritonチェック
try:
    import triton
    print(f"Triton available: True")
    print(f"Triton version: {triton.__version__}")
except ImportError:
    print("Triton available: False")

print("=" * 50)
