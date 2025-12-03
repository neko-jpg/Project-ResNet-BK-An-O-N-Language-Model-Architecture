
import torch
import sys
import platform

print(f"Python: {sys.version}")
print(f"Platform: {platform.system()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

try:
    import triton
    print(f"Triton: {triton.__version__}")
except ImportError as e:
    print(f"Triton Import Failed: {e}")

try:
    from src.kernels.bitnet_triton import bitnet_matmul
    print("Successfully imported bitnet_matmul")
except ImportError as e:
    print(f"Failed to import bitnet_matmul: {e}")
except Exception as e:
    print(f"Error importing bitnet_matmul: {e}")
