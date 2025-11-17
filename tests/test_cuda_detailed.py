import torch
import sys

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version (compiled):", torch.version.cuda)

if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))
    print("CUDA capability:", torch.cuda.get_device_capability(0))
else:
    print("\nCUDA not available. Checking possible issues...")
    print("PyTorch build:", torch.__version__)
    
    # Check if CUDA libraries are accessible
    try:
        import torch.cuda
        print("torch.cuda module loaded successfully")
    except Exception as e:
        print(f"Error loading torch.cuda: {e}")
