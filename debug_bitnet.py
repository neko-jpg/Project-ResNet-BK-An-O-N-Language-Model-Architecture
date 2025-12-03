
import torch
import triton
import sys

try:
    from src.kernels.bitnet_triton import bitnet_matmul
except ImportError:
    print("Could not import bitnet_matmul")
    sys.exit(1)

def test_kernel():
    print("Initializing tensors...")
    device = 'cuda'
    # Case 2: Second matmul in SemiseparableMatrix
    # x: (32, 16), w: (16, 1024)
    B, N, K = 32, 16, 1024
    
    # Inputs
    x = torch.randn(B, N, device=device, dtype=torch.float32)
    w_int8 = torch.randint(-1, 2, (N, K), device=device, dtype=torch.int8)
    w_scale = torch.ones(K, device=device, dtype=torch.float32)
    
    print(f"x: {x.dtype}, w: {w_int8.dtype}, scale: {w_scale.dtype}")
    
    print("Running bitnet_matmul loop...")
    try:
        for i in range(100):
            y = bitnet_matmul(x, w_int8, w_scale)
        torch.cuda.synchronize()
        print("Kernel execution successful (100 iterations).")
        print(f"Output shape: {y.shape}, dtype: {y.dtype}")
    except Exception as e:
        print(f"Kernel execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(0)
    test_kernel()
