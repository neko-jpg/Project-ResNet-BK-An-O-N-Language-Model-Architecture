"""
Benchmark: Python for loop vs torch._foreach_mul_

Compares gradient scaling performance between:
1. OLD: Python for loop with per-parameter p.grad.mul_(scale)
2. NEW: torch._foreach_mul_(grads, scale) - single fused GPU op
"""

import torch
import torch.nn as nn
import time

def benchmark_scaling():
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available - running CPU benchmark (less meaningful)")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Create a mock model similar to 10B params structure
    # We can't create 10B params, but we'll use many small tensors
    print("\nCreating mock model (simulating many parameters)...")
    
    class MockModel(nn.Module):
        def __init__(self, num_layers=100, hidden_size=4096):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size, bias=True)
                for _ in range(num_layers)
            ])
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    model = MockModel(num_layers=100, hidden_size=4096).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Mock model: {num_params:,} parameters ({num_params/1e9:.2f}B)")
    
    # Create fake gradients
    print("Creating fake gradients...")
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    
    scale = 2.5
    num_iterations = 100
    
    # Warmup
    print("\nWarming up GPU...")
    for _ in range(10):
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(1.0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # =========================================
    # Benchmark OLD: Python for loop
    # =========================================
    print(f"\nBenchmarking OLD method (Python for loop) x{num_iterations}...")
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_old = time.perf_counter()
    
    for _ in range(num_iterations):
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_old = time.perf_counter()
    
    time_old = (end_old - start_old) / num_iterations * 1000  # ms
    
    # =========================================
    # Benchmark NEW: torch._foreach_mul_
    # =========================================
    print(f"Benchmarking NEW method (torch._foreach_mul_) x{num_iterations}...")
    
    # Pre-collect gradients (done once)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_new = time.perf_counter()
    
    for _ in range(num_iterations):
        torch._foreach_mul_(grads, scale)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_new = time.perf_counter()
    
    time_new = (end_new - start_new) / num_iterations * 1000  # ms
    
    # =========================================
    # Results
    # =========================================
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Parameters: {num_params:,}")
    print(f"Iterations: {num_iterations}")
    print("-"*60)
    print(f"OLD (Python for loop):      {time_old:8.3f} ms/call")
    print(f"NEW (torch._foreach_mul_):  {time_new:8.3f} ms/call")
    print("-"*60)
    speedup = time_old / time_new
    print(f"SPEEDUP: {speedup:.2f}x faster")
    print(f"TIME SAVED: {time_old - time_new:.3f} ms/call")
    print("="*60)
    
    return speedup


if __name__ == "__main__":
    benchmark_scaling()
