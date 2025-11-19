"""
LNS (Logarithmic Number System) Kernel Demo

このデモでは、LNSカーネルの基本的な使用方法と、
標準matmulとの比較を示します。

物理的直観:
LNSカーネルは、乗算器(FMA)を加算器(ADD)に変換することで、
推論時の計算コストと消費電力を削減します。

Requirements: 4.4, 12.2
"""

import torch
import time

try:
    from src.kernels.lns_kernel import lns_matmul, TRITON_AVAILABLE
    from src.models.phase1 import LNSLinear, convert_linear_to_lns
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure src/ is in PYTHONPATH")
    exit(1)


def demo_basic_usage():
    """
    Demo 1: Basic LNS kernel usage
    
    物理的直観:
    対数領域での行列積を計算。入力は対数値、出力も対数値です。
    """
    print("\n" + "="*80)
    print("Demo 1: Basic LNS Kernel Usage")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
        return
    
    if not TRITON_AVAILABLE:
        print("Triton not available. Skipping demo.")
        return
    
    # Create test matrices (positive values for log domain)
    M, K, N = 256, 256, 256
    a_linear = torch.abs(torch.randn(M, K, device='cuda', dtype=torch.float32)) + 0.1
    b_linear = torch.abs(torch.randn(K, N, device='cuda', dtype=torch.float32)) + 0.1
    
    print(f"Matrix sizes: A={a_linear.shape}, B={b_linear.shape}")
    
    # Convert to log domain
    log_a = torch.log(a_linear)
    log_b = torch.log(b_linear)
    
    print(f"Log domain: log(A)={log_a.shape}, log(B)={log_b.shape}")
    
    # LNS matrix multiplication
    log_c = lns_matmul(log_a, log_b)
    
    print(f"Result in log domain: log(C)={log_c.shape}")
    
    # Convert back to linear domain
    c_lns = torch.exp(log_c)
    
    print(f"Result in linear domain: C={c_lns.shape}")
    
    # Compare with standard matmul
    c_true = torch.matmul(a_linear, b_linear)
    
    # Compute error
    abs_error = torch.abs(c_true - c_lns).mean().item()
    rel_error = (torch.abs(c_true - c_lns) / (torch.abs(c_true) + 1e-8)).mean().item()
    
    print(f"\nAccuracy:")
    print(f"  Mean absolute error: {abs_error:.6f}")
    print(f"  Mean relative error: {rel_error*100:.2f}%")
    print(f"  Max absolute error: {torch.abs(c_true - c_lns).max().item():.6f}")


def demo_lns_linear_layer():
    """
    Demo 2: LNSLinear layer usage
    
    物理的直観:
    nn.Linearの代替として使用。学習時は通常の計算、
    推論時はLNSカーネルを使用します。
    """
    print("\n" + "="*80)
    print("Demo 2: LNSLinear Layer")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
        return
    
    if not TRITON_AVAILABLE:
        print("Triton not available. Skipping demo.")
        return
    
    # Create LNSLinear layer
    layer = LNSLinear(512, 256, bias=True, use_lns=True).cuda()
    
    print(f"Layer: {layer}")
    print(f"Parameters: {sum(p.numel() for p in layer.parameters())} total")
    
    # Training mode: uses standard matmul
    layer.train()
    x_train = torch.randn(32, 512, device='cuda')
    
    print("\nTraining mode:")
    y_train = layer(x_train)
    print(f"  Input: {x_train.shape}")
    print(f"  Output: {y_train.shape}")
    print(f"  Log weights computed: {layer.log_weight is not None}")
    
    # Inference mode: uses LNS kernel
    layer.eval()
    x_infer = torch.randn(32, 512, device='cuda')
    
    print("\nInference mode:")
    y_infer = layer(x_infer)
    print(f"  Input: {x_infer.shape}")
    print(f"  Output: {y_infer.shape}")
    print(f"  Log weights computed: {layer.log_weight is not None}")
    
    # Pre-compute log weights for faster inference
    layer.prepare_lns_weights()
    print(f"  Log weights pre-computed: {layer.log_weight is not None}")


def demo_model_conversion():
    """
    Demo 3: Convert existing model to use LNS
    
    物理的直観:
    既存のモデルの線形層をLNS版に置き換え。
    学習済みの重みをそのまま使用できます。
    """
    print("\n" + "="*80)
    print("Demo 3: Model Conversion")
    print("="*80)
    
    # Create a simple model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(512, 256)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(256, 128)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    model = SimpleModel()
    
    print("Original model:")
    print(f"  fc1: {type(model.fc1).__name__}")
    print(f"  fc2: {type(model.fc2).__name__}")
    
    # Convert to LNS
    model_lns = convert_linear_to_lns(model, inplace=False)
    
    print("\nConverted model:")
    print(f"  fc1: {type(model_lns.fc1).__name__}")
    print(f"  fc2: {type(model_lns.fc2).__name__}")
    
    # Test forward pass
    x = torch.randn(16, 512)
    
    with torch.no_grad():
        y_original = model(x)
        y_lns = model_lns(x)
    
    print(f"\nForward pass:")
    print(f"  Input: {x.shape}")
    print(f"  Original output: {y_original.shape}")
    print(f"  LNS output: {y_lns.shape}")
    
    # Compare outputs (should be similar in training mode)
    error = torch.abs(y_original - y_lns).mean().item()
    print(f"  Mean difference: {error:.6f}")


def demo_performance_comparison():
    """
    Demo 4: Performance comparison
    
    物理的直観:
    LNSカーネルと標準matmulの速度を比較。
    大きな行列ほどLNSの優位性が顕著になります。
    """
    print("\n" + "="*80)
    print("Demo 4: Performance Comparison")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
        return
    
    if not TRITON_AVAILABLE:
        print("Triton not available. Skipping demo.")
        return
    
    sizes = [(128, 128, 128), (256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]
    num_warmup = 10
    num_iterations = 100
    
    print(f"{'Size':<20} {'torch.matmul':<15} {'LNS Kernel':<15} {'Speedup':<10}")
    print("-"*70)
    
    for M, K, N in sizes:
        # Create test matrices
        a = torch.abs(torch.randn(M, K, device='cuda', dtype=torch.float16)) + 0.1
        b = torch.abs(torch.randn(K, N, device='cuda', dtype=torch.float16)) + 0.1
        log_a = torch.log(a)
        log_b = torch.log(b)
        
        # Warmup
        for _ in range(num_warmup):
            _ = torch.matmul(a, b)
            _ = lns_matmul(log_a, log_b)
        torch.cuda.synchronize()
        
        # Benchmark torch.matmul
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(num_iterations):
            _ = torch.matmul(a, b)
        end.record()
        torch.cuda.synchronize()
        matmul_time = start.elapsed_time(end) / num_iterations
        
        # Benchmark LNS kernel
        start.record()
        for _ in range(num_iterations):
            _ = lns_matmul(log_a, log_b)
        end.record()
        torch.cuda.synchronize()
        lns_time = start.elapsed_time(end) / num_iterations
        
        speedup = matmul_time / lns_time
        
        print(f"{M}x{K}x{N:<13} {matmul_time:>12.3f}ms {lns_time:>12.3f}ms {speedup:>8.2f}x")
    
    print("-"*70)


def demo_accuracy_analysis():
    """
    Demo 5: Accuracy analysis
    
    物理的直観:
    Max-Log近似による精度低下を分析。
    スパースな活性化では精度が高く、密な活性化では低下します。
    """
    print("\n" + "="*80)
    print("Demo 5: Accuracy Analysis")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
        return
    
    if not TRITON_AVAILABLE:
        print("Triton not available. Skipping demo.")
        return
    
    M, K, N = 256, 256, 256
    
    # Test different sparsity levels
    sparsity_levels = [0.0, 0.5, 0.7, 0.9, 0.95]
    
    print(f"{'Sparsity':<15} {'Mean Rel Error':<20} {'Max Rel Error':<20}")
    print("-"*60)
    
    for sparsity in sparsity_levels:
        # Create sparse matrices
        a = torch.abs(torch.randn(M, K, device='cuda')) + 0.1
        b = torch.abs(torch.randn(K, N, device='cuda')) + 0.1
        
        # Apply sparsity
        mask_a = torch.rand_like(a) > sparsity
        mask_b = torch.rand_like(b) > sparsity
        a = a * mask_a
        b = b * mask_b
        
        # Standard matmul
        c_true = torch.matmul(a, b)
        
        # LNS matmul
        log_a = torch.log(a + 1e-8)
        log_b = torch.log(b + 1e-8)
        log_c = lns_matmul(log_a, log_b)
        c_lns = torch.exp(log_c)
        
        # Compute errors
        rel_error = torch.abs(c_true - c_lns) / (torch.abs(c_true) + 1e-8)
        mean_rel_error = rel_error.mean().item()
        max_rel_error = rel_error.max().item()
        
        print(f"{sparsity*100:>6.1f}%{'':<8} {mean_rel_error*100:>17.2f}% {max_rel_error*100:>17.2f}%")
    
    print("-"*60)
    print("\nNote: Higher sparsity → Lower error (Max-Log approximation works better)")


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("LNS (Logarithmic Number System) Kernel Demo")
    print("="*80)
    print("\n物理的直観:")
    print("LNSカーネルは、乗算器(FMA)を加算器(ADD)に変換することで、")
    print("推論時の計算コストと消費電力を削減します。")
    print("\nMax-Log近似により精度は低下しますが、スパースな活性化では")
    print("精度低下が小さく、実用的な範囲に収まります。")
    
    # Check availability
    print(f"\nCUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Triton Available: {TRITON_AVAILABLE}")
    
    # Run demos
    demo_basic_usage()
    demo_lns_linear_layer()
    demo_model_conversion()
    demo_performance_comparison()
    demo_accuracy_analysis()
    
    print("\n" + "="*80)
    print("Demo Complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Run benchmarks: python scripts/benchmark_lns_kernel.py")
    print("2. Read documentation: docs/implementation/LNS_KERNEL.md")
    print("3. Run tests: pytest tests/test_lns_kernel.py -v")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
