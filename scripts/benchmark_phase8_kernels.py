#!/usr/bin/env python3
"""
Phase 8 Kernel Optimization Benchmark

新しいカーネル最適化の効果を測定するベンチマーク:
1. FusedMobiusOperations vs PyTorch fallback
2. GreenFunctionCache hit/miss performance
3. LowRankSSMScan vs sequential SSM
4. FusedScatteringGate vs naive implementation
5. BatchedHyperbolicDistance vs loop implementation
"""

import torch
import time
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


def benchmark_fn(fn, warmup=5, iterations=20, name=""):
    """Benchmark a function and return average time in ms"""
    # Warmup
    for _ in range(warmup):
        fn()
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed = (time.perf_counter() - start) / iterations * 1000  # ms
    print(f"  {name}: {elapsed:.3f} ms")
    return elapsed


def benchmark_mobius_operations():
    """Benchmark Möbius chain operations"""
    print("\n" + "="*60)
    print("1. Möbius Operations Benchmark")
    print("="*60)
    
    batch, seq_len, d_model = 4, 512, 256
    x = torch.randn(batch, seq_len, d_model, device=device)
    a = torch.randn(batch, seq_len, d_model, device=device)
    b = torch.randn(batch, seq_len, d_model, device=device)
    scale = torch.randn(batch, seq_len, d_model, device=device)
    c = 1.0
    
    # PyTorch fallback implementation
    def mobius_add_pytorch(x, a, c=1.0):
        x_norm_sq = (x * x).sum(dim=-1, keepdim=True)
        a_norm_sq = (a * a).sum(dim=-1, keepdim=True)
        x_dot_a = (x * a).sum(dim=-1, keepdim=True)
        
        num = (1 + 2*c*x_dot_a + c*a_norm_sq) * x + (1 - c*x_norm_sq) * a
        denom = 1 + 2*c*x_dot_a + c*c*x_norm_sq*a_norm_sq
        return num / (denom + 1e-7)
    
    def chain_pytorch():
        y1 = mobius_add_pytorch(x, a, c)
        y2 = mobius_add_pytorch(y1, b, c)
        return y2 * scale
    
    pytorch_time = benchmark_fn(chain_pytorch, name="PyTorch Fallback")
    
    # Try Triton optimized version
    try:
        from src.kernels.hyperbolic_mobius_chain import mobius_chain_fused, FusedMobiusOperations
        
        def chain_triton():
            return mobius_chain_fused(x, a, b, scale, c)
        
        triton_time = benchmark_fn(chain_triton, name="Triton Fused")
        speedup = pytorch_time / triton_time
        print(f"  Speedup: {speedup:.2f}x")
    except Exception as e:
        print(f"  Triton not available: {e}")


def benchmark_green_function_cache():
    """Benchmark Green function caching"""
    print("\n" + "="*60)
    print("2. Green Function Cache Benchmark")
    print("="*60)
    
    try:
        from src.kernels.green_function_cache import GreenFunctionCache
        
        cache = GreenFunctionCache(cache_size=256)
        
        batch, seq_len, d_model = 4, 256, 256
        
        # Simulate G_ii computation (expensive)
        def compute_g_ii(x):
            # Simulated expensive computation
            for _ in range(10):
                x = torch.tanh(x @ x.transpose(-1, -2))
            return x.mean(dim=-1)
        
        # Without cache
        inputs = [torch.randn(batch, seq_len, d_model, device=device) for _ in range(20)]
        
        start = time.perf_counter()
        for x in inputs:
            _ = compute_g_ii(x)
        no_cache_time = (time.perf_counter() - start) * 1000
        print(f"  Without Cache: {no_cache_time:.3f} ms (20 calls)")
        
        # With cache (repeated inputs simulate similar inputs)
        cache.clear()
        repeated_inputs = inputs[:5] * 4  # Repeat to get cache hits
        
        start = time.perf_counter()
        for x in repeated_inputs:
            _ = cache.get_or_compute(x, compute_g_ii)
        with_cache_time = (time.perf_counter() - start) * 1000
        
        stats = cache.get_stats()
        print(f"  With Cache: {with_cache_time:.3f} ms (20 calls)")
        print(f"  Cache Hit Rate: {stats['hit_rate']:.1%}")
        print(f"  Speedup (with hits): {no_cache_time / with_cache_time:.2f}x")
        
    except Exception as e:
        print(f"  Error: {e}")


def benchmark_ssm_scan():
    """Benchmark Low-Rank SSM parallel scan"""
    print("\n" + "="*60)
    print("3. Low-Rank SSM Scan Benchmark")
    print("="*60)
    
    batch, seq_len, d_model = 4, 512, 256
    d_state = 64
    x = torch.randn(batch, seq_len, d_model, device=device)
    
    # Sequential SSM (baseline)
    A = torch.randn(d_state, d_state, device=device) * 0.1
    B_proj = torch.randn(d_model, d_state, device=device) * 0.1
    C_proj = torch.randn(d_state, d_model, device=device) * 0.1
    
    def ssm_sequential():
        u = x @ B_proj  # [B, L, d_state]
        h = torch.zeros(batch, d_state, device=device)
        outputs = []
        for t in range(seq_len):
            h = h @ A.T + u[:, t]
            y = h @ C_proj
            outputs.append(y)
        return torch.stack(outputs, dim=1)
    
    seq_time = benchmark_fn(ssm_sequential, name="Sequential SSM")
    
    # Try parallel scan version
    try:
        from src.kernels.low_rank_ssm_scan import LowRankSSMScan
        
        ssm_parallel = LowRankSSMScan(d_model=d_model, d_state=d_state, rank=16).to(device)
        
        def run_parallel():
            return ssm_parallel(x)
        
        parallel_time = benchmark_fn(run_parallel, name="Parallel SSM Scan")
        speedup = seq_time / parallel_time
        print(f"  Speedup: {speedup:.2f}x")
        
    except Exception as e:
        print(f"  Parallel SSM not available: {e}")


def benchmark_scattering_gate():
    """Benchmark Fused Scattering Gate"""
    print("\n" + "="*60)
    print("4. Scattering Gate Benchmark")
    print("="*60)
    
    batch, heads, seq_len = 4, 8, 256
    d_model = 256
    
    G_ii = torch.randn(batch, seq_len, device=device) + 1j * torch.randn(batch, seq_len, device=device)
    attn = torch.softmax(torch.randn(batch, heads, seq_len, seq_len, device=device), dim=-1)
    
    # Naive implementation
    def scattering_naive():
        energy = G_ii.abs()  # [B, L]
        gate = torch.softmax(energy, dim=-1)  # [B, L]
        gate_exp = gate.unsqueeze(1).unsqueeze(-1)  # [B, 1, L, 1]
        gated = attn * gate_exp
        return gated / (gated.sum(dim=-1, keepdim=True) + 1e-7)
    
    naive_time = benchmark_fn(scattering_naive, name="Naive Implementation")
    
    # Try fused version
    try:
        from src.kernels.scattering_gate_fused import FusedScatteringGate
        
        fused_gate = FusedScatteringGate(d_model=d_model).to(device)
        
        def run_fused():
            return fused_gate(G_ii, attn)
        
        fused_time = benchmark_fn(run_fused, name="Fused Scattering Gate")
        speedup = naive_time / fused_time
        print(f"  Speedup: {speedup:.2f}x")
        
    except Exception as e:
        print(f"  Fused version not available: {e}")


def benchmark_hyperbolic_distance():
    """Benchmark Batched Hyperbolic Distance"""
    print("\n" + "="*60)
    print("5. Hyperbolic Distance Benchmark")
    print("="*60)
    
    batch, seq_len, d_model = 4, 512, 256
    x = torch.randn(batch, seq_len, d_model, device=device)
    x = x / (x.norm(dim=-1, keepdim=True) + 0.1)  # Normalize to Poincaré ball
    c = 1.0
    
    # Naive loop implementation
    def distance_naive():
        sqrt_c = c ** 0.5
        norm = x.norm(dim=-1).clamp(max=1.0/sqrt_c - 1e-5)
        return (2 / sqrt_c) * torch.atanh(sqrt_c * norm)
    
    naive_time = benchmark_fn(distance_naive, name="Naive Implementation")
    
    # Try batched version
    try:
        from src.kernels.hyperbolic_distance_batch import BatchedHyperbolicDistance
        
        batched = BatchedHyperbolicDistance(curvature=c).to(device)
        
        def run_batched():
            return batched(x)
        
        batched_time = benchmark_fn(run_batched, name="Batched (Triton)")
        speedup = naive_time / batched_time
        print(f"  Speedup: {speedup:.2f}x")
        
    except Exception as e:
        print(f"  Batched version not available: {e}")


def benchmark_full_model():
    """Benchmark full Phase 8 model with optimizations on/off"""
    print("\n" + "="*60)
    print("6. Full Model Forward Pass Benchmark")
    print("="*60)
    
    try:
        from src.models.phase8.integrated_model import Phase8IntegratedModel
        from src.models.phase8.config import Phase8Config
        
        batch, seq_len = 2, 128
        d_model, n_layers = 256, 4
        
        # With optimizations
        config_opt = Phase8Config(
            vocab_size=1000,
            d_model=d_model,
            n_layers=n_layers,
            n_seq=seq_len,
            num_heads=4,
            use_fused_mobius=True,
            use_green_function_cache=True,
            use_parallel_ssm_scan=True,
            use_fused_scattering_gate=True,
            use_batched_hyperbolic_distance=True,
        )
        
        model_opt = Phase8IntegratedModel(config_opt).to(device).eval()
        x = torch.randint(0, 1000, (batch, seq_len), device=device)
        
        def forward_opt():
            with torch.no_grad():
                return model_opt(x)
        
        opt_time = benchmark_fn(forward_opt, warmup=3, iterations=10, name="With Optimizations")
        
        # Without optimizations
        config_noopt = Phase8Config(
            vocab_size=1000,
            d_model=d_model,
            n_layers=n_layers,
            n_seq=seq_len,
            num_heads=4,
            use_fused_mobius=False,
            use_green_function_cache=False,
            use_parallel_ssm_scan=False,
            use_fused_scattering_gate=False,
            use_batched_hyperbolic_distance=False,
        )
        
        model_noopt = Phase8IntegratedModel(config_noopt).to(device).eval()
        
        def forward_noopt():
            with torch.no_grad():
                return model_noopt(x)
        
        noopt_time = benchmark_fn(forward_noopt, warmup=3, iterations=10, name="Without Optimizations")
        
        speedup = noopt_time / opt_time
        print(f"  Overall Speedup: {speedup:.2f}x")
        
        # Tokens per second
        tokens = batch * seq_len
        print(f"\n  Throughput:")
        print(f"    With Opts: {tokens / (opt_time/1000):.0f} tokens/sec")
        print(f"    Without:   {tokens / (noopt_time/1000):.0f} tokens/sec")
        
    except Exception as e:
        import traceback
        print(f"  Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    print("="*60)
    print("Phase 8 Kernel Optimization Benchmark")
    print("="*60)
    
    benchmark_mobius_operations()
    benchmark_green_function_cache()
    benchmark_ssm_scan()
    benchmark_scattering_gate()
    benchmark_hyperbolic_distance()
    benchmark_full_model()
    
    print("\n" + "="*60)
    print("Benchmark Complete!")
    print("="*60)
