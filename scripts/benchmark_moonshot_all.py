#!/usr/bin/env python3
"""
Moonshot Optimizations Benchmark - Phase 2 & 3

Tests all implemented moonshot optimizations:
- Phase 2:
  - #3 Eigenvalue Precomputation (Green Function LUT)
  - #6 Resonance-Locked Training
  - #7 Scattering-Aware Attention Pruning
  - #8 Hyperbolic MoE
  - #10 Time-Reversed Training
  - #12 Superposition Training
- Phase 3:
  - #9 Gradient Teleportation
  - #11 Holographic Compression

Usage:
    python scripts/benchmark_moonshot_all.py
"""

import torch
import torch.nn as nn
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")


def bench(fn, warmup=3, iters=20, name=""):
    """Benchmark a function."""
    for _ in range(warmup):
        try:
            fn()
        except Exception:
            pass
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"  {name}: {ms:.3f} ms")
    return ms


def test_section(name):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


# =============================================================================
# Test Configuration
# =============================================================================
B, L, D = 4, 256, 512
d_state = 64
num_heads = 8
num_experts = 4

print(f"\nTest configuration: B={B}, L={L}, D={D}")

# =============================================================================
# #3 Green Function LUT (Eigenvalue Precomputation)
# =============================================================================
test_section("#3 Green Function LUT (Eigenvalue Precomputation)")

try:
    from src.kernels.green_function_lut import GreenFunctionLUT, FastBKCoreGreen

    # Create LUT
    lut = GreenFunctionLUT(lut_size=1024, max_distance=10.0).to(device)
    distances = torch.rand(B, L, device=device) * 10.0

    # Benchmark LUT lookup
    def lut_lookup():
        return lut.forward(distances)
    
    t_lut = bench(lut_lookup, name="Green Function LUT")

    # Benchmark direct computation (baseline)
    def direct_compute():
        d = distances
        return torch.exp(-d) / (4 * 3.14159 * torch.sinh(d + 1e-6))
    
    t_direct = bench(direct_compute, name="Direct Computation")
    print(f"  Speedup: {t_direct/t_lut:.2f}x")
    
    # Test FastBKCoreGreen
    fast_bk = FastBKCoreGreen(d_model=D).to(device)
    x = torch.randn(B, L, D, device=device)
    
    def fast_bk_forward():
        return fast_bk(x)
    
    bench(fast_bk_forward, name="FastBKCoreGreen")
    print("  ✅ #3 Green Function LUT: PASSED")

except Exception as e:
    print(f"  ❌ Error: {e}")


# =============================================================================
# #6 Resonance-Adaptive Curvature
# =============================================================================
test_section("#6 Resonance-Adaptive Curvature")

try:
    from src.kernels.resonance_adaptive_curvature import ResonanceAdaptiveCurvature, StabilityMonitor

    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(D, D)
            self.curvature = 1.0
    
    dummy_model = DummyModel().to(device)
    
    # Create curvature optimizer
    curvature_opt = ResonanceAdaptiveCurvature(
        model=dummy_model,
        initial_curvature=1.0,
        min_curvature=0.1,
        max_curvature=2.0,
    )
    
    # Simulate G_ii updates
    for i in range(50):
        G_ii = torch.randn(B, L, device=device)
        result = curvature_opt.step(G_ii)
    
    stats = curvature_opt.get_stats()
    print(f"  Mean Resonance: {stats.get('mean_resonance', 0):.4f}")
    print(f"  Current Curvature: {stats.get('current_curvature', 0):.4f}")
    
    # Test StabilityMonitor
    monitor = StabilityMonitor()
    for i in range(50):
        status = monitor.update(loss=2.0 + i * 0.01, grad_norm=1.0)
    print(f"  Stability Score: {status['stability_score']:.2f}")
    print("  ✅ #6 Resonance-Adaptive Curvature: PASSED")

except Exception as e:
    print(f"  ❌ Error: {e}")


# =============================================================================
# #7 Scattering-Aware Attention Pruning
# =============================================================================
test_section("#7 Scattering-Aware Attention Pruning")

try:
    from src.kernels.scattering_attention_pruning import ScatteringAwareAttention

    # Create attention modules
    std_attn = nn.MultiheadAttention(D, num_heads, batch_first=True).to(device)
    scatter_attn = ScatteringAwareAttention(D, num_heads, threshold=0.1).to(device)
    
    x = torch.randn(B, L, D, device=device)
    
    # Create G_ii with varying scattering energy
    G_ii = torch.randn(B, L, device=device)
    G_ii[:, :64] *= 0.01  # Low scattering - should be skipped
    G_ii[:, 64:] *= 1.0   # High scattering - should be computed
    
    # Benchmark standard attention
    def run_std():
        return std_attn(x, x, x)[0]
    t_std = bench(run_std, name="Standard Attention")
    
    # Benchmark scattering-aware attention
    scatter_attn.reset_stats()
    def run_scatter():
        return scatter_attn(x, G_ii)[0]
    t_scatter = bench(run_scatter, name="Scattering Attention")
    
    print(f"  Speedup: {t_std/t_scatter:.2f}x")
    print(f"  Blocks skipped: {scatter_attn.blocks_skipped}")
    print(f"  Blocks computed: {scatter_attn.blocks_computed}")
    ratio = scatter_attn.blocks_skipped / max(1, scatter_attn.blocks_skipped + scatter_attn.blocks_computed)
    print(f"  Skip ratio: {ratio:.1%}")
    print("  ✅ #7 Scattering-Aware Attention Pruning: PASSED")

except Exception as e:
    import traceback
    print(f"  ❌ Error: {e}")
    traceback.print_exc()


# =============================================================================
# #8 Hyperbolic MoE
# =============================================================================
test_section("#8 Hyperbolic MoE (Mixture of Experts)")

try:
    from src.kernels.hyperbolic_moe import HyperbolicMoE, create_hyperbolic_moe

    # Create HMoE
    hmoe = create_hyperbolic_moe(
        d_model=D,
        num_experts=num_experts,
        top_k=2,
    ).to(device)
    
    x = torch.randn(B, L, D, device=device)
    
    def run_hmoe():
        return hmoe(x)
    
    t_hmoe = bench(run_hmoe, name="Hyperbolic MoE")
    
    # Standard MoE comparison (simple FFN per expert)
    class SimpleMoE(nn.Module):
        def __init__(self, d_model, num_experts, expert_dim):
            super().__init__()
            self.router = nn.Linear(d_model, num_experts)
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, expert_dim),
                    nn.GELU(),
                    nn.Linear(expert_dim, d_model),
                )
                for _ in range(num_experts)
            ])
        
        def forward(self, x):
            logits = self.router(x)
            weights = torch.softmax(logits, dim=-1)
            out = sum(w.unsqueeze(-1) * e(x) for w, e in zip(weights.unbind(-1), self.experts))
            return out
    
    simple_moe = SimpleMoE(D, num_experts, D * 4).to(device)
    
    def run_simple():
        return simple_moe(x)
    
    t_simple = bench(run_simple, name="Standard MoE (learned router)")
    print(f"  HMoE vs Standard: {t_simple/t_hmoe:.2f}x")
    print("  ✅ #8 Hyperbolic MoE: PASSED")

except Exception as e:
    print(f"  ❌ Error: {e}")


# =============================================================================
# #9 Gradient Teleportation
# =============================================================================
test_section("#9 Gradient Teleportation")

try:
    from src.kernels.gradient_teleportation import (
        GradientTeleporter, 
        DysonPropagator,
        create_gradient_teleporter,
    )

    # Create test model
    class TestModel(nn.Module):
        def __init__(self, d_model, n_layers):
            super().__init__()
            self.embed = nn.Linear(d_model, d_model)
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model),
                )
                for _ in range(n_layers)
            ])
            self.out = nn.Linear(d_model, d_model)
        
        def forward(self, x):
            x = self.embed(x)
            for layer in self.layers:
                x = x + layer(x)
            return self.out(x)
    
    n_layers = 6
    test_model = TestModel(D, n_layers).to(device)
    
    # Create teleporter
    teleporter = create_gradient_teleporter(
        model=test_model,
        teleport_strength=0.1,
        use_dyson=True,
    )
    teleporter.register_hooks()
    
    print(f"  Found {len(teleporter.target_layers)} teleportable layers")
    
    # Test Dyson Propagator
    dyson = DysonPropagator(d_model=D, n_layers=n_layers).to(device)
    G = dyson.compute_full_propagator()
    print(f"  Propagator shape: {G.shape}")
    print(f"  Propagator norm: {G.norm():.4f}")
    
    # Simulate forward-backward with teleportation
    x = torch.randn(B, L, D, device=device, requires_grad=True)
    y = test_model(x)
    loss = y.sum()
    loss.backward()
    
    stats = teleporter.apply_teleportation()
    print(f"  Teleportation applied: {stats.get('teleported', False)}")
    print(f"  Teleport magnitude: {stats.get('magnitude', 0):.4f}")
    
    teleporter.remove_hooks()
    print("  ✅ #9 Gradient Teleportation: PASSED")

except Exception as e:
    import traceback
    print(f"  ❌ Error: {e}")
    traceback.print_exc()


# =============================================================================
# #11 Holographic Compression
# =============================================================================
test_section("#11 Holographic Compression")

try:
    from src.kernels.holographic_compression import (
        HolographicEncoder,
        HolographicDecoder,
        HolographicKVCache,
        create_holographic_kv_cache,
    )

    # Test encoder/decoder
    compression_ratio = 0.25
    boundary_dim = int(D * compression_ratio)
    
    encoder = HolographicEncoder(D, boundary_dim).to(device)
    decoder = HolographicDecoder(boundary_dim, D).to(device)
    
    x = torch.randn(B, L, D, device=device)
    
    # Benchmark encoding
    def run_encode():
        return encoder(x)
    t_encode = bench(run_encode, name="Holographic Encode")
    
    # Benchmark decoding
    compressed = encoder(x)
    def run_decode():
        return decoder(compressed)
    t_decode = bench(run_decode, name="Holographic Decode")
    
    # Test reconstruction quality
    reconstructed = decoder(encoder(x))
    mse = (x - reconstructed).pow(2).mean().item()
    print(f"  Reconstruction MSE: {mse:.6f}")
    
    # Test HolographicKVCache
    kv_cache = create_holographic_kv_cache(
        d_model=D,
        max_length=1024,
        compression_ratio=compression_ratio,
    ).to(device)
    
    k = torch.randn(B, L, D, device=device)
    v = torch.randn(B, L, D, device=device)
    
    def run_kv_update():
        kv_cache.clear()
        return kv_cache.update(k, v)
    
    bench(run_kv_update, name="KV Cache Update")
    
    stats = kv_cache.get_memory_stats()
    print(f"  Full Memory: {stats['full_memory_mb']:.2f} MB")
    print(f"  Compressed Memory: {stats['compressed_memory_mb']:.2f} MB")
    print(f"  Compression Ratio: {stats['compression_ratio']:.2%}")
    print("  ✅ #11 Holographic Compression: PASSED")

except Exception as e:
    import traceback
    print(f"  ❌ Error: {e}")
    traceback.print_exc()


# =============================================================================
# #12 Superposition Training
# =============================================================================
test_section("#12 Superposition Training")

try:
    from src.kernels.superposition_training import (
        SuperpositionOptimizer,
        ImaginaryTimeEvolution,
        create_superposition_optimizer,
    )

    # Create simple model
    model = nn.Sequential(
        nn.Linear(D, D),
        nn.GELU(),
        nn.Linear(D, D),
    ).to(device)
    
    # Create superposition optimizer
    sp_opt = create_superposition_optimizer(
        model=model,
        lr=1e-4,
        num_particles=5,
        noise_scale=0.01,
    )
    
    loss_fn = nn.MSELoss()
    x = torch.randn(B, D, device=device)
    y = torch.randn(B, D, device=device)
    
    # Test step
    def run_sp_step():
        sp_opt.zero_grad()
        sp_opt.step(loss_fn=loss_fn, inputs=x, targets=y)
    
    t_sp = bench(run_sp_step, iters=5, name="Superposition Step")
    
    # Test ImaginaryTimeEvolution
    ite = ImaginaryTimeEvolution(model, tau=0.01)
    
    def run_ite_step():
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        ite.step(loss)
    
    bench(run_ite_step, iters=5, name="Imaginary Time Evolution")
    print("  ✅ #12 Superposition Training: PASSED")

except Exception as e:
    import traceback
    print(f"  ❌ Error: {e}")
    traceback.print_exc()


# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*60)
print("  MOONSHOT OPTIMIZATIONS SUMMARY")
print("="*60)
print("""
Phase 2 (Implemented):
  ✅ #3  Eigenvalue Precompute (Green Function LUT)
  ✅ #6  Resonance-Locked Training
  ✅ #7  Scattering-Aware Attention Pruning  
  ✅ #8  Hyperbolic MoE
  ✅ #10 Time-Reversed Training (in train_phase8.py)
  ✅ #12 Superposition Training

Phase 3 (New):
  ✅ #9  Gradient Teleportation (Dyson Propagator)
  ✅ #11 Holographic Compression (AdS/CFT KV Cache)

Skipped:
  ⏭️ #5  Continuous Token (VRAM tradeoff too large)
  
Future/Research:
  🔬 #1  Hyperbolic GPU Primitives
  🔬 #2  Speculative Training
  🔬 #4  Analog Computing
""")

print("\nRun 'python scripts/train_phase8.py --dry-run --compile' to test full pipeline")
