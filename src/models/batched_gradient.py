"""
Batched Analytic Gradient Computation with vmap
Vectorizes gradient computation across batch dimension for efficiency.
"""

import torch
import torch.nn as nn
from torch.func import vmap
from typing import Tuple


def compute_single_gradient(
    G_ii: torch.Tensor,
    grad_G: torch.Tensor,
    grad_blend: float
) -> torch.Tensor:
    """
    Compute gradient for a single sequence.
    
    Args:
        G_ii: (N,) complex resolvent diagonal
        grad_G: (N,) complex gradient w.r.t. G_ii
        grad_blend: blending coefficient
    
    Returns:
        grad_v: (N,) gradient w.r.t. potential
    """
    # Compute G² and 1/G² safely
    G_sq = G_ii ** 2

    # Stabilize denominator for 1/G²
    denom = G_sq
    denom_mag = denom.abs()
    min_denom = 1e-3
    denom = torch.where(
        denom_mag < min_denom,
        denom / (denom_mag + 1e-9) * min_denom,
        denom,
    )

    # Theoretical gradient: dG/dv = -G²
    grad_v_analytic = -(grad_G * G_sq).real

    # Hypothesis-7 gradient: inverse square type
    grad_v_h7 = -(grad_G / (denom + 1e-6)).real

    # Hybrid blend
    grad_v = (1.0 - grad_blend) * grad_v_analytic + grad_blend * grad_v_h7

    # Numerical safety
    grad_v = torch.where(torch.isfinite(grad_v), grad_v, torch.zeros_like(grad_v))
    grad_v = torch.clamp(grad_v, -1000.0, 1000.0)

    return grad_v


# Vectorized version using vmap
batched_compute_gradient = vmap(
    compute_single_gradient,
    in_dims=(0, 0, None),  # Batch over G_ii and grad_G, broadcast grad_blend
    out_dims=0
)


class BatchedAnalyticBKCoreFunction(torch.autograd.Function):
    """
    BK-Core with fully batched analytic gradient computation.
    
    Uses vmap to vectorize gradient computation across batch dimension,
    improving cache efficiency and reducing Python overhead.
    """
    GRAD_BLEND = 0.5

    @staticmethod
    def forward(ctx, he_diag, h0_super, h0_sub, z):
        """
        Forward pass: compute G_ii features.
        
        Args:
            he_diag: (B, N) effective Hamiltonian diagonal
            h0_super: (B, N-1) super-diagonal
            h0_sub: (B, N-1) sub-diagonal
            z: complex scalar shift
        
        Returns:
            features: (B, N, 2) [real(G_ii), imag(G_ii)]
        """
        from .bk_core import vmapped_get_diag
        
        # G_ii = diag((H - zI)^-1)
        G_ii = vmapped_get_diag(he_diag, h0_super, h0_sub, z)
        ctx.save_for_backward(G_ii)

        # Convert to real features
        output_features = torch.stack(
            [G_ii.real, G_ii.imag], dim=-1
        ).to(torch.float32)

        return output_features

    @staticmethod
    def backward(ctx, grad_output_features):
        """
        Batched backward pass using vmap.
        
        Args:
            grad_output_features: (B, N, 2) gradient w.r.t. output features
        
        Returns:
            grad_he_diag: (B, N) gradient w.r.t. effective Hamiltonian diagonal
        """
        (G_ii,) = ctx.saved_tensors  # (B, N) complex
        
        # dL/dG = dL/dRe(G) + i*dL/dIm(G)
        grad_G = torch.complex(
            grad_output_features[..., 0],
            grad_output_features[..., 1],
        )

        # Batched gradient computation using vmap
        grad_v = batched_compute_gradient(
            G_ii,
            grad_G,
            BatchedAnalyticBKCoreFunction.GRAD_BLEND
        )

        grad_he_diag = grad_v.to(torch.float32)

        # No gradients for h0_super, h0_sub, z
        return grad_he_diag, None, None, None


def compute_gradient_with_memory_optimization(
    G_ii: torch.Tensor,
    grad_G: torch.Tensor,
    grad_blend: float,
    chunk_size: int = 32
) -> torch.Tensor:
    """
    Compute gradients in chunks to optimize memory layout.
    
    Processes batch in chunks for better cache efficiency.
    
    Args:
        G_ii: (B, N) complex resolvent diagonal
        grad_G: (B, N) complex gradient w.r.t. G_ii
        grad_blend: blending coefficient
        chunk_size: number of sequences to process at once
    
    Returns:
        grad_v: (B, N) gradient w.r.t. potential
    """
    B, N = G_ii.shape
    grad_v = torch.zeros(B, N, dtype=torch.float32, device=G_ii.device)
    
    # Process in chunks
    for i in range(0, B, chunk_size):
        end_idx = min(i + chunk_size, B)
        chunk_G_ii = G_ii[i:end_idx]
        chunk_grad_G = grad_G[i:end_idx]
        
        # Compute gradient for chunk
        chunk_grad_v = batched_compute_gradient(
            chunk_G_ii,
            chunk_grad_G,
            grad_blend
        )
        
        grad_v[i:end_idx] = chunk_grad_v
    
    return grad_v


class MemoryOptimizedBKCoreFunction(torch.autograd.Function):
    """
    BK-Core with memory-optimized batched gradient computation.
    
    Processes gradients in chunks for better cache efficiency.
    """
    GRAD_BLEND = 0.5
    CHUNK_SIZE = 32

    @staticmethod
    def forward(ctx, he_diag, h0_super, h0_sub, z):
        """Forward pass."""
        from .bk_core import vmapped_get_diag
        
        G_ii = vmapped_get_diag(he_diag, h0_super, h0_sub, z)
        ctx.save_for_backward(G_ii)

        output_features = torch.stack(
            [G_ii.real, G_ii.imag], dim=-1
        ).to(torch.float32)

        return output_features

    @staticmethod
    def backward(ctx, grad_output_features):
        """Memory-optimized backward pass."""
        (G_ii,) = ctx.saved_tensors
        
        grad_G = torch.complex(
            grad_output_features[..., 0],
            grad_output_features[..., 1],
        )

        # Chunked gradient computation
        grad_v = compute_gradient_with_memory_optimization(
            G_ii,
            grad_G,
            MemoryOptimizedBKCoreFunction.GRAD_BLEND,
            MemoryOptimizedBKCoreFunction.CHUNK_SIZE
        )

        grad_he_diag = grad_v.to(torch.float32)

        return grad_he_diag, None, None, None


def profile_batched_gradient(
    batch_sizes: list = [1, 4, 8, 16, 32, 64],
    seq_len: int = 128,
    num_trials: int = 100,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> dict:
    """
    Profile performance of batched gradient computation.
    
    Args:
        batch_sizes: list of batch sizes to test
        seq_len: sequence length
        num_trials: number of trials for timing
        device: torch device
    
    Returns:
        results: dictionary with timing results
    """
    import time
    from .bk_core import BKCoreFunction
    
    results = {
        'batch_sizes': batch_sizes,
        'sequential_times': [],
        'batched_times': [],
        'memory_optimized_times': [],
        'speedups': []
    }
    
    for batch_size in batch_sizes:
        print(f"Profiling batch_size={batch_size}...")
        
        # Setup inputs
        he_diag = torch.randn(batch_size, seq_len, device=device)
        h0_super = torch.ones(batch_size, seq_len-1, device=device)
        h0_sub = torch.ones(batch_size, seq_len-1, device=device)
        z = torch.tensor(1.0j, dtype=torch.complex64, device=device)
        grad_output = torch.randn(batch_size, seq_len, 2, device=device)
        
        # Benchmark sequential (original)
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        for _ in range(num_trials):
            features = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z)
            features.backward(grad_output)
        torch.cuda.synchronize() if device == 'cuda' else None
        time_sequential = time.time() - start
        
        # Benchmark batched
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        for _ in range(num_trials):
            features = BatchedAnalyticBKCoreFunction.apply(he_diag, h0_super, h0_sub, z)
            features.backward(grad_output)
        torch.cuda.synchronize() if device == 'cuda' else None
        time_batched = time.time() - start
        
        # Benchmark memory-optimized
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        for _ in range(num_trials):
            features = MemoryOptimizedBKCoreFunction.apply(he_diag, h0_super, h0_sub, z)
            features.backward(grad_output)
        torch.cuda.synchronize() if device == 'cuda' else None
        time_memory_opt = time.time() - start
        
        results['sequential_times'].append(time_sequential / num_trials)
        results['batched_times'].append(time_batched / num_trials)
        results['memory_optimized_times'].append(time_memory_opt / num_trials)
        results['speedups'].append(time_sequential / time_batched)
        
        print(f"  Sequential: {time_sequential/num_trials*1000:.2f}ms")
        print(f"  Batched: {time_batched/num_trials*1000:.2f}ms")
        print(f"  Memory-optimized: {time_memory_opt/num_trials*1000:.2f}ms")
        print(f"  Speedup: {time_sequential/time_batched:.2f}x")
    
    return results
