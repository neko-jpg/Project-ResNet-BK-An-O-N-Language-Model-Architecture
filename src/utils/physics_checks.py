"""
Physics utility functions for stability and unitarity diagnostics.

Provides:
- Non-Hermitian decay stability analysis (Γ dynamics over time)
- Scattering matrix unitarity error computation and threshold suggestion
- CUDA VRAM emulation helper for 8GB target development

All functions are lightweight (O(N)) and CPU-friendly by default.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch


def analyze_gamma_stability(
    gamma: torch.Tensor,
    time_horizon: float = 10.0,
    vanish_threshold: float = 1e-4,
    blowup_threshold: float = 1e3,
) -> Dict[str, float]:
    """
    Evaluate stability of decay rates Γ over a time horizon.

    物理的直観: |ψ| ~ exp(-Γ t)。Γが大きすぎると情報が即死、負や極小だと発散。

    Args:
        gamma: Decay rates (any shape), expected non-negative.
        time_horizon: Simulated time horizon.
        vanish_threshold: Amplitude below this is treated as vanished.
        blowup_threshold: Amplitude above this is treated as divergent.

    Returns:
        Dictionary with fractions of vanished/divergent/healthy trajectories.
    """
    if gamma.is_complex():
        gamma = gamma.real

    amp = torch.exp(-gamma * time_horizon)
    vanished = (amp < vanish_threshold).float().mean().item()
    divergent = (amp > blowup_threshold).float().mean().item()
    healthy = 1.0 - vanished - divergent

    return {
        "time_horizon": float(time_horizon),
        "vanish_threshold": float(vanish_threshold),
        "blowup_threshold": float(blowup_threshold),
        "vanished_frac": float(vanished),
        "divergent_frac": float(divergent),
        "healthy_frac": float(healthy),
        "gamma_mean": float(gamma.mean().item()),
        "gamma_std": float(gamma.std().item()),
        "gamma_min": float(gamma.min().item()),
        "gamma_max": float(gamma.max().item()),
    }


def compute_unitarity_error(S: torch.Tensor) -> float:
    """
    Compute Frobenius norm of unitarity defect ||S^H S - I||_F / N.

    Args:
        S: (..., N, N) complex scattering matrices.

    Returns:
        Mean Frobenius norm per matrix.
    """
    if not torch.is_complex(S):
        S = S.to(torch.complex64)
    n = S.shape[-1]
    eye = torch.eye(n, device=S.device, dtype=S.dtype)
    defect = torch.matmul(S.conj().transpose(-1, -2), S) - eye
    norm = torch.linalg.norm(defect, dim=(-2, -1))
    return float(norm.mean().item() / math.sqrt(n))


def sweep_unitarity_threshold(
    noise_levels: Iterable[float],
    trials: int = 64,
) -> Tuple[float, Dict[str, float]]:
    """
    Sweep noise levels to suggest an anomaly threshold for unitarity violation.

    Args:
        noise_levels: Iterable of additive noise scales applied to unitary matrices.
        trials: Number of random samples per noise level.

    Returns:
        (recommended_threshold, stats) where stats maps level -> mean error.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = {}
    for eps in noise_levels:
        # Construct Haar-random unitary via QR, then inject noise.
        z = torch.randn(trials, 4, 4, device=device) + 1j * torch.randn(trials, 4, 4, device=device)
        q, _ = torch.linalg.qr(z)
        noisy = q + eps * (torch.randn_like(q) + 1j * torch.randn_like(q))
        stats[eps] = compute_unitarity_error(noisy)
    # Recommend threshold slightly above the 90th percentile of errors at smallest noise.
    base_level = min(noise_levels)
    recommended = max(stats[base_level] * 5.0, 1e-4)
    return float(recommended), {str(k): float(v) for k, v in stats.items()}


def ensure_results_dir(path: Path) -> None:
    """
    Ensure parent directory exists for result files.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def set_cuda_vram_limit(target_gb: float = 8.0) -> Dict[str, float]:
    """
    Soft-limit CUDA memory using PyTorch's per-process fraction API.

    Note: This is a cooperative limit (not a hard cgroup cap) but helps enforce
    8GB開発ターゲットでの挙動確認。

    Args:
        target_gb: Desired memory cap in GB.

    Returns:
        Dictionary with configured fraction and detected total memory.
    """
    if not torch.cuda.is_available():
        return {"configured": 0.0, "total_gb": 0.0, "note": "cuda_unavailable"}

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024**3)
    fraction = min(1.0, target_gb / total_gb)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    return {"configured": fraction, "total_gb": total_gb, "note": "cuda_fraction_set"}
