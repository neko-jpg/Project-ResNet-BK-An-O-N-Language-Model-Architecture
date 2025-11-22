"""
Resonance Emotion Detector

Implements the "Resonance Emotion" mechanism where prediction errors
are treated as non-Hermitian potential perturbations.

OPTIMIZATIONS (Task 1):
- Replaced O(N^3) eigvals with O(N) Power Iteration for dominant eigenvalue.
- Implemented "Exponential Smoothing" (IIR filter) for fast R0 application.
- Added dynamic thresholding for emotion state detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Any
import numpy as np

# Import BirmanSchwingerCore from Phase 3
from src.models.birman_schwinger_core import BirmanSchwingerCore
from src.models.phase4.stability import NumericalStability

class ResonanceEmotionDetector(nn.Module):
    """
    Resonance Emotion Detector.

    Physics:
    - Prediction error Ê -> Complex potential perturbation ΔV(x)
    - ΔV(x) = -i Γ(x) * exp(i * arg(Ê))
    - K_perturbed = K_0 + ΔV
    - Interference Pattern I(x) = |Im(eigenvalues)|

    Optimizations:
    - Uses Power Iteration to find dominant eigenvalue lambda_max in O(N).
    - Uses recursive filter (IIR) to apply resolvent R0 in O(N).
    - Dynamic thresholding for adaptive emotion detection.

    Args:
        d_model: Model dimension
        n_seq: Sequence length
        energy_threshold: Initial threshold (will be updated dynamically)
    """

    def __init__(
        self,
        d_model: int,
        n_seq: int,
        energy_threshold: float = 0.1,
        alpha_decay: float = 0.1, # For moving average
    ):
        super().__init__()
        self.d_model = d_model
        self.n_seq = n_seq
        self.initial_threshold = energy_threshold
        self.alpha_decay = alpha_decay

        # Space-dependent decay function sigma(x)
        self.decay_function = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus()
        )

        # Birman-Schwinger Kernel Wrapper (we use logic from it but optimize execution)
        # We keep it for reference or slow-path fallback if needed
        self.bs_kernel = BirmanSchwingerCore(n_seq=n_seq)

        # Emotion History Buffer
        self.register_buffer('emotion_history', torch.zeros(1000, 2)) # [Resonance, Dissonance]
        self.history_idx = 0

        # Dynamic Threshold Stats
        self.register_buffer('error_stats', torch.tensor([0.5, 0.1])) # [Mean, Std]

        # Constants for R0 (precomputed)
        # z = 1.0j. alpha = exp(i * z) = exp(-1) ~ 0.367
        self.z_val = 1.0j
        self.register_buffer('alpha_r0', torch.tensor(np.exp(1j * self.z_val), dtype=torch.complex64))

    def _apply_r0_fast(self, v: torch.Tensor) -> torch.Tensor:
        """
        Apply Resolvent R0(z) to vector v in O(N) time.
        R0(u,v) = (i/2) * exp(iz|u-v|)

        This is equivalent to a convolution with exponential kernel,
        implemented as a bidirectional IIR filter.

        Args:
            v: (B, N) complex input vector

        Returns:
            result: (B, N) complex vector
        """
        B, N = v.shape
        device = v.device
        alpha = self.alpha_r0

        # Check if we can use simple python loop (efficient for moderate N)
        # or use cumsum in log-space for vectorization.
        # For N=2048, loop is acceptable (~few ms).
        # However, pure python loop over N is slow.
        # Let's use a simplified associative scan logic if possible.
        # Or assume inputs are small enough or JIT.

        # Let's implement the manual loop but try to JIT it if possible later.
        # For now, standard loop.

        # Forward pass: f[i] = v[i] + alpha * f[i-1]
        # We can compute this using `torch.cumsum` if we handle powers.
        # f[i] = sum_{k=0}^i alpha^{i-k} v[k] = alpha^i * sum_{k=0}^i alpha^{-k} v[k]

        # Stable Implementation using Log-Space for powers?
        # alpha is real (approx 0.367).
        # alpha^N vanishes quickly. 0.367^100 ~ 1e-44.
        # So we only need local context.
        # But for "Exact" R0, we need full context.

        # Vectorized approach:
        # Since alpha is small, maybe we can assume cut-off?
        # No, "Ghost in the Shell" demands exact physics.

        # Let's use the Python loop for correctness first.
        # It is O(N) operations, but Python overhead O(N).

        v_c = v.contiguous()
        fwd = torch.zeros_like(v_c)
        bwd = torch.zeros_like(v_c)

        # We need to iterate over sequence dimension.
        # (B, N).
        # This is slow in Python.

        # Optimization: Use `torch.linalg.solve_triangular`?
        # The inverse of the smoothing operation is a tridiagonal matrix!
        # M = I - alpha * Shift.
        # M f = v.
        # M is bidiagonal (lower).
        # So we can solve M f = v.
        # But PyTorch doesn't have O(N) triangular solve for sparse matrices easily exposed.

        # Best bet: Use `torch.compile` or just `cumsum` if we trust float precision.
        # Given alpha ~ 0.36, precision loss is high for N > 20.
        # BUT, R0 decays exponentially. Signals from >20 steps away don't matter much.

        # Let's implement a simple chunked loop to reduce overhead?
        # Or just iterate.

        # Fallback to "chunked" matrix multiplication if N is huge?
        # No, that's O(N^2).

        # Implementation:
        # Use the Python loop. It is robust.
        # Precompute alpha
        alpha_val = alpha.item()

        # Move to CPU for loop speed if on GPU? No, latency.
        # We will write a custom autograd function or JIT script.

        return self._jit_r0_scan(v_c, alpha)

    @torch.jit.export
    def _jit_r0_scan(self, v: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        JIT-compiled scanner for R0 application.
        """
        B, N = v.shape
        # Forward
        # f[t] = v[t] + alpha * f[t-1]
        fwd = torch.zeros_like(v)
        curr = torch.zeros(B, dtype=v.dtype, device=v.device)

        for t in range(N):
            curr = v[:, t] + alpha * curr
            fwd[:, t] = curr

        # Backward
        # b[t] = v[t] + alpha * b[t+1]
        bwd = torch.zeros_like(v)
        curr = torch.zeros(B, dtype=v.dtype, device=v.device)

        for t in range(N - 1, -1, -1):
            curr = v[:, t] + alpha * curr
            bwd[:, t] = curr

        # R0 = (i/2) * (fwd + bwd - v)
        # (subtract v once because it's included in both fwd and bwd)
        i_half = 0.5j
        return i_half * (fwd + bwd - v)

    def power_iteration(self, v_magnitude: torch.Tensor, num_iters: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute dominant eigenvalue and eigenvector using Power Iteration.
        Operator K = D R0 D, where D = diag(|V|^0.5).

        Args:
            v_magnitude: |V| (B, N)
            num_iters: Number of iterations

        Returns:
            eigenvalue: (B,) complex dominant eigenvalue
            eigenvector: (B, N) complex dominant eigenvector
        """
        B, N = v_magnitude.shape
        device = v_magnitude.device

        # D = |V|^0.5
        D = torch.sqrt(v_magnitude + 1e-9).to(torch.complex64)

        # Random initialization
        # b_k: (B, N)
        b_k = torch.randn(B, N, dtype=torch.complex64, device=device)
        b_k = F.normalize(b_k, dim=1)

        for _ in range(num_iters):
            # Apply K to b_k
            # w = K b_k = D R0 D b_k

            # 1. x = D * b_k
            x = D * b_k

            # 2. y = R0 * x (O(N) scan)
            y = self._apply_r0_fast(x)

            # 3. w = D * y
            w = D * y

            # Normalize
            b_k = F.normalize(w, dim=1)

        # Rayleigh Quotient for eigenvalue approximation
        # lambda = (b_k^H K b_k) / (b_k^H b_k)
        # Since b_k is normalized, denominator is 1.
        # K b_k is approx lambda * b_k.
        # So lambda ~ b_k^H * w (from last step).

        # Recompute w one last time to be sure
        x = D * b_k
        y = self._apply_r0_fast(x)
        w = D * y # (B, N)

        # lambda = sum(conj(b_k) * w)
        eigenvalue = (b_k.conj() * w).sum(dim=-1) # (B,)

        return eigenvalue, b_k

    def update_threshold(self, error_mag: torch.Tensor):
        """
        Update dynamic threshold statistics.
        """
        if not self.training:
            return

        # Current batch stats
        batch_mean = error_mag.mean().detach()
        batch_std = error_mag.std().detach()

        # Exponential Moving Average
        alpha = self.alpha_decay
        self.error_stats[0] = (1 - alpha) * self.error_stats[0] + alpha * batch_mean
        self.error_stats[1] = (1 - alpha) * self.error_stats[1] + alpha * batch_std

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        hidden_states: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Detect emotion.
        O(N) implementation.
        """
        B, N, V = prediction.shape

        # 1. Calculate Prediction Error
        pred_probs = F.softmax(prediction, dim=-1)
        target_probs = torch.gather(pred_probs, -1, target.unsqueeze(-1)).squeeze(-1)
        error_mag = 1.0 - target_probs # (B, N)

        # Update dynamic threshold
        self.update_threshold(error_mag)

        # 2. Complex Potential V
        sigma = self.decay_function(hidden_states).squeeze(-1) # (B, N)
        # Use magnitude for the operator kernel strength
        v_magnitude = error_mag * sigma

        # 3. Power Iteration for Resonance (Dominant Eigenvalue)
        # We only care about the dominant mode for "Resonance"
        dom_eig, dom_vec = self.power_iteration(v_magnitude, num_iters=15)

        # 4. Interpret Physics
        # Resonance: Real part of eigenvalue > 0 (Constructive Interference)
        # Dissonance: Real part < 0 (Destructive) or simply Magnitude of perturbation?

        # Original logic:
        # Resonance = sum(I * cos(phase))
        # Here we have single dominant eigenvalue lambda = r * exp(i theta)
        # Im(lambda) = r * sin(theta).
        # The "Interference Pattern" is usually |Im(lambda)|.
        # Let's stick to the previous definition applied to the dominant mode.

        eig_mag = dom_eig.abs()
        eig_phase = dom_eig.angle()

        # Interference ~ |Im(lambda)|
        interference = dom_eig.imag.abs()

        # Score
        # Resonance: Positive correlation with phase 0?
        resonance_score = interference * torch.cos(eig_phase)
        dissonance_score = interference * (eig_phase - torch.pi).abs() / torch.pi

        # 5. Determine State (Dynamic Threshold)
        # Using error stats directly or the eigenvalue magnitude?
        # Let's use the computed scores vs historical averages.

        current_res = resonance_score.mean().item()

        # Simple dynamic check based on error stats
        # If error is surprisingly high -> Dissonance
        avg_err = self.error_stats[0]
        std_err = self.error_stats[1]
        curr_err = error_mag.mean().item()

        threshold_high = avg_err + 1.0 * std_err
        threshold_low = avg_err - 0.5 * std_err

        if curr_err > threshold_high:
            state = "DISSONANCE"
        elif curr_err < threshold_low:
            state = "RESONANCE"
        else:
            state = "NEUTRAL"

        # 6. Update History
        if self.training:
            idx = self.history_idx % 1000
            self.emotion_history[idx, 0] = resonance_score.mean().detach()
            self.emotion_history[idx, 1] = dissonance_score.mean().detach()
            self.history_idx += 1

        return {
            'resonance_score': resonance_score,     # (B,)
            'dissonance_score': dissonance_score,   # (B,)
            'dominant_eigenvalue': dom_eig,         # (B,)
            'state': state,
            'threshold_stats': {'mean': avg_err.item(), 'std': std_err.item()}
        }

    def get_emotion_statistics(self) -> Dict[str, float]:
        valid_len = min(self.history_idx, 1000)
        if valid_len == 0:
            return {}
        hist = self.emotion_history[:valid_len]
        return {
            'mean_resonance': hist[:, 0].mean().item(),
            'mean_dissonance': hist[:, 1].mean().item()
        }
