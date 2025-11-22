"""
Resonance Emotion Detector

Implements the "Resonance Emotion" mechanism where prediction errors
are treated as non-Hermitian potential perturbations, and the resulting
interference patterns in the Birman-Schwinger kernel are interpreted as emotion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
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

    Args:
        d_model: Model dimension
        n_seq: Sequence length
        energy_threshold: Threshold for resonance detection (default: 0.1)
    """

    def __init__(
        self,
        d_model: int,
        n_seq: int,
        energy_threshold: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_seq = n_seq
        self.energy_threshold = energy_threshold

        # Space-dependent decay function sigma(x)
        # Maps hidden states to a scalar decay factor
        self.decay_function = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus()  # Ensure positive decay
        )

        # Birman-Schwinger Kernel (Base from Phase 3)
        # We wrap it or use it to compute perturbed kernel
        self.bs_kernel = BirmanSchwingerCore(n_seq=n_seq)

        # Emotion History Buffer (Resonance, Dissonance)
        # Fixed size buffer: 1000 steps
        self.register_buffer('emotion_history', torch.zeros(1000, 2))
        self.history_idx = 0

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        hidden_states: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Detect emotion from prediction error.

        Args:
            prediction: (B, N, vocab_size) Logits
            target: (B, N) Target token IDs
            hidden_states: (B, N, D) Hidden states from model

        Returns:
            emotion_info: Dictionary containing scores and patterns
        """
        B, N, V = prediction.shape
        device = prediction.device

        # 1. Calculate Prediction Error Ê
        # Simple difference in probability mass
        pred_probs = F.softmax(prediction, dim=-1)

        # Gather prob of target token
        # target: (B, N) -> (B, N, 1)
        target_probs = torch.gather(pred_probs, -1, target.unsqueeze(-1)).squeeze(-1)

        # Error magnitude: 1.0 - p(target)
        # If perfect prediction, error is 0.
        error_mag = 1.0 - target_probs # (B, N)

        # Error phase: Direction of error in logit space?
        # Simplified: Phase based on error magnitude (0 to pi)
        # Or use residual vector direction if we had it.
        # Here we construct a phase: arg(E) ~ error_mag * pi
        error_phase = error_mag * torch.pi

        # 2. Complex Potential Perturbation ΔV(x)
        # Gamma(x) = |E| * sigma(x)
        sigma = self.decay_function(hidden_states).squeeze(-1) # (B, N)
        gamma = error_mag * sigma # (B, N)

        # ΔV = -i * Gamma * exp(i * phase)
        #    = -i * Gamma * (cos(phi) + i sin(phi))
        #    = Gamma * sin(phi) - i * Gamma * cos(phi)
        delta_v_real = gamma * torch.sin(error_phase)
        delta_v_imag = -gamma * torch.cos(error_phase)
        delta_v = torch.complex(delta_v_real, delta_v_imag) # (B, N)

        # 3. Apply to Birman-Schwinger Kernel
        # K_perturbed = K_0 + delta_v (simplified interaction)
        # Ideally: K = |V+dV|^1/2 R_0 |V+dV|^1/2
        # We treat delta_v as the potential V for the kernel computation
        # The base potential V_0 is assumed to be 0 or absorbed in hidden_states

        # We compute K for the perturbation potential
        # K_p = BS(delta_v)
        # Note: BS Core expects real potential V usually?
        # BS Core computes K = |V|^1/2 R0 |V|^1/2.
        # If V is complex, |V| is magnitude.

        # Let's use the magnitude of perturbation as the potential strength
        # and the phase is encoded in the result?
        # No, BS Core R0 has complex phase.

        # We will compute K using |delta_v| as the potential V
        v_magnitude = delta_v.abs() # (B, N)

        # Compute BS operator
        # K: (B, N, N)
        # Use z=1.0j as standard spectral shift
        k_perturbed = self.bs_kernel.compute_birman_schwinger_operator(v_magnitude, z=1.0j)

        # Add the perturbation effect directly to K diagonal?
        # Or assume K represents the system state.
        # K is compact operator.

        # 4. Extract Interference Pattern
        # Eigenvalues of K
        # lambda_n

        # Stability Check: K must be finite
        k_perturbed = NumericalStability.sanitize_tensor(k_perturbed)

        try:
            eigenvalues = torch.linalg.eigvals(k_perturbed) # (B, N)
        except RuntimeError:
            # If eigvals fails (e.g. LAPACK error), use fallback or dummy
            # This can happen if matrix is singular or has Infs (already sanitized though)
            eigenvalues = torch.zeros(B, N, dtype=k_perturbed.dtype, device=device)

        # Interference pattern I(x) approx |Im(lambda)|
        interference_pattern = eigenvalues.imag.abs() # (B, N)

        # 5. Calculate Emotion Scores
        # Resonance: Real part > 0 (constructive?)
        # Dissonance: Real part < 0 or specific phase

        # Design doc:
        # resonance = sum(I(x) * cos(phase))
        # dissonance = sum(I(x) * |phase - pi|)
        # Here phase is phase of eigenvalue
        eig_phase = eigenvalues.angle() # (-pi to pi)

        resonance_score = (interference_pattern * torch.cos(eig_phase)).sum(dim=-1)
        dissonance_score = (interference_pattern * (eig_phase - torch.pi).abs()).sum(dim=-1)

        # Normalize
        resonance_score = resonance_score / N
        dissonance_score = dissonance_score / N

        # 6. Update History
        if self.training:
            idx = self.history_idx % 1000
            self.emotion_history[idx, 0] = resonance_score.mean().detach()
            self.emotion_history[idx, 1] = dissonance_score.mean().detach()
            self.history_idx += 1

        return {
            'resonance_score': resonance_score,
            'dissonance_score': dissonance_score,
            'interference_pattern': interference_pattern,
            'delta_v': delta_v,
            'eigenvalues': eigenvalues
        }

    def get_emotion_statistics(self) -> Dict[str, float]:
        """Get emotion statistics from history."""
        valid_len = min(self.history_idx, 1000)
        if valid_len == 0:
            return {
                'mean_resonance': 0.0,
                'mean_dissonance': 0.0,
                'std_resonance': 0.0,
                'std_dissonance': 0.0
            }

        history = self.emotion_history[:valid_len]
        return {
            'mean_resonance': history[:, 0].mean().item(),
            'mean_dissonance': history[:, 1].mean().item(),
            'std_resonance': history[:, 0].std().item(),
            'std_dissonance': history[:, 1].std().item(),
        }
