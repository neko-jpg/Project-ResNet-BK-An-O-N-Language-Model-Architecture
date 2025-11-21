import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import warnings

class ScatteringOperator(nn.Module):
    """
    Scattering Operator for Quantum Observation.
    Mimics the Phase 3 Scattering logic on vocabulary space.

    Implements a simplified Resolvent-based scattering: S(z) = (H - z)^-1
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # H0: Free Hamiltonian (diagonal, e.g. kinetic energy or frequency)
        # We initialize it with a range to simulate spectrum
        self.register_buffer('H0', torch.linspace(-1, 1, dim))
        # V: Potential (interaction). simplified to diagonal for efficiency
        self.register_buffer('V_diag', torch.randn(dim) * 0.1)

    def forward(self, energy_complex: torch.Tensor) -> torch.Tensor:
        """
        Compute Resolvent R(z) = (H - z)^-1

        Args:
            energy_complex: (B, N, 1) complex energy parameter

        Returns:
            R_z: (B, N, V) complex resolvent diagonal
        """
        # H = H0 + V
        H_diag = self.H0 + self.V_diag # (V,)

        # R(z) = 1 / (H - z)
        # energy_complex: (B, N, 1)

        denom = H_diag.unsqueeze(0).unsqueeze(0) - energy_complex # (B, N, V)
        R_z = 1.0 / denom

        return R_z

class QuantumObserver(nn.Module):
    """
    Quantum Observer: Wave Function Collapse via von Neumann Projection.

    Physics-informed attention/selection mechanism that uses scattering theory
    to "collapse" the probability distribution.
    """
    def __init__(
        self,
        vocab_size: int,
        n_candidates: int = 3,
        entropy_threshold: float = 0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_candidates = n_candidates
        self.entropy_threshold = entropy_threshold

        self.scattering_op = ScatteringOperator(vocab_size)

        self.logical_entropy_net = nn.Sequential(
            nn.Linear(vocab_size, vocab_size // 16),
            nn.Tanh(),
            nn.Linear(vocab_size // 16, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        logits: torch.Tensor,
        user_prompt: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            logits: (B, N, V)
            user_prompt: (B, N, V) one-hot or embedding (optional)
        """
        B, N, V = logits.shape

        # 1. Superposition (Top-K candidates)
        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, self.n_candidates, dim=-1)
        # Normalize restricted probabilities
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # 2. Observation Operator P_obs
        if user_prompt is not None:
            # Calculate logical entropy from prompt context
            # Assume user_prompt is (B, N, V) or similar.
            # We project it to scalar.
            logical_entropy = self.logical_entropy_net(user_prompt.float()).squeeze(-1) # (B, N)
            collapse_strength = 1.0 - logical_entropy
        else:
            collapse_strength = torch.ones(B, N, device=logits.device)
            logical_entropy = None

        # Construct P_obs (Spectral Density)
        obs_operator = self._construct_observation_operator(logits, collapse_strength) # (B, N, V)

        # 3. von Neumann Projection
        # Project the superposition state onto the subspace defined by the observation
        # We use the spectral density at the candidate indices as the projection weights

        obs_weights = torch.gather(obs_operator, -1, top_k_indices) # (B, N, K)

        # Apply projection: p'_i = p_i * w_i / Z
        collapsed_probs = top_k_probs * obs_weights
        collapsed_probs = collapsed_probs / (collapsed_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # 4. Entropy Monitor
        entropy_before = self._compute_entropy(top_k_probs)
        entropy_after = self._compute_entropy(collapsed_probs)
        entropy_reduction = (entropy_before - entropy_after) / (entropy_before + 1e-6)

        # 5. Collapse to unique reality (Argmax)
        final_indices_local = collapsed_probs.argmax(dim=-1, keepdim=True)
        collapsed_tokens = torch.gather(top_k_indices, -1, final_indices_local).squeeze(-1)

        diagnostics = {
            'superposition_probs': top_k_probs,
            'collapsed_probs': collapsed_probs,
            'entropy_before': entropy_before,
            'entropy_after': entropy_after,
            'entropy_reduction': entropy_reduction,
            'logical_entropy': logical_entropy
        }

        return collapsed_tokens, diagnostics

    def _construct_observation_operator(
        self,
        logits: torch.Tensor,
        collapse_strength: torch.Tensor
    ) -> torch.Tensor:
        """
        Construct P_obs using Stone's Formula on Scattering Operator.
        P_obs ~ Im(R(E+ie))
        """
        epsilon = 1e-3
        # Energy of the system (e.g. mean logit energy)
        energy = logits.mean(dim=-1, keepdim=True) # (B, N, 1)

        # R(E + i*eps)
        s_plus = self.scattering_op(energy + 1j * epsilon)
        # s_minus = self.scattering_op(energy - 1j * epsilon) # Conjugate

        # Spectral Density = -1/pi * Im(R(E+ie))
        obs_operator = -s_plus.imag / np.pi

        # Ensure positive (it should be for R(z) if H is hermitian and eps > 0?)
        # R(E+ie) = (H - E - ie)^-1 = (H-E + ie) / |...|^2
        # Im part is epsilon / |...|^2 > 0.
        # So -Im is negative?
        # Wait. 1/(x - iy) = (x + iy)/(x^2 + y^2). Im is y/...
        # 1/(H - E - ie). x = H-E, y = -eps.
        # Im is -eps / ...
        # So Im is negative. -Im is positive.
        # So obs_operator is positive.

        obs_operator = obs_operator.abs()

        # Apply collapse strength
        obs_operator = obs_operator * collapse_strength.unsqueeze(-1)

        return obs_operator

    def _compute_entropy(self, probs):
        return -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
