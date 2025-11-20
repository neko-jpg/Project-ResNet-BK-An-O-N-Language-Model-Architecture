"""
Non-Hermitian Potential Layer for Phase 2

Implements complex potential V - iΓ where:
- V (real part): Semantic potential
- Γ (imaginary part): Dissipation rate (always positive)

Physical interpretation:
- Open quantum system Hamiltonian: H_eff = H_0 + V - iΓ
- Time evolution: ||ψ(t)||² = exp(-2Γt) ||ψ(0)||²
- Natural forgetting through energy dissipation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import Optional, Dict, Tuple

from ..bk_core import BKCoreFunction


class NonHermitianPotential(nn.Module):
    """
    Complex potential generator: V - iΓ
    
    Args:
        d_model: Model dimension
        n_seq: Sequence length
        base_decay: Minimum decay rate (default: 0.01)
        adaptive_decay: Use input-dependent decay (default: True)
        schatten_p: Schatten norm p-value (default: 1.0 = Nuclear norm)
        stability_threshold: Threshold for stability warnings (default: 1e-3)
    """
    
    def __init__(
        self,
        d_model: int,
        n_seq: int,
        base_decay: float = 0.001,  # Reduced from 0.01 to prevent overdamping
        adaptive_decay: bool = True,
        schatten_p: float = 1.0,
        stability_threshold: float = 1e-3,
    ):
        super().__init__()
        
        if base_decay <= 0:
            raise ValueError(f"base_decay must be > 0, got {base_decay}")
        
        self.d_model = d_model
        self.n_seq = n_seq
        self.base_decay = base_decay
        self.adaptive_decay = adaptive_decay
        self.schatten_p = schatten_p
        self.stability_threshold = stability_threshold
        
        # Real part: semantic potential
        self.v_proj = nn.Linear(d_model, 1, bias=False)
        
        # Imaginary part: decay rate (always positive)
        if adaptive_decay:
            self.gamma_proj = nn.Linear(d_model, 1, bias=False)
        else:
            self.gamma_proj = None
        
        # Stability monitoring buffers
        self.register_buffer('gamma_history', torch.zeros(100))
        self.register_buffer('energy_ratio_history', torch.zeros(100))
        self.register_buffer('history_idx', torch.tensor(0, dtype=torch.long))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate complex potential from input features.
        
        Args:
            x: (B, N, D) input features
        
        Returns:
            complex_potential: (B, N) complex64 tensor
                real part: semantic potential V
                imag part: -Γ (negative for dissipation)
        """
        B, N, D = x.shape
        
        # Real part: semantic potential
        v = self.v_proj(x).squeeze(-1)  # (B, N)
        
        # Imaginary part: decay rate (ensure positive)
        if self.adaptive_decay and self.gamma_proj is not None:
            gamma_raw = self.gamma_proj(x).squeeze(-1)  # (B, N)
            # Softplus ensures positivity: softplus(x) = log(1 + exp(x))
            gamma = F.softplus(gamma_raw) + self.base_decay
        else:
            gamma = torch.full_like(v, self.base_decay)
        
        # Monitor stability during training
        if self.training:
            self._monitor_stability(v, gamma)
        
        # Return complex potential: V - iΓ
        # Note: negative imaginary part for dissipation
        return torch.complex(v, -gamma)
    
    def _monitor_stability(self, v: torch.Tensor, gamma: torch.Tensor):
        """
        Monitor Schatten norm and detect overdamping.
        
        Overdamping condition: Γ >> |V|
        When this occurs, the system becomes purely dissipative
        and information vanishes immediately.
        
        Args:
            v: (B, N) real potential
            gamma: (B, N) decay rate
        """
        with torch.no_grad():
            # Compute energy and damping statistics
            energy = torch.abs(v).mean()
            damping = gamma.mean()
            ratio = damping / (energy + 1e-6)
            
            # Update history buffers
            idx = self.history_idx.item() % 100
            self.gamma_history[idx] = damping
            self.energy_ratio_history[idx] = ratio
            self.history_idx += 1
            
            # Overdamping warning: Γ/|V| > 100 (increased threshold to reduce noise)
            # Only warn once every 100 steps to avoid spam
            if ratio > 100.0 and idx % 10 == 0:
                warnings.warn(
                    f"Overdamped system detected: Γ/|V| = {ratio:.2f}. "
                    f"Information may vanish too quickly. "
                    f"Consider reducing base_decay or checking input features.",
                    UserWarning
                )
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get stability statistics.
        
        Returns:
            Dictionary with gamma and energy ratio statistics
        """
        with torch.no_grad():
            valid_len = min(self.history_idx.item(), 100)
            if valid_len == 0:
                return {
                    'mean_gamma': 0.0,
                    'std_gamma': 0.0,
                    'mean_energy_ratio': 0.0,
                    'max_energy_ratio': 0.0,
                }
            
            valid_gamma = self.gamma_history[:valid_len]
            valid_ratio = self.energy_ratio_history[:valid_len]
            
            return {
                'mean_gamma': valid_gamma.mean().item(),
                'std_gamma': valid_gamma.std().item() if valid_len > 1 else 0.0,
                'mean_energy_ratio': valid_ratio.mean().item(),
                'max_energy_ratio': valid_ratio.max().item(),
            }


class DissipativeBKLayer(nn.Module):
    """
    Integrate NonHermitian potential with BK-Core.
    
    This layer combines:
    1. NonHermitianPotential: generates V - iΓ
    2. BK-Core: computes G_ii = diag((H - zI)^-1)
    
    The complex potential modulates the effective Hamiltonian,
    enabling natural forgetting through dissipation.
    
    Args:
        d_model: Model dimension
        n_seq: Sequence length
        use_triton: Use Triton kernel if available (default: True)
        **potential_kwargs: Arguments for NonHermitianPotential
    """
    
    def __init__(
        self,
        d_model: int,
        n_seq: int,
        use_triton: bool = True,
        **potential_kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_seq = n_seq
        self.use_triton = use_triton
        
        # Non-Hermitian potential generator
        self.potential = NonHermitianPotential(d_model, n_seq, **potential_kwargs)
        
        # BK-Core parameters (inherited from Phase 1)
        self.h0_super = nn.Parameter(torch.randn(n_seq - 1) * 0.1)
        self.h0_sub = nn.Parameter(torch.randn(n_seq - 1) * 0.1)
        self.z = nn.Parameter(torch.tensor(0.1 + 0.1j))
    
    def forward(
        self,
        x: torch.Tensor,
        return_potential: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with complex gradient support.
        
        Args:
            x: (B, N, D) input features
            return_potential: Return complex potential for diagnostics
        
        Returns:
            features: (B, N, 2) [Re(G_ii), Im(G_ii)]
            potential: (B, N) complex potential (if return_potential=True)
        """
        B, N, D = x.shape
        
        # Generate complex potential: V - iΓ
        V_complex = self.potential(x)  # (B, N) complex64
        
        # Extract real and imaginary parts
        v_real = V_complex.real  # (B, N)
        gamma = -V_complex.imag  # (B, N) positive decay rate
        
        # Construct effective Hamiltonian diagonal
        # he_diag = v_real (for now, can be extended to include -iΓ)
        he_diag = v_real
        
        # Expand h0_super and h0_sub to batch dimension
        h0_super_batch = self.h0_super.unsqueeze(0).expand(B, -1)
        h0_sub_batch = self.h0_sub.unsqueeze(0).expand(B, -1)
        
        # BK-Core computation with gradient support
        features = BKCoreFunction.apply(
            he_diag,
            h0_super_batch,
            h0_sub_batch,
            self.z,
            self.use_triton
        )  # (B, N, 2)
        
        if return_potential:
            return features, V_complex
        return features, None
    
    def get_gamma(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract decay rate Γ from input.
        
        Args:
            x: (B, N, D) input features
        
        Returns:
            gamma: (B, N) decay rate
        """
        V_complex = self.potential(x)
        return -V_complex.imag  # Positive decay rate
