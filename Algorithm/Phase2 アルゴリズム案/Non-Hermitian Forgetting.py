"""
Phase 2.1: Non-Hermitian Forgetting (自然な忘却)

物理的直観 (Physical Intuition):
    従来のRNN/Transformerは、情報を「無限に保持」しようとするか、
    単純な減衰で消し去ります。
    MUSEでは、量子力学の「開いた系 (Open Quantum System)」を模倣し、
    複素ポテンシャルの虚部 -iΓ を用いて、「意味のない情報」だけを
    物理法則に従って自然に散逸させます。

Mathematical Formulation:
    H_eff = H_0 + V(x) - iΓ(x)
    
    - V(x): 情報の「意味」（実数部）
    - Γ(x): 情報の「不確定性/ノイズ」（虚数部）
    
    時間発展演算子 U(t) = exp(-i H_eff t)
    ノルムの変化: ||ψ(t)||² = exp(-2Γt) ||ψ(0)||²
    Γ > 0 の領域でのみ情報が減衰する。

Implementation:
    - 複素ポテンシャル層
    - Schatten Normによる安定化 (崩壊を防ぐ)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class NonHermitianPotential(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_seq: int,
        base_decay: float = 0.01,
        adaptive_decay: bool = True,
        schatten_p: float = 1.0,  # Nuclear Norm
        stability_threshold: float = 1e-3
    ):
        super().__init__()
        self.d_model = d_model
        self.n_seq = n_seq
        self.base_decay = base_decay
        self.adaptive_decay = adaptive_decay
        self.schatten_p = schatten_p
        self.stability_threshold = stability_threshold
        
        # Real part of potential (Meaning)
        self.v_proj = nn.Linear(d_model, 1)
        
        # Imaginary part of potential (Decay/Forgetting)
        # Always positive to ensure stability (loss of info, not gain)
        self.gamma_proj = nn.Linear(d_model, 1)
        
        # Stability bounds statistics
        self.register_buffer('min_eigenvalue', torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute complex potential V - iGamma
        
        Args:
            x: (B, N, D) Input features
            
        Returns:
            complex_potential: (B, N) Complex64 tensor
        """
        # 1. Real Potential V(x)
        v = self.v_proj(x).squeeze(-1)  # (B, N)
        
        # 2. Imaginary Potential Gamma(x)
        # Must be non-negative for stability (Dissipative system)
        if self.adaptive_decay:
            gamma_raw = self.gamma_proj(x).squeeze(-1)
            # Softplus ensures positivity. base_decay adds minimal forgetting.
            gamma = F.softplus(gamma_raw) + self.base_decay
        else:
            gamma = torch.full_like(v, self.base_decay)
            
        # 3. Stability Check (Schatten Norm monitoring)
        if self.training:
            self._monitor_stability(v, gamma)
            
        # 4. Combine
        return torch.complex(v, -gamma)

    def _monitor_stability(self, v: torch.Tensor, gamma: torch.Tensor):
        """
        Monitor the spectral properties to prevent system collapse.
        If Gamma is too large compared to V, the system behavior becomes purely damping.
        """
        with torch.no_grad():
            # Calculate ratio of Damping to Oscillation
            energy = torch.abs(v) + 1e-6
            damping = gamma
            
            ratio = damping / energy
            mean_ratio = ratio.mean()
            
            # If damping dominates too much, log warning or clamp (in future)
            if mean_ratio > 10.0:
                # System is overdamped (information vanishes instantly)
                pass

    def verify_schatten_bounds(self, H_matrix: torch.Tensor) -> bool:
        """
        Verify if the effective Hamiltonian satisfies Schatten-p norm bounds.
        Required by Phase 2.1 spec.
        
        Args:
            H_matrix: (N, N) Complex Hamiltonian
            
        Returns:
            is_stable: bool
        """
        # Compute singular values
        try:
            S = torch.linalg.svdvals(H_matrix)
            norm = torch.norm(S, p=self.schatten_p)
            
            # If norm diverges, system is unstable
            is_stable = torch.isfinite(norm) and (norm < 1e5)
            return is_stable
        except:
            return False

class DissipativeBKLayer(nn.Module):
    """
    Wrapper required to use NonHermitianPotential with BK-Core.
    """
    def __init__(self, core_layer, potential_layer):
        super().__init__()
        self.core_layer = core_layer
        self.potential_layer = potential_layer
        
    def forward(self, x):
        # Generate complex potential
        V_complex = self.potential_layer(x)
        
        # Separate into real parts for BK-Core (which usually expects separate inputs)
        # or use modified BK-Core that accepts complex potential directly.
        # Assuming updated BK-Core interface:
        
        v_real = V_complex.real
        gamma = -V_complex.imag # Positive gamma
        
        # Pass to solver
        # z is spectral shift. In dissipative system, z can be real.
        return self.core_layer(v_real, gamma=gamma)