"""
Physics-Informed BK Layer
Implements Hamiltonian structure with kinetic and potential energy for physics-based learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .bk_core import BKCoreFunction, vmapped_get_diag
from .moe import SparseMoELayer


class PhysicsInformedBKLayer(nn.Module):
    """
    ResNet-BK layer with Hamiltonian structure and energy conservation.
    
    Hamiltonian: H = T + V
    - T: Kinetic energy (depends on state derivatives/momentum)
    - V: Potential energy (learned from BK-Core)
    
    Args:
        d_model: hidden dimension
        n_seq: sequence length
        num_experts: number of MoE experts
        dropout_p: dropout probability
    """
    
    def __init__(self, d_model, n_seq, num_experts=4, dropout_p=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_seq = n_seq
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # MoE for potential computation
        self.moe_ffn = SparseMoELayer(
            d_model=d_model,
            num_experts=num_experts,
            top_k=1,
            dropout_p=dropout_p
        )
        
        # Kinetic energy network: computes T from momentum (state derivatives)
        self.kinetic_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )
        
        # Potential energy network: computes V from state
        self.potential_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )
        
        # Potential projection: MoE output → scalar potential v_i
        self.potential_proj = nn.Linear(d_model, 1)
        
        # BK-Core parameters
        self.register_buffer('h0_diag', torch.full((n_seq,), -2.0))
        self.register_buffer('h0_sub', torch.full((n_seq-1,), 1.0))
        self.register_buffer('h0_super', torch.full((n_seq-1,), 1.0))
        self.z = torch.tensor(1.0j, dtype=torch.complex64)
        
        # Output projection: BK features → d_model
        self.output_proj = nn.Linear(2, d_model)
        
        # Residual scaling
        self.bk_scale = nn.Parameter(torch.tensor(1.0))
        
        # Numerical stability parameters
        self.v_max = 3.0
        self.feature_clamp = 10.0
        
        # Lagrange multiplier for energy conservation (learnable)
        self.lambda_energy = nn.Parameter(torch.tensor(0.1))
        
        # Store features for analysis
        self.output_features = None
    
    def compute_energy(self, x, x_prev=None):
        """
        Compute total energy: E = T + V
        
        Args:
            x: (B, N, D) - current state
            x_prev: (B, N, D) - previous state (for kinetic energy)
        
        Returns:
            E_total: (B,) - total energy per batch
            T_total: (B,) - kinetic energy per batch
            V_total: (B,) - potential energy per batch
        """
        B, N, D = x.shape
        
        # Potential energy: V = sum_i V_i(x_i)
        V = self.potential_mlp(x).squeeze(-1)  # (B, N)
        V_total = V.sum(dim=-1)  # (B,)
        
        # Kinetic energy: T = sum_i T_i(momentum_i)
        if x_prev is not None:
            # Momentum approximation: finite difference
            momentum = x - x_prev  # (B, N, D)
            T = self.kinetic_mlp(momentum).squeeze(-1)  # (B, N)
            T_total = T.sum(dim=-1)  # (B,)
        else:
            # No previous state: assume zero kinetic energy
            T_total = torch.zeros(B, device=x.device)
        
        # Total energy
        E_total = T_total + V_total
        
        return E_total, T_total, V_total
    
    def forward(self, x, x_prev=None, return_energy=False):
        """
        Forward pass with Hamiltonian structure.
        
        Args:
            x: (B, N, D) input tensor
            x_prev: (B, N, D) previous state (optional, for kinetic energy)
            return_energy: if True, return energy components
        
        Returns:
            output: (B, N, D) transformed tensor
            energy_dict: dict with energy components (if return_energy=True)
        """
        B, N, D = x.shape
        
        # Layer normalization
        x_norm = self.layer_norm(x)
        
        # MoE forward pass
        moe_out = self.moe_ffn(x_norm)  # (B, N, D)
        
        # Potential: v_i = potential_proj(moe_out)
        v = self.potential_proj(moe_out).squeeze(-1)  # (B, N)
        v = torch.clamp(v, -self.v_max, self.v_max)
        
        # Effective Hamiltonian: He = H0 + diag(v)
        h0_diag_batch = self.h0_diag.unsqueeze(0).expand(B, -1)
        h0_sub_batch = self.h0_sub.unsqueeze(0).expand(B, -1)
        h0_super_batch = self.h0_super.unsqueeze(0).expand(B, -1)
        
        he_diag = h0_diag_batch + v
        
        # BK-Core: compute G_ii = diag((He - zI)^-1)
        spec_features = BKCoreFunction.apply(
            he_diag, h0_super_batch, h0_sub_batch, self.z
        )  # (B, N, 2)
        
        # Clamp features for numerical stability
        spec_features = torch.clamp(spec_features, -self.feature_clamp, self.feature_clamp)
        self.output_features = spec_features
        
        # Project to d_model
        spec_out = self.output_proj(spec_features)  # (B, N, D)
        
        # Residual connection
        output = moe_out + self.bk_scale * spec_out
        
        # Compute energy if requested
        if return_energy:
            E_total, T_total, V_total = self.compute_energy(x, x_prev)
            energy_dict = {
                'E_total': E_total,
                'T_total': T_total,
                'V_total': V_total
            }
            return output, energy_dict
        
        return output
    
    def energy_conservation_loss(self, E_current, E_prev):
        """
        Energy conservation constraint: L_energy = ||E_current - E_prev||^2
        
        Args:
            E_current: (B,) current energy
            E_prev: (B,) previous energy
        
        Returns:
            loss: scalar energy conservation loss
        """
        return F.mse_loss(E_current, E_prev)
    
    def hamiltonian_loss(self, x, x_prev, target_energy=None):
        """
        Combined Hamiltonian loss: energy conservation + optional target energy.
        
        Args:
            x: (B, N, D) current state
            x_prev: (B, N, D) previous state
            target_energy: (B,) optional target energy values
        
        Returns:
            loss: scalar Hamiltonian loss
            loss_dict: dict with loss components
        """
        # Compute energies
        E_current, T_current, V_current = self.compute_energy(x, x_prev)
        E_prev, T_prev, V_prev = self.compute_energy(x_prev, None)
        
        # Energy conservation loss
        loss_conservation = self.energy_conservation_loss(E_current, E_prev)
        
        # Optional: target energy loss
        if target_energy is not None:
            loss_target = F.mse_loss(E_current, target_energy)
        else:
            loss_target = torch.tensor(0.0, device=x.device)
        
        # Combined loss with Lagrange multiplier
        loss = self.lambda_energy * loss_conservation + loss_target
        
        loss_dict = {
            'loss_conservation': loss_conservation.item(),
            'loss_target': loss_target.item() if isinstance(loss_target, torch.Tensor) else 0.0,
            'E_current': E_current.mean().item(),
            'E_prev': E_prev.mean().item(),
            'T_current': T_current.mean().item(),
            'V_current': V_current.mean().item(),
            'lambda_energy': self.lambda_energy.item()
        }
        
        return loss, loss_dict
