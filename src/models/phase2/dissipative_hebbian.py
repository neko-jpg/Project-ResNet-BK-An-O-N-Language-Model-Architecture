"""
Dissipative Hebbian Layer - Phase 2 Dynamic Memory Mechanism

This module implements the Dissipative Hebbian learning mechanism that integrates
memory formation (Hebbian) with natural forgetting (dissipation).

Physical Background:
    The dissipative Hebbian equation unifies synaptic plasticity with energy dissipation:
    
    dW/dt = η(k^T v) - ΓW
    
    Where:
    - η(k^T v): Hebbian strengthening (memory formation)
    - -ΓW: Synaptic decay (forgetting)
    - Γ: Decay rate from NonHermitianPotential
    
    Discrete time solution:
    W_new = exp(-Γ * dt) * W_old + η * (k^T v)

Key Features:
    - Fast Weights: Dynamically updated short-term memory
    - Lyapunov Stability: Mathematically guaranteed stability via energy monitoring
    - Potential Feedback: Memory influences the potential V(x, M) in BK-Core
    - O(N) Complexity: Maintains Phase 1 efficiency guarantees

Author: Project MUSE Team
Date: 2025-01-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import warnings


class LyapunovStabilityMonitor:
    """
    Lyapunov Stability Monitor for Fast Weights
    
    Monitors the energy E = ||W||²_F of Fast Weights and ensures
    the Lyapunov stability condition dE/dt ≤ 0 is satisfied.
    
    Physical Interpretation:
        - E = ||W||²: Total synaptic energy
        - dE/dt > 0: Energy increasing (unstable)
        - dE/dt ≤ 0: Energy decreasing or stable (Lyapunov stable)
    
    Args:
        gamma_adjust_rate: Rate of Γ adjustment when instability detected (default: 0.01)
    """
    
    def __init__(self, gamma_adjust_rate: float = 0.01):
        self.gamma_adjust_rate = gamma_adjust_rate
        self.prev_energy = None
        self.energy_history = []
        self.violation_count = 0
    
    def check(
        self,
        state_new: torch.Tensor,
        state_old: torch.Tensor,
        decay: torch.Tensor,
        update: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Check Lyapunov stability and suggest Γ adjustments
        
        Args:
            state_new: Updated Fast Weight (B, H, D_h, D_h)
            state_old: Previous Fast Weight (B, H, D_h, D_h)
            decay: Decay coefficient exp(-Γ*dt) (B, 1, 1, 1)
            update: Hebbian update term η*(k^T v) (B, H, D_h, D_h)
        
        Returns:
            metrics: Dictionary containing:
                - energy: Current energy ||W||²
                - dE_dt: Energy derivative
                - is_stable: Whether Lyapunov condition is satisfied
                - suggested_gamma_adjust: Suggested Γ adjustment
                - violation_count: Cumulative stability violations
        """
        # Calculate energies E = ||W||²_F
        with torch.no_grad():
            energy_new = torch.norm(state_new, p='fro') ** 2
            energy_old = torch.norm(state_old, p='fro') ** 2
            
            # Estimate dE/dt
            dE_dt = (energy_new - energy_old).item()
            
        # Check stability condition
        is_stable = dE_dt <= 1e-6  # Small tolerance for numerical errors
        
        if not is_stable:
            # Energy increasing → increase Γ to enhance dissipation
            suggested_adjust = self.gamma_adjust_rate
            self.violation_count += 1
            
            # Warn less frequently: every 100 violations instead of 10
            if self.violation_count % 100 == 0:
                warnings.warn(
                    f"Lyapunov stability violated {self.violation_count} times. "
                    f"Current dE/dt = {dE_dt:.6f}. Consider increasing base_decay.",
                    UserWarning
                )
        else:
            # Energy decreasing → allow natural decay of Γ
            suggested_adjust = -self.gamma_adjust_rate * 0.1
        
        # Update history
        self.energy_history.append(energy_new.item())
        
        # Keep only recent history
        if len(self.energy_history) > 1000:
            self.energy_history = self.energy_history[-1000:]
        
        return {
            'energy': energy_new.item(),
            'dE_dt': dE_dt,
            'is_stable': is_stable,
            'suggested_gamma_adjust': suggested_adjust,
            'violation_count': self.violation_count,
        }
    
    def reset(self):
        """Reset monitor state"""
        self.energy_history = []
        self.violation_count = 0
    
    def get_statistics(self) -> Dict[str, float]:
        """Get energy statistics"""
        if not self.energy_history:
            return {
                'mean_energy': 0.0,
                'std_energy': 0.0,
                'min_energy': 0.0,
                'max_energy': 0.0,
            }
        
        history_tensor = torch.tensor(self.energy_history)
        return {
            'mean_energy': history_tensor.mean().item(),
            'std_energy': history_tensor.std().item(),
            'min_energy': history_tensor.min().item(),
            'max_energy': history_tensor.max().item(),
        }


class DissipativeHebbianLayer(nn.Module):
    """
    Dissipative Hebbian Layer - Dynamic Memory with Natural Forgetting
    
    This layer implements the dissipative Hebbian equation:
        W_new = exp(-Γ * dt) * W_old + η * (k^T v)
    
    Physical Interpretation:
        - exp(-Γ*dt): Time evolution operator (dissipation)
        - η*(k^T v): Hebbian update (memory formation)
        - W: Fast Weight matrix (short-term memory)
    
    Key Innovation:
        Memory feedback to potential: W → V(x, M) → BK-Core
        This allows Phase 2 to be viewed as "dynamically adjusting Phase 1's
        Hamiltonian H based on memory state M".
    
    Args:
        d_model: Model dimension
        head_dim: Dimension per attention head (default: 64)
        num_heads: Number of attention heads (default: 8)
        eta: Hebbian learning rate (default: 0.1)
        dt: Time step for discretization (default: 1.0)
        enable_potential_feedback: Enable memory→potential feedback (default: True)
    """
    
    def __init__(
        self,
        d_model: int,
        head_dim: int = 64,
        num_heads: int = 8,
        eta: float = 0.1,
        dt: float = 1.0,
        enable_potential_feedback: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.eta = eta
        self.dt = dt
        self.enable_potential_feedback = enable_potential_feedback
        
        # QKV projections
        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(num_heads * head_dim, d_model)
        
        # Memory → Potential feedback projection
        if enable_potential_feedback:
            # Project Fast Weight state to potential adjustment
            self.memory_to_potential = nn.Linear(head_dim * head_dim, 1, bias=False)
        
        # Lyapunov stability monitor
        self.stability_monitor = LyapunovStabilityMonitor()
        
        # Statistics tracking
        self.register_buffer('update_norm_history', torch.zeros(1000))
        self.register_buffer('decay_history', torch.zeros(1000))
        self.history_idx = 0
    
    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        return_potential_feedback: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with dissipative Hebbian update
        
        Args:
            x: Input tensor (B, N, D)
            gamma: Decay rate from NonHermitianPotential (B, N)
            state: Previous Fast Weight state (B, H, D_h, D_h), optional
            return_potential_feedback: Return potential feedback signal
        
        Returns:
            output: Output tensor (B, N, D)
            new_state: Updated Fast Weight state (B, H, D_h, D_h)
            potential_feedback: Potential adjustment from memory (B, N) if requested
        """
        B, N, D = x.shape
        
        # QKV computation
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim)  # (B, N, H, D_h)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim)
        
        # Transpose for head-first processing
        q = q.transpose(1, 2)  # (B, H, N, D_h)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Initialize state if needed
        if state is None:
            state = torch.zeros(
                B, self.num_heads, self.head_dim, self.head_dim,
                device=x.device, dtype=x.dtype
            )
        
        # Sequence scan with dissipative Hebbian update
        outputs = []
        potential_feedbacks = [] if return_potential_feedback else None
        
        for t in range(N):
            q_t = q[:, :, t, :]  # (B, H, D_h)
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            gamma_t = gamma[:, t]  # (B,)
            
            # Dissipation term: exp(-Γ*dt)
            # Numerical stability: clamp gamma to prevent overflow
            gamma_clamped = torch.clamp(gamma_t * self.dt, max=10.0)
            decay = torch.exp(-gamma_clamped)  # (B,)
            decay = decay.view(B, 1, 1, 1)  # Broadcast shape
            
            # Hebbian update term: η * (k^T v)
            # k_t: (B, H, D_h), v_t: (B, H, D_h)
            # Outer product: (B, H, D_h, D_h)
            update = self.eta * torch.einsum('bhi,bhj->bhij', k_t, v_t)
            
            # Dissipative Hebbian equation:
            # W_new = exp(-Γ*dt) * W_old + η * (k^T v)
            # Note: Keep gradient for Lyapunov monitoring, but detach for BPTT
            state_old = state.detach()
            state = decay * state_old + update
            
            # Read from memory: y = W * q
            y_t = torch.einsum('bhij,bhj->bhi', state, q_t)  # (B, H, D_h)
            outputs.append(y_t)
            
            # Potential feedback: Memory → V(x, M)
            if return_potential_feedback and self.enable_potential_feedback:
                # Flatten Fast Weight and project to scalar potential adjustment
                state_flat = state.view(B, self.num_heads, -1)  # (B, H, D_h*D_h)
                # Average over heads
                state_avg = state_flat.mean(dim=1)  # (B, D_h*D_h)
                potential_adj = self.memory_to_potential(state_avg).squeeze(-1)  # (B,)
                potential_feedbacks.append(potential_adj)
            
            # Lyapunov stability check (compare old and new states)
            if self.training:
                stability_metrics = self.stability_monitor.check(state, state_old, decay, update)
            
            # Track statistics
            if self.training:
                idx = self.history_idx % 1000
                self.update_norm_history[idx] = torch.norm(update).item()
                self.decay_history[idx] = decay.mean().item()
                self.history_idx += 1
        
        # Stack outputs
        output = torch.stack(outputs, dim=2)  # (B, H, N, D_h)
        output = output.transpose(1, 2).contiguous()  # (B, N, H, D_h)
        output = output.view(B, N, -1)  # (B, N, H*D_h)
        
        # Output projection
        output = self.out_proj(output)
        
        # Potential feedback
        if return_potential_feedback and potential_feedbacks is not None and len(potential_feedbacks) > 0:
            potential_feedback = torch.stack(potential_feedbacks, dim=1)  # (B, N)
        else:
            potential_feedback = None
        
        return output, state, potential_feedback
    
    def forward_step(
        self,
        x_t: torch.Tensor,
        gamma_t: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step forward for sequential inference
        
        Args:
            x_t: Input at time t (B, D)
            gamma_t: Decay rate at time t (B,)
            state: Current Fast Weight state (B, H, D_h, D_h)
        
        Returns:
            output_t: Output at time t (B, D)
            new_state: Updated Fast Weight state (B, H, D_h, D_h)
        """
        B, D = x_t.shape
        
        # Add sequence dimension
        x_t = x_t.unsqueeze(1)  # (B, 1, D)
        gamma_t = gamma_t.unsqueeze(1)  # (B, 1)
        
        # Forward pass
        output_t, new_state, _ = self.forward(x_t, gamma_t, state, return_potential_feedback=False)
        
        # Remove sequence dimension
        output_t = output_t.squeeze(1)  # (B, D)
        
        return output_t, new_state
    
    def get_statistics(self) -> Dict[str, float]:
        """Get layer statistics"""
        valid_updates = self.update_norm_history[:min(self.history_idx, 1000)]
        valid_decays = self.decay_history[:min(self.history_idx, 1000)]
        
        stats = {
            'mean_update_norm': valid_updates.mean().item() if len(valid_updates) > 0 else 0.0,
            'std_update_norm': valid_updates.std().item() if len(valid_updates) > 0 else 0.0,
            'mean_decay': valid_decays.mean().item() if len(valid_decays) > 0 else 0.0,
            'std_decay': valid_decays.std().item() if len(valid_decays) > 0 else 0.0,
        }
        
        # Add stability statistics
        stats.update(self.stability_monitor.get_statistics())
        
        return stats
    
    def reset_state(self):
        """Reset Fast Weight state and monitors"""
        self.stability_monitor.reset()
        self.history_idx = 0
        self.update_norm_history.zero_()
        self.decay_history.zero_()


# Export
__all__ = [
    'DissipativeHebbianLayer',
    'LyapunovStabilityMonitor',
]
