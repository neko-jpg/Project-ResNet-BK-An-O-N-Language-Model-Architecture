import torch
import torch.nn as nn
from src.kernels.phantom_core import PhantomCore

class SymplecticFusedIntegrator(nn.Module):
    """
    Symplectic Integrator using Phantom Core Fused Kernels.
    """

    def __init__(self, dt: float = 0.1, mode: str = 'verlet'):
        super().__init__()
        self.dt = dt
        self.mode = mode
        self.phantom = PhantomCore()

    def forward(self, q: torch.Tensor, p: torch.Tensor, force_func: callable) -> torch.Tensor:
        """
        Perform one step of integration.

        Returns:
            next_state: Concatenated [q, p]
        """
        if self.mode == 'euler':
            q_new, p_new = self.phantom.symplectic_fused_step_euler(q, p, force_func, self.dt)
        else:
            q_new, p_new = self.phantom.symplectic_fused_step_verlet(q, p, force_func, self.dt)

        return torch.cat([q_new, p_new], dim=-1)
