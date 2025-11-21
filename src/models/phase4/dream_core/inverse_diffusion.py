import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math

class DreamCore(nn.Module):
    """
    Dream Core: Generates new concepts from memory fragments via Inverse Diffusion.

    Physics-informed architecture:
    - Uses Semi-Implicit Euler integration for stability.
    - Generates dynamic potential V_dream based on memory fragments.
    - Self-organizes new concepts in the latent space.

    Args:
        d_model: Model dimension.
        n_fragments: Number of memory fragments to sample (default: 10).
        diffusion_steps: Number of diffusion steps (default: 20).
        temperature: Noise temperature (default: 0.1).
    """

    def __init__(
        self,
        d_model: int,
        n_fragments: int = 10,
        diffusion_steps: int = 20,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_fragments = n_fragments
        self.diffusion_steps = diffusion_steps
        self.temperature = temperature

        # Potential generation network
        # Computes weights for each fragment based on their content
        self.weight_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=0)
        )

    def forward(
        self,
        memory_fragments: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate a new concept (dream).

        Args:
            memory_fragments: (n_fragments, d_model)
            initial_state: (d_model,) optional

        Returns:
            new_concept: (d_model,)
            diagnostics: dict
        """
        device = memory_fragments.device

        # 1. Initialize state
        if initial_state is None:
            x = torch.randn(self.d_model, device=device)
        else:
            x = initial_state.clone()

        # 2. Compute weights for fragments (Attention-like)
        # Apply weight_net to each fragment
        # weights: (n_fragments,)
        weights = self.weight_net(memory_fragments).squeeze(-1)

        # 3. Inverse Diffusion Loop
        # Store trajectory on CPU to save VRAM
        trajectory = [x.detach().cpu()]

        dt = 1.0 / self.diffusion_steps
        dt_tensor = torch.tensor(dt, device=device)

        for step in range(self.diffusion_steps):
            # Use Gradient Checkpointing for memory efficiency during training
            if self.training and x.requires_grad:
                x = torch.utils.checkpoint.checkpoint(
                    self._diffusion_step,
                    x, memory_fragments, weights, dt_tensor, use_reentrant=False
                )
            else:
                x = self._diffusion_step(x, memory_fragments, weights, dt_tensor)

            trajectory.append(x.detach().cpu())

        # 4. Final Concept
        new_concept = x

        diagnostics = {
            'trajectory': torch.stack(trajectory),
            'weights': weights,
            'final_energy': self._compute_energy(x, memory_fragments, weights)
        }

        return new_concept, diagnostics

    def _diffusion_step(
        self,
        x: torch.Tensor,
        fragments: torch.Tensor,
        weights: torch.Tensor,
        dt: torch.Tensor
    ) -> torch.Tensor:
        """
        Single step of Semi-Implicit Euler integration.

        x_{t+1} = x_t - dt * grad(V) + sqrt(2*T*dt) * noise
        """
        # Ensure we can calculate gradients even in no_grad context (for Langevin dynamics)
        with torch.enable_grad():
            # Ensure x requires grad for potential calculation
            x_in = x.detach().requires_grad_(True)

            # Calculate Potential V(x)
            # V(x) = Sum_i w_i * ||x - f_i||^2
            # "Gravitational" pull towards weighted center of fragments

            diff = x_in.unsqueeze(0) - fragments # (n_frag, d_model)
            dist_sq = torch.sum(diff**2, dim=-1) # (n_frag,)
            potential = torch.sum(weights * dist_sq)

            # Add weak L2 regularization to keep concepts bounded
            potential = potential + 0.01 * torch.sum(x_in**2)

            # Compute gradient
            # create_graph=True is needed if inputs (weights/fragments) require grad and we want to backprop through them
            # Checkpointing passes inputs with requires_grad=True if they need it.
            # We check if we need to create graph based on inputs or if x requires grad.
            create_graph = weights.requires_grad or fragments.requires_grad or x.requires_grad
            grads = torch.autograd.grad(potential, x_in, create_graph=create_graph)[0]

        # Semi-Implicit Euler update
        # x_{t+1} = x_t - dt * grads + noise
        noise = torch.randn_like(x) * torch.sqrt(2 * self.temperature * dt)

        x_next = x - dt * grads + noise

        return x_next

    def _compute_energy(
        self,
        x: torch.Tensor,
        fragments: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            diff = x.unsqueeze(0) - fragments
            dist_sq = torch.sum(diff**2, dim=-1)
            potential = torch.sum(weights * dist_sq)
            return potential
