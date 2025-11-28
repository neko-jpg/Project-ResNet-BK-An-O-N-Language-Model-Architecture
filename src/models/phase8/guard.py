import torch
import torch.nn as nn

class NumericalGuard(nn.Module):
    """
    Implements Numerical Safety Guards (Task 28).
    Prevents boundary collapse (norm -> 1.0) and gradient explosions
    in hyperbolic space.
    """
    def __init__(self, max_norm: float = 0.99, grad_clip: float = 1.0):
        super().__init__()
        self.max_norm = max_norm
        self.grad_clip = grad_clip

        # Diagnostic counters
        self.collapse_count = 0
        self.clip_count = 0

    def check_boundary_collapse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Detects if any tensor in batch exceeds max_norm.
        If so, clamps it back and logs warning/count.
        """
        with torch.no_grad():
            norms = x.norm(dim=-1, keepdim=True)
            mask = norms > self.max_norm
            if mask.any():
                self.collapse_count += mask.sum().item()

        # Safe clamp using renormalization
        # We assume x is in Euclidean representation of Poincare ball
        return self._renormalize(x)

    def _renormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Projects points back within max_norm."""
        norms = x.norm(dim=-1, keepdim=True)
        # Avoid div by zero
        cond = norms > self.max_norm

        # target_norm = max_norm - epsilon
        target = self.max_norm - 1e-5

        scale = torch.where(cond, target / (norms + 1e-8), torch.ones_like(norms))
        return x * scale

    def clip_gradients(self, module: nn.Module):
        """
        Clips gradients based on Riemannian or Euclidean norm.
        """
        # Simple Euclidean clipping for now as first line of defense
        total_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), self.grad_clip)
        if total_norm > self.grad_clip:
            self.clip_count += 1
        return total_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.check_boundary_collapse(x)
