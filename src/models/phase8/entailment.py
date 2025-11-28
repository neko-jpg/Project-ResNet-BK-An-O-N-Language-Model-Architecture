import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class EntailmentCone(nn.Module):
    """
    Implements Entailment Cones for hyperbolic entailment (A -> B).
    Based on the Poincare Ball model.

    A entails B if B lies within the cone defined by A.
    """
    def __init__(self, d_model: int, initial_aperture: float = 0.1):
        super().__init__()
        self.d_model = d_model
        # Aperture is learnable, constrained to be positive
        self.aperture_param = nn.Parameter(torch.tensor([initial_aperture]))
        self.softplus = nn.Softplus()

    @property
    def aperture(self) -> torch.Tensor:
        return self.softplus(self.aperture_param)

    def _mobius_add(self, x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """
        Mobius addition in Poincare ball.
        x + y = ((1 + 2c<x,y> + c|y|^2)x + (1 - c|x|^2)y) / (1 + 2c<x,y> + c^2|x|^2|y|^2)
        """
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2*c*xy + c*y2) * x + (1 - c*x2) * y
        denom = 1 + 2*c*xy + c**2 * x2 * y2
        return num / (denom.clamp(min=1e-5))

    def _poincare_dist(self, u: torch.Tensor, v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """
        Distance in Poincare ball.
        d(u, v) = 2/sqrt(c) * atanh(sqrt(c) * |-u + v|)
        using mobius addition: |-u + v|
        """
        sqrt_c = c ** 0.5
        v_minus_u = self._mobius_add(-u, v, c=c)
        norm_v_minus_u = v_minus_u.norm(dim=-1, keepdim=True)
        # Numerical stability for atanh
        arg = (sqrt_c * norm_v_minus_u).clamp(max=1.0 - 1e-5)
        dist = (2 / sqrt_c) * torch.atanh(arg)
        return dist.squeeze(-1)

    def forward(self, u: torch.Tensor, v: torch.Tensor, c: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Check if u entails v (u -> v).

        Args:
            u: Premise vectors (batch, d_model)
            v: Hypothesis vectors (batch, d_model)
            c: Curvature

        Returns:
            penalty: Entailment violation penalty (0 if u entails v)
            aperture_angle: The calculated half-aperture angle
        """
        # Calculate Euclidean norms and angle between u and v
        u_norm = u.norm(dim=-1, keepdim=True).clamp(max=0.99)
        v_norm = v.norm(dim=-1, keepdim=True).clamp(max=0.99)

        # Avoid division by zero
        dot_prod = (u * v).sum(dim=-1, keepdim=True)
        cos_theta = dot_prod / (u_norm * v_norm).clamp(min=1e-5)
        theta = torch.acos(cos_theta.clamp(-1.0 + 1e-5, 1.0 - 1e-5))

        # Calculate required half-aperture for containment
        # In Hyperbolic Entailment Cones, the condition is roughly:
        # v is in Cone(u) if dist(u, v) + dist(origin, u) approx dist(origin, v)
        # But rigorous formulation uses aperture angles.

        # Simplified cone check:
        # Check if v is "outward" from u relative to origin and within angular bounds.
        # Ideally, we learn an aperture 'K' such that angle(u, v) < K * (1 - |u|)
        # This is a heuristic approximation for Poincare cones.

        # Current logic: Check if v is inside the cone defined by u and self.aperture

        half_aperture = self.aperture

        # Penalty is positive if theta > half_aperture
        # We also enforce that |v| > |u| roughly for entailment in hierarchy (general -> specific)?
        # Actually, in Ganea et al., entailment cones are defined such that generic concepts are near origin.
        # So "mammal" (near origin) entails "dog" (far from origin).
        # So we expect |u| < |v| usually.

        # Angle violation
        angle_violation = F.relu(theta - half_aperture)

        # Order violation: enforce u is closer to root than v (if u -> v)
        # This is optional depending on definition, but standard for hyperbolic embedding hierarchies.
        order_violation = F.relu(u_norm - v_norm)

        total_penalty = angle_violation + order_violation

        return total_penalty.squeeze(-1), half_aperture

    def compute_energy(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Returns the energy (violation) for physics usage."""
        penalty, _ = self.forward(u, v)
        return penalty
