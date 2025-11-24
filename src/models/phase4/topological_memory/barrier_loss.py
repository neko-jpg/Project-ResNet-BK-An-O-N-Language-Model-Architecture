import torch
import torch.nn as nn

class TopologicalBarrierLoss(nn.Module):
    """
    Topological Barrier Loss.

    Creates an "energy barrier" to stabilize high-confidence memories/states.
    If the model is confident about a prediction or state (high confidence score),
    this loss penalizes deviation from the previous state (or a target attractor),
    effectively deepening the energy well and preventing drift.

    Mathematical concept:
    L_barrier = Confidence * || State_t - State_{t-1} ||^2

    This emulates "Topological Protection" where moving out of a state requires
    overcoming a significant barrier.
    """

    def __init__(self, barrier_strength: float = 1.0):
        super().__init__()
        self.barrier_strength = barrier_strength

    def forward(
        self,
        current_state: torch.Tensor,
        previous_state: torch.Tensor,
        confidence_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate barrier loss.

        Args:
            current_state: (B, N, D) or (B, D)
            previous_state: (B, N, D) or (B, D)
            confidence_scores: (B, N) or (B,) - Higher means deeper barrier.
                               Values should be in [0, 1].

        Returns:
            loss: Scalar tensor.
        """
        # Calculate squared distance (Energy)
        # || x_t - x_{t-1} ||^2
        diff = current_state - previous_state
        dist_sq = torch.sum(diff**2, dim=-1) # (B, N) or (B,)

        # Weighted by confidence
        # High confidence -> High Penalty for moving -> Deep Well
        # Low confidence -> Low Penalty -> Shallow Well (Plasticity)
        weighted_energy = confidence_scores * dist_sq

        loss = self.barrier_strength * weighted_energy.mean()

        return loss
