import torch
import torch.nn as nn
from typing import Tuple, Optional

class AdaptiveComputation(nn.Module):
    """
    Implements Adaptive Hyperbolic Computation (Task 20).
    Decides whether to exit early or skip layers based on token complexity.

    Complexity Signal: Hyperbolic distance from origin.
    Idea: Tokens near origin (abstract/general concepts) are "harder" and need deep processing.
          Tokens near boundary (specific/leaf concepts) are "simpler" or "stable" and can exit.
          Wait, is it?
          Usually, abstract concepts need more context resolution (Entailment).
          But "Adaptive Computation Time" usually says: if confident, exit.

    Task Requirement 20.1: "Fewer layers for tokens near origin"
    Wait, let me check the requirements.md/tasks.md again.

    "Fewer layers for tokens near origin"
    Ah, perhaps the logic is: Root concepts are fundamental/axiomatic -> Easy?
    Or maybe it means "Broad concepts don't need detailed leaf processing".

    I will implement the logic as requested:
    - Distance small (Near Origin) -> High Probability of Exit (Fewer layers)
    - Distance large (Near Boundary) -> Low Probability of Exit (More layers)
    """
    def __init__(self, d_model: int, exit_threshold: float = 0.8):
        super().__init__()
        self.d_model = d_model
        self.exit_threshold = exit_threshold

        # Learnable gate to refine the raw distance signal
        self.halting_gate = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, layer_idx: int, total_layers: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            should_exit: (Batch, Seq) boolean mask (1 = exit, 0 = continue)
            halting_prob: (Batch, Seq) float probability
        """
        # 1. Compute Hyperbolic Radius (Distance from origin)
        # Assuming x is in Poincare ball
        x_norm = x.norm(dim=-1).clamp(max=0.99)
        # Dist from origin = 2 * atanh(norm)
        # But we can just use norm as the raw signal.

        # 2. Gate Computation
        # We combine the norm signal with a learned projection
        # gate_logits = self.halting_gate(x).squeeze(-1)

        # But Requirement says: "Fewer layers for tokens near origin"
        # So low norm -> High Halting Probability.
        # High norm -> Low Halting Probability.

        # Let's verify this interpretation.
        # If I have "Animal", maybe I don't need to know it's a "Golden Retriever" to process basic grammar.
        # So I stop early.

        # We model p_halt = Sigmoid( -alpha * norm + beta )
        # If norm is 0, p_halt is high.
        # If norm is 1, p_halt is low.

        # To make it robust, we use the learned gate but bias it with the norm.
        # p_halt = Sigmoid( Learned(x) - w * Norm(x) )

        gate_out = self.halting_gate(x).squeeze(-1)

        # Bias by norm (inverted)
        # We want small norm -> large prob.
        # So we subtract norm (since norm is 0..1).
        # Actually let's do: p ~ (1 - norm)

        bias = (1.0 - x_norm) * 5.0 # Strong bias

        halting_prob = self.sigmoid(gate_out + bias)

        # 3. Thresholding
        # In training, we might use the prob for loss (ACT loss).
        # In inference, we threshold.

        # Force min layers check (handled by caller usually, but here we just give signal)
        should_exit = halting_prob > self.exit_threshold

        return should_exit, halting_prob
