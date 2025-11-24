import torch
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class SuperpositionState:
    """
    Lightweight container for a single path in the quantum superposition.
    """
    token_ids: List[int] = field(default_factory=list)
    hidden_state: Optional[torch.Tensor] = None # The last hidden state
    cumulative_log_prob: float = 0.0
    cumulative_energy: float = 0.0 # Sheaf/Physics energy

    # Process Matrix State
    gamma_history: List[float] = field(default_factory=list)

    def clone(self):
        """Create a deep copy of the state for branching."""
        new_state = SuperpositionState(
            token_ids=list(self.token_ids),
            hidden_state=self.hidden_state.clone() if self.hidden_state is not None else None,
            cumulative_log_prob=self.cumulative_log_prob,
            cumulative_energy=self.cumulative_energy,
            gamma_history=list(self.gamma_history)
        )
        return new_state

    @property
    def action(self) -> float:
        """
        Compute the "Action" S for the Least Action Principle.
        S = -LogProb + beta * Energy
        """
        beta = 0.1 # Coupling constant
        return -self.cumulative_log_prob + beta * self.cumulative_energy
