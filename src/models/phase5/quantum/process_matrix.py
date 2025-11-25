import torch
import torch.nn as nn
from typing import List, Tuple
from src.models.phase5.quantum.superposition_state import SuperpositionState

class QuantumProcessMatrix(nn.Module):
    """
    Quantum Process Matrix (Decision Engine).

    Manages the "Indefinite Causal Order" by maintaining a superposition of
    potential future paths (Beams).

    The "Collapse" of the wavefunction occurs when selecting the path
    that minimizes the Action (Entropy + Energy).
    """

    def __init__(self, beam_width: int = 4):
        super().__init__()
        self.beam_width = beam_width
        self.active_beams: List[SuperpositionState] = []

    def initialize_superposition(self, start_state: SuperpositionState):
        """Start a new process with a single seed state."""
        self.active_beams = [start_state]

    def expand_superposition(
        self,
        next_token_logits: torch.Tensor, # (Beam, Vocab)
        next_hidden_states: torch.Tensor, # (Beam, D)
        energies: torch.Tensor # (Beam,) Sheaf Energy
    ) -> List[SuperpositionState]:
        """
        Expand the current beams into new candidates and prune.

        Args:
            next_token_logits: Logits for the next token for each active beam.
            next_hidden_states: Corresponding hidden states.
            energies: Energy cost associated with the transition/state.

        Returns:
            The newly selected active beams.
        """
        candidates = []

        # Log Softmax for probabilities
        log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1) # (Beam, V)

        # For each current beam
        for b_idx, beam in enumerate(self.active_beams):
            # Get top k tokens for this beam
            top_vals, top_indices = torch.topk(log_probs[b_idx], k=self.beam_width)

            for k in range(self.beam_width):
                token_id = top_indices[k].item()
                log_p = top_vals[k].item()

                # Create new candidate state
                new_state = beam.clone()
                new_state.token_ids.append(token_id)
                new_state.hidden_state = next_hidden_states[b_idx]
                new_state.cumulative_log_prob += log_p

                # Add energy (Energy is extensive quantity here)
                # Energy depends on the transition or state.
                # Passed 'energies' is per beam.
                new_state.cumulative_energy += energies[b_idx].item()

                candidates.append(new_state)

        # Collapse / Prune
        # Sort candidates by Action (Least Action Principle)
        candidates.sort(key=lambda s: s.action)

        # Keep top K
        self.active_beams = candidates[:self.beam_width]

        return self.active_beams

    def collapse(self) -> SuperpositionState:
        """
        Force a final observation (Measurement).
        Returns the single state with minimal action.
        """
        if not self.active_beams:
            return None

        # Sort by Action
        self.active_beams.sort(key=lambda s: s.action)
        return self.active_beams[0]
