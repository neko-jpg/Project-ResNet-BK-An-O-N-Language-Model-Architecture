import torch
import torch.nn as nn
from typing import Dict, Any

class ConceptTemperature:
    """
    Calculates the 'Temperature' (Novelty/Difficulty) of input data.

    Temperature T ~ Entropy + GradientNorm
    """

    def compute(self, logits: torch.Tensor, hidden_state: torch.Tensor) -> float:
        """
        Args:
            logits: (B, N, V)
            hidden_state: (B, N, D)
        """
        # Approximate entropy of predictions
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()

        # Norm of hidden state acts as a proxy for signal energy
        energy = hidden_state.norm(dim=-1).mean()

        # Heuristic temperature
        # High entropy = Confused/Novel = High Temp
        # High energy = Strong Signal
        temp = entropy.item() * (1.0 + 0.1 * energy.item())
        return temp

class FatigueModel(nn.Module):
    """
    Tracks the AI's internal 'Energy' and 'Fatigue'.

    Fatigue accumulates with:
    - High Temperature Data (Hard work)
    - High Condition Number (Instability/Pain)

    Fatigue dissipates with:
    - Low Temperature Data (Rest)
    - Sleep (Reset)
    """

    def __init__(self, max_energy: float = 100.0, recovery_rate: float = 5.0):
        super().__init__()
        self.register_buffer('energy_level', torch.tensor(max_energy))
        self.max_energy = max_energy
        self.recovery_rate = recovery_rate

    def update(self, temp: float, pain: float):
        # Consumption = temp + pain * factor
        consumption = temp + max(0, pain) * 2.0

        current = self.energy_level.item()
        new_level = current - consumption

        # Auto-recovery if load is low
        if consumption < 2.0:
            new_level += self.recovery_rate

        self.energy_level.fill_(min(self.max_energy, max(0.0, new_level)))

    def get_status(self) -> str:
        level = self.energy_level.item()
        if level < 20.0:
            return "exhausted"
        elif level < 50.0:
            return "tired"
        else:
            return "fresh"

class GrowthEstimator:
    """
    Estimates the learning growth rate d(Intelligence)/dt.
    """
    def __init__(self):
        self.loss_history = []

    def update(self, loss: float):
        self.loss_history.append(loss)
        if len(self.loss_history) > 100:
            self.loss_history.pop(0)

    def get_growth_rate(self) -> float:
        if len(self.loss_history) < 2:
            return 0.0

        # Simple derivative: reduction in loss
        start = sum(self.loss_history[:10]) / 10
        end = sum(self.loss_history[-10:]) / 10
        return max(0.0, start - end)

class PacingController(nn.Module):
    """
    Master controller for Curriculum Pacing.
    """

    def __init__(self):
        super().__init__()
        self.temp_calc = ConceptTemperature()
        self.fatigue_model = FatigueModel()
        self.growth_est = GrowthEstimator()

    def step(self, logits: torch.Tensor, hidden_state: torch.Tensor, pain_signal: float) -> Dict[str, Any]:
        """
        Update internal state and return recommended mode.
        """
        # 1. Calculate Data Temp
        temp = self.temp_calc.compute(logits, hidden_state)

        # 2. Update Fatigue
        self.fatigue_model.update(temp, pain_signal)
        status = self.fatigue_model.get_status()

        # 3. Decide Mode
        mode = {
            'symplectic_mode': 'euler' if status in ['exhausted', 'tired'] else 'verlet',
            'precision': 'low' if status == 'exhausted' else 'mixed',
            'status': status,
            'temperature': temp,
            'energy': self.fatigue_model.energy_level.item()
        }

        return mode
