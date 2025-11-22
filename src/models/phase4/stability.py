import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any

class NumericalStability:
    """
    Utilities for ensuring numerical stability in Phase 4 models.
    Handles NaN/Inf prevention, energy conservation monitoring, and gradient clipping.
    """

    @staticmethod
    def safe_complex_division(numerator: torch.Tensor, denominator: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        """
        Performs complex division with epsilon buffering to prevent zero division.

        Args:
            numerator: Complex tensor numerator
            denominator: Complex tensor denominator
            epsilon: Small value to add to denominator magnitude

        Returns:
            Result of division
        """
        # Ensure inputs are complex or float
        if not torch.is_complex(numerator) and not torch.is_complex(denominator):
            return numerator / (denominator + epsilon)

        # For complex division: a/b = a * conj(b) / |b|^2
        denom_mag_sq = denominator.real**2 + denominator.imag**2
        denom_safe = denom_mag_sq + epsilon

        if torch.is_complex(denominator):
            conj_denom = torch.conj(denominator)
        else:
            conj_denom = denominator

        return (numerator * conj_denom) / denom_safe

    @staticmethod
    def check_energy_conservation(
        initial_energy: float,
        final_energy: float,
        threshold: float = 0.10
    ) -> Dict[str, Any]:
        """
        Checks if energy is conserved within a threshold.

        Args:
            initial_energy: Energy at start of step
            final_energy: Energy at end of step
            threshold: Maximum allowed relative drift (default 10%)

        Returns:
            Dictionary with status and drift metrics
        """
        # Avoid zero division
        if abs(initial_energy) < 1e-9:
            drift = abs(final_energy - initial_energy)
        else:
            drift = abs(final_energy - initial_energy) / abs(initial_energy)

        is_conserved = drift < threshold

        return {
            "conserved": is_conserved,
            "drift": drift,
            "initial": initial_energy,
            "final": final_energy
        }

    @staticmethod
    def clip_gradient_norm(model: nn.Module, max_norm: float = 1.0) -> float:
        """
        Clips gradient norms of model parameters.

        Args:
            model: PyTorch model
            max_norm: Maximum norm value

        Returns:
            Total norm of gradients
        """
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()

    @staticmethod
    def sanitize_tensor(tensor: torch.Tensor, replace_value: float = 0.0) -> torch.Tensor:
        """
        Replaces NaN/Inf values in a tensor.

        Args:
            tensor: Input tensor
            replace_value: Value to replace NaN/Inf with

        Returns:
            Sanitized tensor
        """
        if torch.is_complex(tensor):
            # Handle complex tensors by separating components
            real = tensor.real
            imag = tensor.imag

            if torch.isnan(real).any() or torch.isinf(real).any():
                real = torch.nan_to_num(real, nan=replace_value, posinf=replace_value, neginf=replace_value)

            if torch.isnan(imag).any() or torch.isinf(imag).any():
                imag = torch.nan_to_num(imag, nan=replace_value, posinf=replace_value, neginf=replace_value)

            return torch.complex(real, imag)

        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            return torch.nan_to_num(tensor, nan=replace_value, posinf=replace_value, neginf=replace_value)
        return tensor

    @staticmethod
    def compute_bulk_energy(field: torch.Tensor) -> float:
        """
        Computes energy of the Bulk field.
        E = integral(|field|^2)

        Args:
            field: Bulk field tensor

        Returns:
            Total energy
        """
        if torch.is_complex(field):
            return torch.sum(field.abs()**2).item()
        return torch.sum(field**2).item()
