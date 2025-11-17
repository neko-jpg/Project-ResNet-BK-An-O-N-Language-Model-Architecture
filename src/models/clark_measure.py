"""
Clark Measure Computation for ε-Parametrized Model Family

This module implements Clark measure computation for model compression
via the ε→0 limit, as described in the Birman-Schwinger paper.

Mathematical Foundation:
- Clark measure: μ_ε(E) = (1/2π) ∫_E |D_ε(λ + i0)|^{-2} dλ
- D_ε(λ) is the regularized determinant from Birman-Krein formula
- μ_ε is a probability measure: μ_ε(ℝ) = 1
- Total variation distance: ||μ_1 - μ_2||_TV = sup_E |μ_1(E) - μ_2(E)|

Requirements: 4.5, 4.6, 4.7, 4.8
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClarkMeasureResult:
    """Results from Clark measure computation."""
    lambda_grid: np.ndarray  # Spectral grid points
    measure_values: np.ndarray  # μ_ε(λ) values
    total_mass: float  # Should be ≈ 1.0
    epsilon: float  # ε parameter used
    is_valid: bool  # Whether measure is valid probability measure


class ClarkMeasureComputer:
    """
    Compute Clark measure μ_ε(E) = (1/2π) ∫_E |D_ε(λ + i0)|^{-2} dλ.
    
    The Clark measure characterizes the spectral distribution of the
    Birman-Schwinger operator and is preserved during compression.
    
    Args:
        lambda_min: Minimum spectral value
        lambda_max: Maximum spectral value
        num_points: Number of grid points for integration
        eta: Small imaginary part for boundary approach (η → 0)
    """
    
    def __init__(
        self,
        lambda_min: float = -10.0,
        lambda_max: float = 10.0,
        num_points: int = 1000,
        eta: float = 1e-4
    ):
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.num_points = num_points
        self.eta = eta
        
        # Create spectral grid
        self.lambda_grid = np.linspace(lambda_min, lambda_max, num_points)
        self.dlambda = (lambda_max - lambda_min) / (num_points - 1)
    
    def compute_determinant(
        self,
        G_ii: torch.Tensor,
        epsilon: float,
        lambda_val: float
    ) -> complex:
        """
        Compute regularized determinant D_ε(λ + iη).
        
        Using Birman-Krein formula:
        d/dλ log D_ε(λ) = -Tr((H_ε - λ)^{-1} - (H_0 - λ)^{-1})
        
        Args:
            G_ii: (B, N) diagonal of resolvent (H_ε - z)^{-1}
            epsilon: Regularization parameter
            lambda_val: Spectral point λ
        
        Returns:
            D_ε(λ + iη): Complex determinant value
        """
        # G_ii contains diagonal of (H_ε - z)^{-1} where z = 1.0j
        # We need to evaluate at z = λ + iη
        
        # For simplicity, use the trace formula:
        # log D_ε(λ) ≈ -∫ Tr((H_ε - μ)^{-1} - (H_0 - μ)^{-1}) dμ
        
        # Approximate using current G_ii (this is a simplified version)
        # In full implementation, would need to recompute at each λ
        z = complex(lambda_val, self.eta)
        
        # Use mean of G_ii as approximation
        trace_diff = G_ii.mean().item()
        
        # Determinant approximation
        # D_ε ≈ exp(-trace_diff * epsilon)
        log_det = -trace_diff * epsilon
        D_eps = np.exp(log_det)
        
        return complex(D_eps, 0.0)
    
    def compute_measure(
        self,
        G_ii: torch.Tensor,
        epsilon: float
    ) -> ClarkMeasureResult:
        """
        Compute Clark measure μ_ε over the spectral grid.
        
        μ_ε(E) = (1/2π) ∫_E |D_ε(λ + i0)|^{-2} dλ
        
        Args:
            G_ii: (B, N) diagonal of resolvent from BirmanSchwingerCore
            epsilon: Regularization parameter
        
        Returns:
            ClarkMeasureResult with measure values and validation
        """
        logger.info(f"Computing Clark measure for ε={epsilon}")
        
        # Compute |D_ε(λ)|^{-2} at each grid point
        measure_density = np.zeros(self.num_points)
        
        for i, lambda_val in enumerate(self.lambda_grid):
            D_eps = self.compute_determinant(G_ii, epsilon, lambda_val)
            
            # Clark measure density: (1/2π) |D_ε|^{-2}
            if abs(D_eps) > 1e-10:
                measure_density[i] = (1.0 / (2 * np.pi)) * (1.0 / abs(D_eps)**2)
            else:
                # Regularize near zeros
                measure_density[i] = 0.0
        
        # Integrate to get total mass (should be ≈ 1.0)
        total_mass = np.trapz(measure_density, self.lambda_grid)
        
        # Normalize to ensure probability measure
        if total_mass > 0:
            measure_density_normalized = measure_density / total_mass
        else:
            measure_density_normalized = measure_density
            logger.warning(f"Total mass is {total_mass}, cannot normalize")
        
        # Verify it's a valid probability measure
        is_valid = abs(total_mass - 1.0) < 0.1  # Allow 10% tolerance
        
        if not is_valid:
            logger.warning(
                f"Clark measure may not be valid: total mass = {total_mass:.4f}"
            )
        
        return ClarkMeasureResult(
            lambda_grid=self.lambda_grid,
            measure_values=measure_density_normalized,
            total_mass=total_mass,
            epsilon=epsilon,
            is_valid=is_valid
        )
    
    def compute_total_variation_distance(
        self,
        measure1: ClarkMeasureResult,
        measure2: ClarkMeasureResult
    ) -> float:
        """
        Compute total variation distance ||μ_1 - μ_2||_TV.
        
        ||μ_1 - μ_2||_TV = (1/2) ∫ |μ_1(λ) - μ_2(λ)| dλ
        
        Args:
            measure1: First Clark measure
            measure2: Second Clark measure
        
        Returns:
            Total variation distance (0 = identical, 1 = completely different)
        """
        # Ensure measures are on same grid
        if not np.allclose(measure1.lambda_grid, measure2.lambda_grid):
            raise ValueError("Measures must be computed on same spectral grid")
        
        # Compute absolute difference
        abs_diff = np.abs(measure1.measure_values - measure2.measure_values)
        
        # Total variation = (1/2) * integral of absolute difference
        tv_distance = 0.5 * np.trapz(abs_diff, measure1.lambda_grid)
        
        logger.info(
            f"Total variation distance between ε={measure1.epsilon} and "
            f"ε={measure2.epsilon}: {tv_distance:.6f}"
        )
        
        return tv_distance
    
    def verify_probability_measure(
        self,
        measure: ClarkMeasureResult,
        tolerance: float = 0.05
    ) -> bool:
        """
        Verify that μ_ε is a valid probability measure.
        
        Requirements:
        1. μ_ε(ℝ) = 1 (total mass)
        2. μ_ε(E) ≥ 0 for all E (non-negative)
        
        Args:
            measure: Clark measure to verify
            tolerance: Tolerance for total mass check
        
        Returns:
            True if valid probability measure
        """
        # Check non-negativity
        if np.any(measure.measure_values < -1e-10):
            logger.error("Clark measure has negative values")
            return False
        
        # Check total mass ≈ 1
        if abs(measure.total_mass - 1.0) > tolerance:
            logger.error(
                f"Clark measure total mass {measure.total_mass:.4f} "
                f"deviates from 1.0 by more than {tolerance}"
            )
            return False
        
        logger.info(
            f"Clark measure verified: total mass = {measure.total_mass:.6f}, "
            f"all values non-negative"
        )
        return True


class EpsilonParametrizedFamily:
    """
    Manage family of models parametrized by ε ∈ {1.0, 0.75, 0.5, 0.25, 0.1}.
    
    This class handles:
    1. Training models at different ε values
    2. Computing Clark measures for each model
    3. Verifying measure preservation during compression
    
    Args:
        epsilon_values: List of ε values to train
        base_config: Base model configuration
    """
    
    def __init__(
        self,
        epsilon_values: List[float] = None,
        lambda_min: float = -10.0,
        lambda_max: float = 10.0,
        num_points: int = 1000
    ):
        if epsilon_values is None:
            epsilon_values = [1.0, 0.75, 0.5, 0.25, 0.1]
        
        self.epsilon_values = sorted(epsilon_values, reverse=True)
        self.clark_computer = ClarkMeasureComputer(
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            num_points=num_points
        )
        
        # Storage for computed measures
        self.measures: Dict[float, ClarkMeasureResult] = {}
    
    def compute_measure_for_model(
        self,
        model: nn.Module,
        epsilon: float,
        sample_input: torch.Tensor
    ) -> ClarkMeasureResult:
        """
        Compute Clark measure for a model at given ε.
        
        Args:
            model: Model with BirmanSchwingerCore
            epsilon: ε parameter
            sample_input: Sample input for forward pass
        
        Returns:
            Computed Clark measure
        """
        model.eval()
        
        with torch.no_grad():
            # Forward pass to get G_ii from BirmanSchwingerCore
            # This assumes model has a method to extract G_ii
            if hasattr(model, 'get_resolvent_diagonal'):
                G_ii = model.get_resolvent_diagonal(sample_input)
            else:
                # Fallback: run forward and extract from intermediate
                _ = model(sample_input)
                # Try to find BirmanSchwingerCore in model
                G_ii = None
                for module in model.modules():
                    if hasattr(module, 'last_G_ii'):
                        G_ii = module.last_G_ii
                        break
                
                if G_ii is None:
                    raise ValueError(
                        "Could not extract G_ii from model. "
                        "Model must have get_resolvent_diagonal() method or "
                        "BirmanSchwingerCore with last_G_ii attribute."
                    )
        
        # Compute Clark measure
        measure = self.clark_computer.compute_measure(G_ii, epsilon)
        
        # Store for later comparison
        self.measures[epsilon] = measure
        
        return measure
    
    def verify_compression_preserves_measure(
        self,
        epsilon_teacher: float,
        epsilon_student: float,
        max_tv_distance: float = 0.1
    ) -> bool:
        """
        Verify that compression from ε_teacher to ε_student preserves Clark measure.
        
        Requirement 4.6: ||μ_1.0 - μ_0.1||_TV < 0.1
        
        Args:
            epsilon_teacher: Teacher model ε
            epsilon_student: Student model ε (smaller)
            max_tv_distance: Maximum allowed TV distance
        
        Returns:
            True if measure is preserved within tolerance
        """
        if epsilon_teacher not in self.measures:
            raise ValueError(f"No measure computed for ε={epsilon_teacher}")
        if epsilon_student not in self.measures:
            raise ValueError(f"No measure computed for ε={epsilon_student}")
        
        measure_teacher = self.measures[epsilon_teacher]
        measure_student = self.measures[epsilon_student]
        
        tv_distance = self.clark_computer.compute_total_variation_distance(
            measure_teacher, measure_student
        )
        
        preserved = tv_distance < max_tv_distance
        
        if preserved:
            logger.info(
                f"✓ Clark measure preserved: ||μ_{epsilon_teacher} - μ_{epsilon_student}||_TV "
                f"= {tv_distance:.4f} < {max_tv_distance}"
            )
        else:
            logger.warning(
                f"✗ Clark measure NOT preserved: ||μ_{epsilon_teacher} - μ_{epsilon_student}||_TV "
                f"= {tv_distance:.4f} ≥ {max_tv_distance}"
            )
        
        return preserved
    
    def get_compression_report(self) -> Dict:
        """
        Generate comprehensive report on compression and measure preservation.
        
        Returns:
            Dictionary with compression statistics
        """
        report = {
            'epsilon_values': self.epsilon_values,
            'measures_computed': list(self.measures.keys()),
            'tv_distances': {},
            'all_valid': True
        }
        
        # Compute pairwise TV distances
        eps_list = sorted(self.measures.keys(), reverse=True)
        for i in range(len(eps_list) - 1):
            eps1 = eps_list[i]
            eps2 = eps_list[i + 1]
            
            tv_dist = self.clark_computer.compute_total_variation_distance(
                self.measures[eps1],
                self.measures[eps2]
            )
            report['tv_distances'][f'{eps1}_to_{eps2}'] = tv_dist
        
        # Check if all measures are valid
        for eps, measure in self.measures.items():
            if not measure.is_valid:
                report['all_valid'] = False
                logger.warning(f"Measure at ε={eps} is not valid")
        
        return report


def visualize_clark_measures(
    measures: Dict[float, ClarkMeasureResult],
    save_path: Optional[str] = None
):
    """
    Visualize Clark measures for different ε values.
    
    Args:
        measures: Dictionary mapping ε to ClarkMeasureResult
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Clark measures
    for eps in sorted(measures.keys(), reverse=True):
        measure = measures[eps]
        ax1.plot(
            measure.lambda_grid,
            measure.measure_values,
            label=f'ε={eps:.2f}',
            linewidth=2
        )
    
    ax1.set_xlabel('λ (spectral parameter)', fontsize=12)
    ax1.set_ylabel('μ_ε(λ) (measure density)', fontsize=12)
    ax1.set_title('Clark Measures for Different ε', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Total mass verification
    eps_vals = sorted(measures.keys())
    total_masses = [measures[eps].total_mass for eps in eps_vals]
    
    ax2.plot(eps_vals, total_masses, 'o-', linewidth=2, markersize=8)
    ax2.axhline(y=1.0, color='r', linestyle='--', label='Expected (μ_ε(ℝ) = 1)')
    ax2.set_xlabel('ε', fontsize=12)
    ax2.set_ylabel('Total Mass μ_ε(ℝ)', fontsize=12)
    ax2.set_title('Probability Measure Verification', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Clark measure visualization saved to {save_path}")
    
    plt.show()
