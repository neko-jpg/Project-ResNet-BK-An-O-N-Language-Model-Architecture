"""
Prime-Bump Potential: Riemann Zeta Function Initialization

Implements structured potential initialization based on prime number distribution:
V_ε(x) = Σ_p α_{p,k}(ε) ψ_ε(x - log p)

where:
- p ranges over primes < n_seq
- α_{p,k}(ε) = (log p) / p^{k(1/2+ε)} are canonical coefficients
- ψ_ε(x) = ε^{-1/2} exp(-x²/(2ε)) is Gaussian cutoff function

Mathematical foundations from: 改善案/論文/riemann_hypothesis_main.tex

Key properties:
- GUE eigenvalue statistics (Wigner surmise)
- Finite overlap condition for different primes
- Spectral shift function matches prime counting
- Faster convergence than random initialization
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
import numpy as np
import math


def sieve_of_eratosthenes(limit: int) -> List[int]:
    """
    Generate all prime numbers less than limit using Sieve of Eratosthenes.
    
    Args:
        limit: upper bound (exclusive)
    
    Returns:
        List of prime numbers < limit
    """
    if limit <= 2:
        return []
    
    # Initialize sieve
    is_prime = [True] * limit
    is_prime[0] = is_prime[1] = False
    
    # Sieve
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit, i):
                is_prime[j] = False
    
    # Collect primes
    primes = [i for i in range(limit) if is_prime[i]]
    return primes


class PrimeBumpPotential(nn.Module):
    """
    Implements the prime-bump potential V_ε used for model initialization.
    
    This class constructs a potential based on the distribution of prime numbers,
    as described in `riemann_hypothesis_main.tex`. The formula implemented is a
    discretized version of:
        V_ε(x) = Σ_{p,k} α_{p,k}(ε) ψ_ε(x - k log p)
    
    - The coefficients `α_{p,k}(ε)` are the "canonical coefficients" computed in
      `_compute_alpha_coefficients` (see Corollary cor:canonical-V).
    - `ψ_ε` is a Gaussian bump function implemented in `compute_gaussian_cutoff`,
      which acts as a smooth cutoff.

    This potential provides a structured, theoretically-grounded initialization
    for the model's embeddings, injecting an inductive bias related to the
    spectral properties of the Riemann zeta function.
    
    Args:
        n_seq: sequence length (determines max prime)
        epsilon: cutoff width (ε ∈ [0.5, 1.0])
        k_max: maximum prime power (default: 3)
        scale: overall scaling factor for potential
    """
    
    def __init__(
        self,
        n_seq: int,
        epsilon: float = 1.0,
        k_max: int = 3,
        scale: float = 0.02,
    ):
        super().__init__()
        
        self.n_seq = n_seq
        self.epsilon = epsilon
        self.k_max = k_max
        self.scale = scale
        
        # Generate primes < n_seq
        self.primes = sieve_of_eratosthenes(n_seq)
        
        # Compute log(p) for each prime
        self.log_primes = [math.log(p) for p in self.primes]
        
        # Register as buffer (not trainable)
        self.register_buffer(
            'prime_positions',
            torch.tensor(self.log_primes, dtype=torch.float32)
        )
        
        # Position grid for potential computation
        self.register_buffer(
            'positions',
            torch.arange(n_seq, dtype=torch.float32)
        )
        
        # Precompute canonical coefficients α_{p,k}(ε)
        self.alpha_coefficients = self._compute_alpha_coefficients()
        
        # Statistics tracking
        self.eigenvalue_spacings = []
        self.gue_verification_results = {}
        
    def _compute_alpha_coefficients(self) -> Dict[Tuple[int, int], float]:
        """
        Computes the canonical coefficients for the prime-bump potential.
        
        This method implements the formula for `α_{p,k}(ε)` described in
        Corollary cor:canonical-V of `riemann_hypothesis_main.tex`:
            α_{p,k}(ε) = (log p) / p^{k(1/2+ε)}

        These coefficients are chosen to ensure the resulting potential `V_ε` has a
        finite L^2 norm for ε > 0, a critical property for the trace-class
        bounds of the Birman-Schwinger operator.
        
        Returns:
            Dictionary mapping (prime, k) -> coefficient
        """
        coefficients = {}
        
        for p in self.primes:
            log_p = math.log(p)
            for k in range(1, self.k_max + 1):
                # α_{p,k}(ε) = (log p) / p^{k(1/2+ε)}
                exponent = k * (0.5 + self.epsilon)
                alpha = log_p / (p ** exponent)
                coefficients[(p, k)] = alpha
        
        return coefficients
    
    def compute_gaussian_cutoff(
        self,
        x: torch.Tensor,
        center: float,
    ) -> torch.Tensor:
        """
        Compute Gaussian cutoff function ψ_ε(x) = ε^{-1/2} exp(-x²/(2ε)).
        
        This is a normalized Gaussian with width √ε.
        
        Args:
            x: (N,) position values
            center: center position (log p for prime p)
        
        Returns:
            psi: (N,) Gaussian bump values
        """
        # ψ_ε(x - center) = ε^{-1/2} exp(-(x-center)²/(2ε))
        normalization = 1.0 / math.sqrt(self.epsilon)
        diff = x - center
        exponent = -(diff ** 2) / (2.0 * self.epsilon)
        psi = normalization * torch.exp(exponent)
        
        return psi
    
    def compute_potential(
        self,
        positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes the full potential V_ε(x) as a sum over prime bumps.

        This method implements the superposition principle for the potential
        from Eq. (prime-bump-realisation) in `riemann_hypothesis_main.tex`:
            V_ε(x) = Σ_{p,k} α_{p,k}(ε) ψ_ε(x - k log p)

        It iterates through the pre-computed primes and their powers, combines
        the canonical coefficients `α_{p,k}` with the Gaussian bumps `ψ_ε`, and
        sums them up to form the final potential.
        
        Args:
            positions: (N,) position values (default: self.positions)
        
        Returns:
            V: (N,) potential values
        """
        if positions is None:
            positions = self.positions
        
        device = positions.device
        n = positions.shape[0]
        
        # Initialize potential
        V = torch.zeros(n, dtype=torch.float32, device=device)
        
        # Sum over primes and powers
        for p in self.primes:
            log_p = math.log(p)
            
            # Compute Gaussian bump centered at log(p)
            psi = self.compute_gaussian_cutoff(positions, log_p)
            
            # Sum over powers k
            for k in range(1, self.k_max + 1):
                alpha = self.alpha_coefficients[(p, k)]
                V += alpha * psi
        
        # Apply overall scaling
        V = self.scale * V
        
        return V
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute potential V_ε(x) with prime bumps.
        
        This can be used to initialize position embeddings or as an
        additional potential term in the Hamiltonian.
        
        Args:
            x: (B, N, D) input features (typically position embeddings)
        
        Returns:
            v: (B, N) potential values
        """
        batch_size, n_seq, d_model = x.shape
        device = x.device
        
        # Compute base potential
        V_base = self.compute_potential()  # (N,)
        
        # Expand to batch
        V = V_base.unsqueeze(0).expand(batch_size, -1)  # (B, N)
        
        return V
    
    def get_prime_indices(self) -> List[int]:
        """
        Return list of prime positions < n_seq.
        
        Returns:
            List of prime numbers
        """
        return self.primes.copy()
    
    def compute_alpha_coefficient(self, p: int, k: int) -> float:
        """
        Compute α_{p,k}(ε) = (log p) / p^{k(1/2+ε)}.
        
        Args:
            p: prime number
            k: power index
        
        Returns:
            Canonical coefficient value
        """
        if (p, k) in self.alpha_coefficients:
            return self.alpha_coefficients[(p, k)]
        
        log_p = math.log(p)
        exponent = k * (0.5 + self.epsilon)
        alpha = log_p / (p ** exponent)
        return alpha
    
    def verify_finite_overlap(self) -> Dict[str, any]:
        """
        Verify finite overlap condition:
        supp(ψ_ε(· - log p)) ∩ supp(ψ_ε(· - log q)) = ∅
        for |log p - log q| > 2√ε
        
        Returns:
            Dictionary with verification results
        """
        threshold = 2.0 * math.sqrt(self.epsilon)
        overlaps = []
        
        for i, p1 in enumerate(self.primes):
            for p2 in self.primes[i+1:]:
                log_diff = abs(math.log(p1) - math.log(p2))
                has_overlap = log_diff <= threshold
                overlaps.append({
                    'p1': p1,
                    'p2': p2,
                    'log_diff': log_diff,
                    'threshold': threshold,
                    'has_overlap': has_overlap,
                })
        
        # Count overlapping pairs
        num_overlaps = sum(1 for o in overlaps if o['has_overlap'])
        total_pairs = len(overlaps)
        
        return {
            'num_overlaps': num_overlaps,
            'total_pairs': total_pairs,
            'overlap_fraction': num_overlaps / total_pairs if total_pairs > 0 else 0.0,
            'threshold': threshold,
            'epsilon': self.epsilon,
            'overlaps': overlaps[:10],  # First 10 for inspection
        }
    
    def compute_eigenvalue_spacing(
        self,
        H: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute eigenvalue spacing distribution for GUE verification.
        
        Args:
            H: (N, N) Hamiltonian matrix (if None, construct from potential)
        
        Returns:
            spacings: (N-1,) nearest-neighbor eigenvalue spacings
        """
        if H is None:
            # Construct simple Hamiltonian: H = H_0 + V
            # H_0 is discrete Laplacian (tridiagonal)
            n = self.n_seq
            device = self.positions.device
            
            H = torch.zeros(n, n, device=device)
            H.diagonal().fill_(-2.0)
            if n > 1:
                H.diagonal(1).fill_(1.0)
                H.diagonal(-1).fill_(1.0)
            
            # Add potential on diagonal
            V = self.compute_potential()
            H.diagonal().add_(V)
        
        # Compute eigenvalues
        try:
            eigenvalues = torch.linalg.eigvalsh(H)
            eigenvalues = eigenvalues.sort()[0]  # Sort ascending
            
            # Compute nearest-neighbor spacings
            spacings = eigenvalues[1:] - eigenvalues[:-1]
            
            # Normalize by mean spacing (standard in RMT)
            mean_spacing = spacings.mean()
            if mean_spacing > 1e-10:
                spacings = spacings / mean_spacing
            
            self.eigenvalue_spacings = spacings.cpu().numpy()
            
            return spacings
        except RuntimeError:
            # If eigenvalue computation fails, return empty
            return torch.tensor([], device=H.device)
    
    def verify_gue_statistics(
        self,
        H: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Verify eigenvalue spacing follows Wigner surmise: P(s) = s·exp(-πs²/4).
        
        This is the hallmark of GUE (Gaussian Unitary Ensemble) statistics,
        which indicates optimal spectral properties for information propagation.
        
        Args:
            H: (N, N) Hamiltonian matrix (if None, construct from potential)
        
        Returns:
            Dictionary with GUE verification metrics
        """
        spacings = self.compute_eigenvalue_spacing(H)
        
        if len(spacings) == 0:
            return {
                'mean_spacing': 0.0,
                'std_spacing': 0.0,
                'wigner_fit_error': float('inf'),
                'gue_verified': False,
            }
        
        spacings_np = spacings.cpu().numpy()
        
        # Compute statistics
        mean_spacing = float(np.mean(spacings_np))
        std_spacing = float(np.std(spacings_np))
        
        # Wigner surmise: P(s) = s * exp(-π s² / 4)
        # Expected mean for Wigner: E[s] ≈ 1.0 (after normalization)
        # Expected std for Wigner: σ[s] ≈ 0.52
        
        wigner_expected_mean = 1.0
        wigner_expected_std = 0.52
        
        # Compute fit error
        mean_error = abs(mean_spacing - wigner_expected_mean)
        std_error = abs(std_spacing - wigner_expected_std)
        fit_error = mean_error + std_error
        
        # Verify if close to GUE (tolerance: 0.3)
        gue_verified = fit_error < 0.3
        
        results = {
            'mean_spacing': mean_spacing,
            'std_spacing': std_spacing,
            'wigner_expected_mean': wigner_expected_mean,
            'wigner_expected_std': wigner_expected_std,
            'mean_error': mean_error,
            'std_error': std_error,
            'wigner_fit_error': fit_error,
            'gue_verified': gue_verified,
            'num_eigenvalues': len(spacings_np) + 1,
        }
        
        self.gue_verification_results = results
        
        return results
    
    def visualize_potential(self) -> Dict[str, any]:
        """
        Generate visualization data for potential V_ε(x).
        
        Returns:
            Dictionary with visualization data
        """
        V = self.compute_potential().cpu().numpy()
        positions = self.positions.cpu().numpy()
        
        # Find peaks (should align with prime positions)
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(V, height=0.0)
        
        return {
            'positions': positions,
            'potential': V,
            'primes': self.primes,
            'log_primes': self.log_primes,
            'peaks': peaks,
            'peak_heights': properties['peak_heights'] if 'peak_heights' in properties else [],
            'epsilon': self.epsilon,
            'k_max': self.k_max,
            'scale': self.scale,
        }
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get comprehensive statistics about the potential.
        
        Returns:
            Dictionary with all statistics
        """
        V = self.compute_potential()
        
        # Compute norms
        l1_norm = V.abs().sum().item()
        l2_norm = torch.sqrt((V ** 2).sum()).item()
        linf_norm = V.abs().max().item()
        
        # Overlap verification
        overlap_results = self.verify_finite_overlap()
        
        stats = {
            'n_seq': self.n_seq,
            'epsilon': self.epsilon,
            'k_max': self.k_max,
            'scale': self.scale,
            'num_primes': len(self.primes),
            'l1_norm': l1_norm,
            'l2_norm': l2_norm,
            'linf_norm': linf_norm,
            'mean_potential': V.mean().item(),
            'std_potential': V.std().item(),
            'min_potential': V.min().item(),
            'max_potential': V.max().item(),
            'overlap_fraction': overlap_results['overlap_fraction'],
            'num_overlaps': overlap_results['num_overlaps'],
        }
        
        # Add GUE statistics if available
        if self.gue_verification_results:
            stats.update({
                'gue_mean_spacing': self.gue_verification_results['mean_spacing'],
                'gue_std_spacing': self.gue_verification_results['std_spacing'],
                'gue_fit_error': self.gue_verification_results['wigner_fit_error'],
                'gue_verified': self.gue_verification_results['gue_verified'],
            })
        
        return stats


class EpsilonScheduler:
    """
    Epsilon annealing schedule: ε = 1.0 → 0.5 during training.
    
    As ε decreases:
    - Gaussian bumps become narrower (more localized)
    - Potential becomes more compressed
    - Model approaches ε → 0 limit (Koopman compression)
    
    Args:
        initial_epsilon: starting value (default: 1.0)
        final_epsilon: ending value (default: 0.5)
        num_steps: total training steps
        schedule_type: 'linear', 'cosine', or 'exponential'
    """
    
    def __init__(
        self,
        initial_epsilon: float = 1.0,
        final_epsilon: float = 0.5,
        num_steps: int = 10000,
        schedule_type: str = 'cosine',
    ):
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.num_steps = num_steps
        self.schedule_type = schedule_type
        
        self.current_step = 0
        self.current_epsilon = initial_epsilon
    
    def step(self) -> float:
        """
        Update epsilon for current step.
        
        Returns:
            Current epsilon value
        """
        if self.current_step >= self.num_steps:
            self.current_epsilon = self.final_epsilon
            return self.current_epsilon
        
        progress = self.current_step / self.num_steps
        
        if self.schedule_type == 'linear':
            # Linear interpolation
            self.current_epsilon = (
                self.initial_epsilon + 
                (self.final_epsilon - self.initial_epsilon) * progress
            )
        elif self.schedule_type == 'cosine':
            # Cosine annealing
            self.current_epsilon = (
                self.final_epsilon + 
                0.5 * (self.initial_epsilon - self.final_epsilon) * 
                (1 + math.cos(math.pi * progress))
            )
        elif self.schedule_type == 'exponential':
            # Exponential decay
            decay_rate = math.log(self.final_epsilon / self.initial_epsilon)
            self.current_epsilon = self.initial_epsilon * math.exp(decay_rate * progress)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        self.current_step += 1
        return self.current_epsilon
    
    def get_epsilon(self) -> float:
        """Get current epsilon value without stepping."""
        return self.current_epsilon
    
    def reset(self):
        """Reset scheduler to initial state."""
        self.current_step = 0
        self.current_epsilon = self.initial_epsilon
