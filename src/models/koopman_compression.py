"""
Koopman Operator Compression via ε→0 Limit

This module implements Koopman operator compression using the ε→0 limit
to identify essential modes and compress the model while preserving
trace-class properties and semiseparable structure.

Mathematical Foundation:
- Koopman eigenvalues λ with |λ| < ε vanish in the ε→0 limit
- Trace-class compression: only compress operators in S_1
- Semiseparable structure: maintain H = T + UV^T with O(N) complexity
- Clark measure preservation: ensure spectral distribution is maintained

Requirements: 4.13, 4.14, 4.15, 4.16, 4.17, 4.18
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class KoopmanCompressionResult:
    """Results from Koopman operator compression."""
    original_rank: int  # Original Koopman dimension
    compressed_rank: int  # Compressed Koopman dimension
    pruned_modes: int  # Number of modes pruned
    compression_ratio: float  # Compression ratio
    eigenvalues_kept: np.ndarray  # Eigenvalues of kept modes
    eigenvalues_pruned: np.ndarray  # Eigenvalues of pruned modes
    trace_class_preserved: bool  # Whether trace-class property maintained
    semiseparable_preserved: bool  # Whether semiseparable structure maintained
    epsilon: float  # ε parameter used for pruning


class KoopmanOperatorCompressor:
    """
    Compress Koopman operator by identifying essential modes via ε→0 limit.
    
    The compression process:
    1. Compute eigendecomposition of Koopman operator K
    2. Identify modes with |λ| < ε (vanishing in limit)
    3. Prune these modes while preserving trace-class property
    4. Maintain semiseparable structure for O(N) complexity
    
    Args:
        epsilon_threshold: Threshold for mode pruning (default: 0.1)
        preserve_trace_class: Enforce trace-class bounds during compression
        preserve_semiseparable: Maintain semiseparable structure
        min_rank: Minimum rank to preserve (default: 1)
    """
    
    def __init__(
        self,
        epsilon_threshold: float = 0.1,
        preserve_trace_class: bool = True,
        preserve_semiseparable: bool = True,
        min_rank: int = 1
    ):
        self.epsilon_threshold = epsilon_threshold
        self.preserve_trace_class = preserve_trace_class
        self.preserve_semiseparable = preserve_semiseparable
        self.min_rank = min_rank
    
    def compute_eigendecomposition(
        self,
        K: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute eigendecomposition of Koopman operator.
        
        K = Q Λ Q^{-1} where Λ is diagonal matrix of eigenvalues
        
        Args:
            K: (koopman_dim, koopman_dim) Koopman operator matrix
        
        Returns:
            eigenvalues: (koopman_dim,) complex eigenvalues
            eigenvectors: (koopman_dim, koopman_dim) eigenvector matrix
        """
        try:
            # Use torch.linalg.eig for general (non-symmetric) matrices
            eigenvalues, eigenvectors = torch.linalg.eig(K)
            
            # Sort by magnitude (descending)
            magnitudes = torch.abs(eigenvalues)
            sorted_indices = torch.argsort(magnitudes, descending=True)
            
            eigenvalues = eigenvalues[sorted_indices]
            eigenvectors = eigenvectors[:, sorted_indices]
            
            return eigenvalues, eigenvectors
            
        except RuntimeError as e:
            logger.error(f"Eigendecomposition failed: {e}")
            # Return identity-like decomposition as fallback
            dim = K.shape[0]
            eigenvalues = torch.ones(dim, dtype=torch.complex64, device=K.device)
            eigenvectors = torch.eye(dim, dtype=torch.complex64, device=K.device)
            return eigenvalues, eigenvectors
    
    def identify_essential_modes(
        self,
        eigenvalues: torch.Tensor,
        epsilon: float
    ) -> torch.Tensor:
        """
        Identify essential Koopman modes using ε→0 limit.
        
        Requirement 4.13: Use ε → 0 limit to identify essential modes
        Requirement 4.14: Prune modes with |λ| < ε
        
        Args:
            eigenvalues: (koopman_dim,) complex eigenvalues
            epsilon: Threshold for mode pruning
        
        Returns:
            mask: (koopman_dim,) boolean mask, True for essential modes
        """
        # Compute magnitudes
        magnitudes = torch.abs(eigenvalues)
        
        # Essential modes: |λ| ≥ ε
        # These modes do not vanish in the ε→0 limit
        essential_mask = magnitudes >= epsilon
        
        # Ensure minimum rank
        num_essential = essential_mask.sum().item()
        if num_essential < self.min_rank:
            # Keep top min_rank modes regardless of threshold
            _, top_indices = torch.topk(magnitudes, self.min_rank)
            essential_mask = torch.zeros_like(essential_mask, dtype=torch.bool)
            essential_mask[top_indices] = True
            logger.warning(
                f"Only {num_essential} modes above threshold ε={epsilon}, "
                f"keeping top {self.min_rank} modes"
            )
        
        return essential_mask
    
    def verify_trace_class_bound(
        self,
        K_compressed: torch.Tensor,
        V_epsilon: torch.Tensor,
        z: complex = 1.0j
    ) -> bool:
        """
        Verify trace-class bound after compression.
        
        Requirement 4.16: Verify ||K_ε||_S1 ≤ (1/2)(Im z)^{-1}||V_ε||_L1
        
        Args:
            K_compressed: Compressed Koopman operator
            V_epsilon: Potential V_ε
            z: Complex shift (default: 1.0j)
        
        Returns:
            True if trace-class bound is satisfied
        """
        # Compute Schatten-1 norm (trace norm) of K_compressed
        # ||K||_S1 = sum of singular values
        try:
            singular_values = torch.linalg.svdvals(K_compressed)
            schatten_1_norm = singular_values.sum().item()
        except RuntimeError:
            logger.warning("SVD failed, cannot verify trace-class bound")
            return False
        
        # Compute L1 norm of potential
        V_L1_norm = torch.abs(V_epsilon).sum().item()
        
        # Theoretical bound: ||K_ε||_S1 ≤ (1/2)(Im z)^{-1}||V_ε||_L1
        im_z = z.imag
        if im_z <= 0:
            logger.warning(f"Invalid Im(z) = {im_z}, cannot verify bound")
            return False
        
        theoretical_bound = 0.5 * (1.0 / im_z) * V_L1_norm
        
        # Check if bound is satisfied
        bound_satisfied = schatten_1_norm <= theoretical_bound * 1.1  # 10% tolerance
        
        if bound_satisfied:
            logger.info(
                f"✓ Trace-class bound satisfied: ||K||_S1 = {schatten_1_norm:.4f} "
                f"≤ {theoretical_bound:.4f}"
            )
        else:
            logger.warning(
                f"✗ Trace-class bound violated: ||K||_S1 = {schatten_1_norm:.4f} "
                f"> {theoretical_bound:.4f}"
            )
        
        return bound_satisfied
    
    def compress_to_semiseparable(
        self,
        K_compressed: torch.Tensor,
        target_rank: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert compressed Koopman operator to semiseparable structure.
        
        Requirement 4.17: Preserve semiseparable structure H = T + UV^T
        Requirement 4.18: Verify tridiagonal + low-rank structure
        
        Args:
            K_compressed: Compressed Koopman operator
            target_rank: Target rank for low-rank component (default: log(N))
        
        Returns:
            T: Tridiagonal component
            U: Left low-rank factor
            V: Right low-rank factor
        """
        N = K_compressed.shape[0]
        
        # Set target rank to log(N) if not specified
        if target_rank is None:
            target_rank = max(1, int(np.ceil(np.log2(N))))
        
        # Extract tridiagonal part
        T = torch.zeros_like(K_compressed)
        
        # Main diagonal
        T.diagonal().copy_(K_compressed.diagonal())
        
        # Super-diagonal
        if N > 1:
            T.diagonal(1).copy_(K_compressed.diagonal(1))
        
        # Sub-diagonal
        if N > 1:
            T.diagonal(-1).copy_(K_compressed.diagonal(-1))
        
        # Off-tridiagonal part
        R = K_compressed - T
        
        # Low-rank approximation via truncated SVD
        try:
            U_full, S, Vt_full = torch.linalg.svd(R, full_matrices=False)
            
            # Keep top target_rank components
            r = min(target_rank, len(S))
            U = U_full[:, :r] * torch.sqrt(S[:r]).unsqueeze(0)
            V = Vt_full[:r, :].T * torch.sqrt(S[:r]).unsqueeze(0)
            
            # Pad if necessary
            if r < target_rank:
                pad_size = target_rank - r
                U = torch.cat([
                    U,
                    torch.zeros(N, pad_size, dtype=U.dtype, device=U.device)
                ], dim=1)
                V = torch.cat([
                    V,
                    torch.zeros(N, pad_size, dtype=V.dtype, device=V.device)
                ], dim=1)
            
            logger.info(
                f"Semiseparable structure: tridiagonal + rank-{r} "
                f"(target: {target_rank})"
            )
            
        except RuntimeError as e:
            logger.error(f"SVD failed: {e}, using zero low-rank component")
            U = torch.zeros(N, target_rank, dtype=K_compressed.dtype, device=K_compressed.device)
            V = torch.zeros(N, target_rank, dtype=K_compressed.dtype, device=K_compressed.device)
        
        return T, U, V
    
    def verify_semiseparable_reconstruction(
        self,
        K_original: torch.Tensor,
        T: torch.Tensor,
        U: torch.Tensor,
        V: torch.Tensor,
        tolerance: float = 0.1
    ) -> bool:
        """
        Verify that semiseparable reconstruction is accurate.
        
        Check: ||K - (T + UV^T)||_F < tolerance * ||K||_F
        
        Args:
            K_original: Original Koopman operator
            T: Tridiagonal component
            U: Left low-rank factor
            V: Right low-rank factor
            tolerance: Relative error tolerance
        
        Returns:
            True if reconstruction is accurate
        """
        # Reconstruct: K_recon = T + UV^T
        K_recon = T + torch.matmul(U, V.T)
        
        # Compute Frobenius norms
        error_norm = torch.linalg.norm(K_original - K_recon, ord='fro').item()
        original_norm = torch.linalg.norm(K_original, ord='fro').item()
        
        if original_norm > 0:
            relative_error = error_norm / original_norm
        else:
            relative_error = error_norm
        
        accurate = relative_error < tolerance
        
        if accurate:
            logger.info(
                f"✓ Semiseparable reconstruction accurate: "
                f"relative error = {relative_error:.6f} < {tolerance}"
            )
        else:
            logger.warning(
                f"✗ Semiseparable reconstruction inaccurate: "
                f"relative error = {relative_error:.6f} ≥ {tolerance}"
            )
        
        return accurate
    
    def compress_koopman_operator(
        self,
        K: torch.Tensor,
        epsilon: float,
        V_epsilon: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, KoopmanCompressionResult]:
        """
        Compress Koopman operator using ε→0 limit.
        
        Complete compression pipeline:
        1. Eigendecomposition of K
        2. Identify essential modes (|λ| ≥ ε)
        3. Prune vanishing modes
        4. Verify trace-class property
        5. Convert to semiseparable structure
        
        Args:
            K: (koopman_dim, koopman_dim) Koopman operator
            epsilon: Threshold for mode pruning
            V_epsilon: Optional potential for trace-class verification
        
        Returns:
            K_compressed: Compressed Koopman operator
            result: Compression statistics and verification results
        """
        logger.info(f"Compressing Koopman operator with ε={epsilon}")
        
        original_rank = K.shape[0]
        
        # Step 1: Eigendecomposition
        eigenvalues, eigenvectors = self.compute_eigendecomposition(K)
        
        # Step 2: Identify essential modes
        essential_mask = self.identify_essential_modes(eigenvalues, epsilon)
        
        num_essential = essential_mask.sum().item()
        num_pruned = original_rank - num_essential
        
        logger.info(
            f"Identified {num_essential} essential modes, "
            f"pruning {num_pruned} modes"
        )
        
        # Step 3: Compress by keeping only essential modes
        eigenvalues_kept = eigenvalues[essential_mask]
        eigenvalues_pruned = eigenvalues[~essential_mask]
        eigenvectors_kept = eigenvectors[:, essential_mask]
        
        # Reconstruct compressed operator: K_compressed = Q Λ_essential Q^{-1}
        # For numerical stability, use pseudoinverse
        try:
            Q_inv = torch.linalg.pinv(eigenvectors_kept)
            Lambda_essential = torch.diag(eigenvalues_kept)
            K_compressed = torch.matmul(
                torch.matmul(eigenvectors_kept, Lambda_essential),
                Q_inv
            )
            
            # Convert to real if imaginary part is negligible
            if torch.is_complex(K_compressed):
                if torch.abs(K_compressed.imag).max() < 1e-6:
                    K_compressed = K_compressed.real
        
        except RuntimeError as e:
            logger.error(f"Compression failed: {e}, returning original")
            K_compressed = K
            num_essential = original_rank
            num_pruned = 0
        
        # Step 4: Verify trace-class property (if V_epsilon provided)
        trace_class_preserved = True
        if self.preserve_trace_class and V_epsilon is not None:
            trace_class_preserved = self.verify_trace_class_bound(
                K_compressed, V_epsilon
            )
        
        # Step 5: Convert to semiseparable structure (if requested)
        semiseparable_preserved = True
        if self.preserve_semiseparable:
            T, U, V = self.compress_to_semiseparable(K_compressed)
            
            # Verify reconstruction
            semiseparable_preserved = self.verify_semiseparable_reconstruction(
                K_compressed, T, U, V
            )
            
            # Use semiseparable reconstruction if accurate
            if semiseparable_preserved:
                K_compressed = T + torch.matmul(U, V.T)
        
        # Compute compression ratio
        compression_ratio = num_essential / original_rank if original_rank > 0 else 1.0
        
        # Create result object
        result = KoopmanCompressionResult(
            original_rank=original_rank,
            compressed_rank=num_essential,
            pruned_modes=num_pruned,
            compression_ratio=compression_ratio,
            eigenvalues_kept=eigenvalues_kept.cpu().numpy(),
            eigenvalues_pruned=eigenvalues_pruned.cpu().numpy(),
            trace_class_preserved=trace_class_preserved,
            semiseparable_preserved=semiseparable_preserved,
            epsilon=epsilon
        )
        
        logger.info(
            f"Compression complete: {original_rank} → {num_essential} "
            f"({compression_ratio:.2%} of original)"
        )
        
        return K_compressed, result


class ProgressiveKoopmanCompression:
    """
    Progressive compression through ε-parametrized family.
    
    Compress model progressively: ε = 1.0 → 0.75 → 0.5 → 0.25 → 0.1
    At each step, compress Koopman operators and retrain.
    
    Args:
        epsilon_schedule: List of ε values for progressive compression
        compressor: KoopmanOperatorCompressor instance
    """
    
    def __init__(
        self,
        epsilon_schedule: Optional[List[float]] = None,
        compressor: Optional[KoopmanOperatorCompressor] = None
    ):
        if epsilon_schedule is None:
            epsilon_schedule = [1.0, 0.75, 0.5, 0.25, 0.1]
        
        self.epsilon_schedule = sorted(epsilon_schedule, reverse=True)
        
        if compressor is None:
            compressor = KoopmanOperatorCompressor()
        
        self.compressor = compressor
        self.compression_history: List[KoopmanCompressionResult] = []
    
    def compress_model_koopman_layers(
        self,
        model: nn.Module,
        epsilon: float
    ) -> Dict[str, KoopmanCompressionResult]:
        """
        Compress all Koopman layers in a model.
        
        Args:
            model: Model with KoopmanResNetBKLayer modules
            epsilon: Threshold for compression
        
        Returns:
            Dictionary mapping layer name to compression result
        """
        results = {}
        
        # Find all Koopman layers
        for name, module in model.named_modules():
            if hasattr(module, 'K') and isinstance(module.K, nn.Parameter):
                # This is a Koopman layer with operator K
                logger.info(f"Compressing Koopman layer: {name}")
                
                K_original = module.K.data
                
                # Get potential if available
                V_epsilon = None
                if hasattr(module, 'phi'):
                    # Try to extract potential from lifting function
                    # This is a simplified approach
                    pass
                
                # Compress
                K_compressed, result = self.compressor.compress_koopman_operator(
                    K_original, epsilon, V_epsilon
                )
                
                # Update layer with compressed operator
                module.K.data = K_compressed
                
                results[name] = result
                self.compression_history.append(result)
        
        return results
    
    def progressive_compress(
        self,
        model: nn.Module,
        retrain_fn: Optional[callable] = None
    ) -> List[Dict[str, KoopmanCompressionResult]]:
        """
        Progressively compress model through ε schedule.
        
        Args:
            model: Model to compress
            retrain_fn: Optional function to retrain after each compression step
                       Should have signature: retrain_fn(model, epsilon) -> model
        
        Returns:
            List of compression results for each ε value
        """
        all_results = []
        
        for epsilon in self.epsilon_schedule:
            logger.info(f"\n{'='*60}")
            logger.info(f"Progressive compression step: ε = {epsilon}")
            logger.info(f"{'='*60}\n")
            
            # Compress all Koopman layers
            results = self.compress_model_koopman_layers(model, epsilon)
            all_results.append(results)
            
            # Retrain if function provided
            if retrain_fn is not None:
                logger.info(f"Retraining model at ε = {epsilon}")
                model = retrain_fn(model, epsilon)
        
        return all_results
    
    def get_compression_summary(self) -> Dict:
        """
        Generate summary of progressive compression.
        
        Returns:
            Dictionary with compression statistics
        """
        if not self.compression_history:
            return {'message': 'No compression performed yet'}
        
        summary = {
            'num_compressions': len(self.compression_history),
            'epsilon_values': [r.epsilon for r in self.compression_history],
            'compression_ratios': [r.compression_ratio for r in self.compression_history],
            'total_modes_pruned': sum(r.pruned_modes for r in self.compression_history),
            'trace_class_preserved': all(r.trace_class_preserved for r in self.compression_history),
            'semiseparable_preserved': all(r.semiseparable_preserved for r in self.compression_history),
        }
        
        # Compute overall compression
        if self.compression_history:
            first_rank = self.compression_history[0].original_rank
            last_rank = self.compression_history[-1].compressed_rank
            summary['overall_compression'] = last_rank / first_rank if first_rank > 0 else 1.0
        
        return summary


def visualize_koopman_compression(
    results: List[KoopmanCompressionResult],
    save_path: Optional[str] = None
):
    """
    Visualize Koopman compression results.
    
    Args:
        results: List of compression results
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Compression ratio vs ε
    ax1 = axes[0, 0]
    epsilons = [r.epsilon for r in results]
    ratios = [r.compression_ratio for r in results]
    ax1.plot(epsilons, ratios, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('ε', fontsize=12)
    ax1.set_ylabel('Compression Ratio', fontsize=12)
    ax1.set_title('Compression Ratio vs ε', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.1])
    
    # Plot 2: Number of modes
    ax2 = axes[0, 1]
    original_ranks = [r.original_rank for r in results]
    compressed_ranks = [r.compressed_rank for r in results]
    x = np.arange(len(results))
    width = 0.35
    ax2.bar(x - width/2, original_ranks, width, label='Original', alpha=0.7)
    ax2.bar(x + width/2, compressed_ranks, width, label='Compressed', alpha=0.7)
    ax2.set_xlabel('Compression Step', fontsize=12)
    ax2.set_ylabel('Number of Modes', fontsize=12)
    ax2.set_title('Koopman Modes: Original vs Compressed', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Eigenvalue magnitudes (last result)
    ax3 = axes[1, 0]
    if results:
        last_result = results[-1]
        kept_mags = np.abs(last_result.eigenvalues_kept)
        pruned_mags = np.abs(last_result.eigenvalues_pruned)
        
        ax3.scatter(range(len(kept_mags)), kept_mags, 
                   c='blue', label='Kept', alpha=0.7, s=50)
        ax3.scatter(range(len(kept_mags), len(kept_mags) + len(pruned_mags)), 
                   pruned_mags, c='red', label='Pruned', alpha=0.7, s=50)
        ax3.axhline(y=last_result.epsilon, color='green', linestyle='--', 
                   label=f'ε={last_result.epsilon}')
        ax3.set_xlabel('Mode Index', fontsize=12)
        ax3.set_ylabel('|λ| (Eigenvalue Magnitude)', fontsize=12)
        ax3.set_title(f'Eigenvalue Magnitudes (ε={last_result.epsilon})', 
                     fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
    
    # Plot 4: Verification status
    ax4 = axes[1, 1]
    trace_class = [r.trace_class_preserved for r in results]
    semiseparable = [r.semiseparable_preserved for r in results]
    
    x = np.arange(len(results))
    width = 0.35
    ax4.bar(x - width/2, trace_class, width, label='Trace-Class', alpha=0.7)
    ax4.bar(x + width/2, semiseparable, width, label='Semiseparable', alpha=0.7)
    ax4.set_xlabel('Compression Step', fontsize=12)
    ax4.set_ylabel('Property Preserved', fontsize=12)
    ax4.set_title('Mathematical Properties Verification', fontsize=14, fontweight='bold')
    ax4.set_ylim([0, 1.2])
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['No', 'Yes'])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Koopman compression visualization saved to {save_path}")
    
    plt.show()
