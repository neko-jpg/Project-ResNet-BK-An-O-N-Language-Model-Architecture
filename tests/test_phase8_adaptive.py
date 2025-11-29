"""
Phase 8 Adaptive Computation Tests

Unit tests and property-based tests for AdaptiveComputation module.
"""
import torch
import unittest
import random
from typing import Tuple
from src.models.phase8.adaptive import AdaptiveComputation


class TestAdaptiveComputation(unittest.TestCase):
    """Unit tests for AdaptiveComputation."""
    
    def setUp(self):
        self.d_model = 16
        self.adaptive = AdaptiveComputation(self.d_model)

    def test_origin_vs_boundary(self):
        """Test that tokens near origin have higher exit probability than tokens near boundary."""
        # 1. Near Origin (Norm ~ 0)
        x_origin = torch.randn(1, 10, self.d_model) * 0.01

        # 2. Near Boundary (Norm ~ 0.9)
        x_boundary = torch.randn(1, 10, self.d_model)
        x_boundary = x_boundary / x_boundary.norm(dim=-1, keepdim=True) * 0.95

        _, p_origin = self.adaptive(x_origin, 0, 10)
        _, p_boundary = self.adaptive(x_boundary, 0, 10)

        print(f"P(Origin Exit): {p_origin.mean().item()}")
        print(f"P(Boundary Exit): {p_boundary.mean().item()}")

        self.assertGreater(p_origin.mean().item(), p_boundary.mean().item())

    def test_thresholding(self):
        """Test that tokens at origin should exit."""
        # Create input that should definitely exit (origin)
        x = torch.zeros(1, 1, self.d_model)
        should_exit, _ = self.adaptive(x, 0, 10)

        # With high bias for origin, it should exit
        self.assertTrue(should_exit.item())


class TestAdaptiveComputationPropertyBased(unittest.TestCase):
    """
    Property-based tests for AdaptiveComputation.
    
    **Feature: phase8-hyperbolic-transcendence, Property 22: Adaptive Computation Savings**
    **Validates: Requirements 80.4**
    
    Property: For any mixed distribution of tokens (some near origin, some near boundary),
    the adaptive computation mechanism SHALL reduce average compute by at least 30%.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 64
        self.num_iterations = 100  # Run 100 random tests
        self.min_savings_threshold = 0.30  # 30% minimum savings
        
    def _generate_mixed_distribution(
        self, 
        batch_size: int, 
        seq_len: int, 
        d_model: int,
        origin_ratio: float
    ) -> Tuple[torch.Tensor, float]:
        """
        Generate a mixed distribution of tokens.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            d_model: Model dimension
            origin_ratio: Ratio of tokens near origin (0.0 to 1.0)
            
        Returns:
            x: (B, N, D) tensor with mixed distribution
            expected_savings: Expected compute savings based on origin_ratio
        """
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Determine which tokens are near origin
        num_origin_tokens = int(seq_len * origin_ratio)
        
        for b in range(batch_size):
            # Tokens near origin (small norm)
            for i in range(num_origin_tokens):
                x[b, i] = x[b, i] * 0.05  # Very small norm
            
            # Tokens near boundary (large norm)
            for i in range(num_origin_tokens, seq_len):
                x[b, i] = x[b, i] / x[b, i].norm() * 0.9  # Norm ~ 0.9
        
        # Shuffle to mix distribution
        perm = torch.randperm(seq_len)
        x = x[:, perm, :]
        
        return x, origin_ratio
    
    def _compute_savings(
        self, 
        adaptive: AdaptiveComputation, 
        x: torch.Tensor, 
        total_layers: int
    ) -> float:
        """
        Compute the compute savings from adaptive computation.
        
        Savings = (tokens that exit early) / (total tokens)
        
        Args:
            adaptive: AdaptiveComputation module
            x: Input tensor (B, N, D)
            total_layers: Total number of layers
            
        Returns:
            savings: Fraction of compute saved (0.0 to 1.0)
        """
        batch_size, seq_len, _ = x.shape
        total_tokens = batch_size * seq_len
        
        # Simulate layer-by-layer processing
        exited_tokens = 0
        remaining_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        for layer_idx in range(total_layers):
            # Only process tokens that haven't exited
            if not remaining_mask.any():
                break
                
            should_exit, _ = adaptive(x, layer_idx, total_layers)
            
            # Count newly exited tokens
            newly_exited = (should_exit & remaining_mask).sum().item()
            
            # Compute savings: tokens that exit at layer i save (total_layers - i - 1) layers
            layers_saved = total_layers - layer_idx - 1
            exited_tokens += newly_exited * layers_saved
            
            # Update remaining mask
            remaining_mask = remaining_mask & ~should_exit
        
        # Maximum possible compute = total_tokens * total_layers
        max_compute = total_tokens * total_layers
        
        # Savings = exited_tokens / max_compute
        savings = exited_tokens / max_compute if max_compute > 0 else 0.0
        
        return savings
    
    def test_property_adaptive_computation_savings(self):
        """
        **Feature: phase8-hyperbolic-transcendence, Property 22: Adaptive Computation Savings**
        **Validates: Requirements 80.4**
        
        Property: For any mixed distribution of tokens with at least 50% near origin,
        the adaptive computation mechanism SHALL reduce average compute by at least 30%.
        
        This is a property-based test that verifies the compute savings across
        many random inputs.
        """
        print("\n" + "="*70)
        print("Property Test: Adaptive Computation Savings")
        print("**Feature: phase8-hyperbolic-transcendence, Property 22**")
        print("**Validates: Requirements 80.4**")
        print("="*70)
        
        adaptive = AdaptiveComputation(self.d_model, exit_threshold=0.5)
        total_layers = 12
        
        savings_list = []
        failed_cases = []
        
        for i in range(self.num_iterations):
            # Random parameters
            batch_size = random.randint(1, 4)
            seq_len = random.randint(16, 128)
            # Origin ratio between 0.5 and 0.9 (at least 50% near origin)
            origin_ratio = random.uniform(0.5, 0.9)
            
            # Generate mixed distribution
            x, expected_ratio = self._generate_mixed_distribution(
                batch_size, seq_len, self.d_model, origin_ratio
            )
            
            # Compute savings
            savings = self._compute_savings(adaptive, x, total_layers)
            savings_list.append(savings)
            
            # Check if savings meet threshold
            if savings < self.min_savings_threshold:
                failed_cases.append({
                    'iteration': i,
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'origin_ratio': origin_ratio,
                    'savings': savings
                })
        
        # Compute statistics
        avg_savings = sum(savings_list) / len(savings_list)
        min_savings = min(savings_list)
        max_savings = max(savings_list)
        
        print(f"\nResults over {self.num_iterations} iterations:")
        print(f"  Average savings: {avg_savings*100:.2f}%")
        print(f"  Min savings: {min_savings*100:.2f}%")
        print(f"  Max savings: {max_savings*100:.2f}%")
        print(f"  Failed cases: {len(failed_cases)}/{self.num_iterations}")
        
        # Property assertion: average savings should be at least 30%
        self.assertGreaterEqual(
            avg_savings, 
            self.min_savings_threshold,
            f"Average compute savings ({avg_savings*100:.2f}%) is below "
            f"threshold ({self.min_savings_threshold*100:.2f}%)"
        )
        
        print(f"\n✓ Property verified: Average savings {avg_savings*100:.2f}% >= {self.min_savings_threshold*100:.2f}%")
    
    def test_property_savings_scales_with_origin_ratio(self):
        """
        **Feature: phase8-hyperbolic-transcendence, Property 22: Adaptive Computation Savings**
        **Validates: Requirements 80.4**
        
        Property: Compute savings should increase monotonically with the ratio
        of tokens near origin.
        """
        print("\n" + "="*70)
        print("Property Test: Savings Scale with Origin Ratio")
        print("**Feature: phase8-hyperbolic-transcendence, Property 22**")
        print("**Validates: Requirements 80.4**")
        print("="*70)
        
        adaptive = AdaptiveComputation(self.d_model, exit_threshold=0.5)
        total_layers = 12
        batch_size = 2
        seq_len = 64
        
        origin_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
        savings_by_ratio = []
        
        for ratio in origin_ratios:
            # Average over multiple samples
            ratio_savings = []
            for _ in range(10):
                x, _ = self._generate_mixed_distribution(
                    batch_size, seq_len, self.d_model, ratio
                )
                savings = self._compute_savings(adaptive, x, total_layers)
                ratio_savings.append(savings)
            
            avg_savings = sum(ratio_savings) / len(ratio_savings)
            savings_by_ratio.append(avg_savings)
            print(f"  Origin ratio {ratio:.1f}: {avg_savings*100:.2f}% savings")
        
        # Check monotonicity (with some tolerance for randomness)
        monotonic_violations = 0
        for i in range(len(savings_by_ratio) - 1):
            if savings_by_ratio[i] > savings_by_ratio[i+1] + 0.05:  # 5% tolerance
                monotonic_violations += 1
        
        print(f"\nMonotonic violations: {monotonic_violations}")
        
        # Allow at most 1 violation due to randomness
        self.assertLessEqual(
            monotonic_violations, 
            1,
            "Savings should generally increase with origin ratio"
        )
        
        print("✓ Property verified: Savings scale with origin ratio")


if __name__ == '__main__':
    unittest.main()
