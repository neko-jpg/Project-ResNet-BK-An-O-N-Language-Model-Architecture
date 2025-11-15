"""
Tests for magnitude-based pruning implementation.

Tests both basic magnitude pruning and iterative pruning with retraining.
"""

import torch
import torch.nn as nn
import pytest
from src.models.pruned_moe import MagnitudePruner, IterativeMagnitudePruner


class SimpleModel(nn.Module):
    """Simple model for testing pruning."""
    
    def __init__(self, d_model=64):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_model * 2, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.output_proj(x)
        return x


def test_magnitude_pruner_basic():
    """Test basic magnitude pruning."""
    print("\n=== Test: Basic Magnitude Pruning ===")
    
    # Create simple model
    model = SimpleModel(d_model=32)
    
    # Create pruner
    pruner = MagnitudePruner(threshold=0.01)
    
    # Prune model
    stats = pruner.prune_model(model, verbose=True)
    
    # Check that some weights were pruned
    assert len(stats) > 0, "Should have pruned some layers"
    
    # Check sparsity
    sparsity = pruner.get_model_sparsity(model)
    print(f"\nSparsity after pruning: {sparsity}")
    
    # Verify masks are stored
    assert len(pruner.masks) > 0, "Should have stored masks"
    
    print("✓ Basic magnitude pruning test passed")


def test_magnitude_pruner_with_sparsity():
    """Test magnitude pruning with target sparsity."""
    print("\n=== Test: Magnitude Pruning with Target Sparsity ===")
    
    model = SimpleModel(d_model=32)
    pruner = MagnitudePruner()
    
    # Prune to 50% sparsity
    target_sparsity = 0.5
    stats = pruner.prune_model(model, sparsity=target_sparsity, verbose=True)
    
    # Check achieved sparsity
    sparsity = pruner.get_model_sparsity(model)
    avg_sparsity = sum(sparsity.values()) / len(sparsity)
    
    print(f"\nTarget sparsity: {target_sparsity:.1%}")
    print(f"Achieved sparsity: {avg_sparsity:.1%}")
    
    # Should be close to target (within 5%)
    assert abs(avg_sparsity - target_sparsity) < 0.05, \
        f"Sparsity {avg_sparsity:.1%} not close to target {target_sparsity:.1%}"
    
    print("✓ Target sparsity test passed")


def test_mask_application():
    """Test that masks are correctly applied after training steps."""
    print("\n=== Test: Mask Application ===")
    
    model = SimpleModel(d_model=32)
    pruner = MagnitudePruner()
    
    # Prune model
    pruner.prune_model(model, sparsity=0.5, verbose=False)
    
    # Get initial zero count
    initial_zeros = sum((p == 0).sum().item() for p in model.parameters())
    
    # Simulate training step (modify weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(4, 32)
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    
    # Weights should have changed
    after_step_zeros = sum((p == 0).sum().item() for p in model.parameters())
    print(f"Zeros before mask: {initial_zeros}, after step: {after_step_zeros}")
    
    # Apply masks
    pruner.apply_masks(model)
    
    # Zero count should be restored
    after_mask_zeros = sum((p == 0).sum().item() for p in model.parameters())
    print(f"Zeros after mask application: {after_mask_zeros}")
    
    assert after_mask_zeros == initial_zeros, \
        "Mask application should restore zero count"
    
    print("✓ Mask application test passed")


def test_iterative_pruner_schedule():
    """Test iterative pruner sparsity schedule."""
    print("\n=== Test: Iterative Pruner Schedule ===")
    
    pruner = IterativeMagnitudePruner(
        initial_sparsity=0.2,
        final_sparsity=0.8,
        num_iterations=5
    )
    
    print(f"Sparsity schedule: {pruner.sparsity_schedule}")
    
    # Check schedule properties
    assert len(pruner.sparsity_schedule) == 5, "Should have 5 iterations"
    assert pruner.sparsity_schedule[0] >= 0.2, "First should be >= initial"
    assert pruner.sparsity_schedule[-1] <= 0.8, "Last should be <= final"
    
    # Check monotonic increase
    for i in range(len(pruner.sparsity_schedule) - 1):
        assert pruner.sparsity_schedule[i] <= pruner.sparsity_schedule[i + 1], \
            "Schedule should be monotonically increasing"
    
    print("✓ Iterative pruner schedule test passed")


def test_iterative_pruner_layer_filtering():
    """Test layer filtering in iterative pruner."""
    print("\n=== Test: Layer Filtering ===")
    
    model = SimpleModel(d_model=32)
    
    # Create pruner targeting only output_proj and fc layers
    pruner = IterativeMagnitudePruner(
        initial_sparsity=0.2,
        final_sparsity=0.5,
        num_iterations=2,
        prune_layers=['output_proj', 'fc']
    )
    
    # Get filtered layers
    filtered = pruner._filter_layer_names(model)
    print(f"Filtered layers: {filtered}")
    
    # Should include fc1, fc2, output_proj but not relu
    assert filtered is not None, "Should have filtered layers"
    assert any('fc' in name for name in filtered), "Should include fc layers"
    assert any('output_proj' in name for name in filtered), "Should include output_proj"
    
    print("✓ Layer filtering test passed")


def test_iterative_pruning_single_step():
    """Test single iteration of iterative pruning."""
    print("\n=== Test: Single Iterative Pruning Step ===")
    
    model = SimpleModel(d_model=32)
    
    pruner = IterativeMagnitudePruner(
        initial_sparsity=0.2,
        final_sparsity=0.8,
        num_iterations=3,
        prune_layers=['output_proj', 'fc']
    )
    
    # Execute one pruning step
    stats = pruner.prune_step(model, verbose=True)
    
    # Check stats
    assert 'iteration' in stats, "Should have iteration info"
    assert 'avg_sparsity' in stats, "Should have sparsity info"
    assert stats['iteration'] == 0, "First iteration should be 0"
    
    # Check that iteration counter advanced
    assert pruner.current_iteration == 1, "Should advance to iteration 1"
    
    # Check history
    assert len(pruner.history) == 1, "Should have one history entry"
    
    print(f"Average sparsity: {stats['avg_sparsity']:.2%}")
    print("✓ Single iterative pruning step test passed")


def test_pruning_summary():
    """Test pruning summary generation."""
    print("\n=== Test: Pruning Summary ===")
    
    model = SimpleModel(d_model=32)
    
    pruner = IterativeMagnitudePruner(
        initial_sparsity=0.2,
        final_sparsity=0.6,
        num_iterations=2
    )
    
    # Execute pruning steps
    pruner.prune_step(model, verbose=False)
    pruner.prune_step(model, verbose=False)
    
    # Get summary
    summary = pruner.get_pruning_summary()
    
    print(f"Summary: {summary}")
    
    # Check summary contents
    assert 'num_iterations' in summary, "Should have iteration count"
    assert 'initial_sparsity' in summary, "Should have initial sparsity"
    assert 'final_sparsity' in summary, "Should have final sparsity"
    assert summary['num_iterations'] == 2, "Should have 2 iterations"
    
    print("✓ Pruning summary test passed")


if __name__ == '__main__':
    # Run tests
    test_magnitude_pruner_basic()
    test_magnitude_pruner_with_sparsity()
    test_mask_application()
    test_iterative_pruner_schedule()
    test_iterative_pruner_layer_filtering()
    test_iterative_pruning_single_step()
    test_pruning_summary()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)
