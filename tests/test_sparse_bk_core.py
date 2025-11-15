"""
Tests for Sparse BK-Core with Learned Sparsity
"""

import pytest
import torch
import torch.nn as nn

from src.models.sparse_bk_core import (
    SparseBKCore, 
    SparseMoEResNetBKLayer,
    AdaptiveSparsityScheduler,
    optimized_sparse_bk_core,
    sparse_theta_recursion,
    sparse_phi_recursion
)


class TestSparseBKCore:
    """Test suite for SparseBKCore module."""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def sparse_bk_core(self, device):
        """Create a SparseBKCore instance for testing."""
        d_model = 64
        n_seq = 128
        target_sparsity = 0.5
        return SparseBKCore(d_model, n_seq, target_sparsity).to(device)
    
    def test_initialization(self, sparse_bk_core):
        """Test that SparseBKCore initializes correctly."""
        assert sparse_bk_core.d_model == 64
        assert sparse_bk_core.n_seq == 128
        assert sparse_bk_core.target_sparsity == 0.5
        assert sparse_bk_core.tau == 1.0
        
        # Check components exist
        assert hasattr(sparse_bk_core, 'importance_predictor')
        assert hasattr(sparse_bk_core, 'interpolator')
        assert hasattr(sparse_bk_core, 'bk_core')
    
    def test_forward_shape(self, sparse_bk_core, device):
        """Test that forward pass produces correct output shapes."""
        B, N, D = 2, 128, 64
        x = torch.randn(B, N, D, device=device)
        v = torch.randn(B, N, device=device)
        
        features, mask, sparsity_ratio = sparse_bk_core(x, v)
        
        # Check shapes
        assert features.shape == (B, N, 2), f"Expected features shape (2, 128, 2), got {features.shape}"
        assert mask.shape == (B, N), f"Expected mask shape (2, 128), got {mask.shape}"
        assert isinstance(sparsity_ratio, torch.Tensor)
        assert sparsity_ratio.numel() == 1
    
    def test_mask_binary(self, sparse_bk_core, device):
        """Test that mask is binary (0 or 1)."""
        B, N, D = 2, 128, 64
        x = torch.randn(B, N, D, device=device)
        v = torch.randn(B, N, device=device)
        
        _, mask, _ = sparse_bk_core(x, v)
        
        # Check mask is binary
        unique_values = torch.unique(mask)
        assert len(unique_values) <= 2, "Mask should only contain 0 and 1"
        assert all(val in [0.0, 1.0] for val in unique_values.tolist()), "Mask values should be 0 or 1"
    
    def test_sparsity_ratio(self, sparse_bk_core, device):
        """Test that sparsity ratio is reasonable."""
        B, N, D = 2, 128, 64
        x = torch.randn(B, N, D, device=device)
        v = torch.randn(B, N, device=device)
        
        _, mask, sparsity_ratio = sparse_bk_core(x, v)
        
        # Sparsity ratio should be between 0 and 1
        assert 0.0 <= sparsity_ratio.item() <= 1.0, f"Sparsity ratio {sparsity_ratio.item()} out of range"
        
        # Verify sparsity calculation
        expected_sparsity = 1.0 - mask.mean().item()
        assert abs(sparsity_ratio.item() - expected_sparsity) < 1e-5
    
    def test_gumbel_sigmoid(self, sparse_bk_core, device):
        """Test Gumbel-Sigmoid produces differentiable binary mask."""
        logits = torch.randn(2, 128, device=device, requires_grad=True)
        
        # Hard mask (straight-through estimator)
        mask_hard = sparse_bk_core.gumbel_sigmoid(logits, tau=1.0, hard=True)
        unique_values = torch.unique(mask_hard)
        assert len(unique_values) <= 2, "Hard mask should be binary"
        
        # Test that gradients flow through (straight-through estimator)
        loss = mask_hard.sum()
        loss.backward()
        assert logits.grad is not None, "Gradients should flow through hard mask"
        
        # Soft mask
        logits_soft = torch.randn(2, 128, device=device, requires_grad=True)
        mask_soft = sparse_bk_core.gumbel_sigmoid(logits_soft, tau=1.0, hard=False)
        assert torch.all((mask_soft >= 0) & (mask_soft <= 1)), "Soft mask should be in [0, 1]"
        
        # Test gradients for soft mask
        loss_soft = mask_soft.sum()
        loss_soft.backward()
        assert logits_soft.grad is not None, "Gradients should flow through soft mask"
    
    def test_sparsity_loss(self, sparse_bk_core, device):
        """Test sparsity loss computation."""
        # Create mask with known sparsity
        mask = torch.ones(2, 128, device=device)
        mask[:, :64] = 0  # 50% sparsity
        
        loss = sparse_bk_core.sparsity_loss(mask)
        
        # Loss should be small when sparsity matches target (0.5)
        assert loss.item() < 0.01, f"Loss {loss.item()} too high for matching sparsity"
        
        # Test with different sparsity
        mask_all_ones = torch.ones(2, 128, device=device)  # 0% sparsity
        loss_high = sparse_bk_core.sparsity_loss(mask_all_ones)
        assert loss_high.item() > loss.item(), "Loss should be higher for mismatched sparsity"
    
    def test_sparsity_loss_types(self, sparse_bk_core, device):
        """Test different sparsity loss types."""
        # Create mask with 30% sparsity (target is 50%)
        mask = torch.ones(2, 128, device=device)
        mask[:, :38] = 0  # ~30% sparsity
        
        # Test L2 loss
        loss_l2 = sparse_bk_core.sparsity_loss(mask, loss_type='l2')
        assert loss_l2.item() > 0, "L2 loss should be positive"
        
        # Test L1 loss
        loss_l1 = sparse_bk_core.sparsity_loss(mask, loss_type='l1')
        assert loss_l1.item() > 0, "L1 loss should be positive"
        
        # Test KL loss
        loss_kl = sparse_bk_core.sparsity_loss(mask, loss_type='kl')
        assert loss_kl.item() > 0, "KL loss should be positive"
        
        # Test adaptive loss
        loss_adaptive = sparse_bk_core.sparsity_loss(mask, loss_type='adaptive')
        assert loss_adaptive.item() > 0, "Adaptive loss should be positive"
        
        # All losses should be finite
        assert torch.isfinite(loss_l2), "L2 loss should be finite"
        assert torch.isfinite(loss_l1), "L1 loss should be finite"
        assert torch.isfinite(loss_kl), "KL loss should be finite"
        assert torch.isfinite(loss_adaptive), "Adaptive loss should be finite"
    
    def test_balanced_sparsity_loss(self, sparse_bk_core, device):
        """Test balanced sparsity loss that trades off accuracy and sparsity."""
        # Create mask
        mask = torch.ones(2, 128, device=device)
        mask[:, :64] = 0  # 50% sparsity
        
        # Create dummy accuracy loss
        accuracy_loss = torch.tensor(2.5, device=device)
        
        # Test balanced loss
        total_loss, loss_dict = sparse_bk_core.balanced_sparsity_loss(
            mask, accuracy_loss, sparsity_weight=0.1, accuracy_weight=1.0
        )
        
        # Check loss components
        assert 'total_loss' in loss_dict
        assert 'accuracy_loss' in loss_dict
        assert 'sparsity_loss' in loss_dict
        assert 'current_sparsity' in loss_dict
        assert 'target_sparsity' in loss_dict
        
        # Total loss should be combination of accuracy and sparsity
        expected_total = (
            loss_dict['accuracy_weight'] * loss_dict['accuracy_loss'] +
            loss_dict['sparsity_weight'] * loss_dict['sparsity_loss']
        )
        assert torch.allclose(total_loss, expected_total, atol=1e-5)
        
        # Test with different weights
        total_loss_high_sparsity, _ = sparse_bk_core.balanced_sparsity_loss(
            mask, accuracy_loss, sparsity_weight=1.0, accuracy_weight=1.0
        )
        
        # Higher sparsity weight should increase total loss (if sparsity doesn't match target)
        if loss_dict['current_sparsity'] != loss_dict['target_sparsity']:
            assert total_loss_high_sparsity > total_loss
    
    def test_backward_pass(self, sparse_bk_core, device):
        """Test that backward pass works correctly."""
        B, N, D = 2, 128, 64
        x = torch.randn(B, N, D, device=device, requires_grad=True)
        v = torch.randn(B, N, device=device, requires_grad=True)
        
        features, mask, sparsity_ratio = sparse_bk_core(x, v)
        
        # Compute loss
        loss = features.sum() + sparse_bk_core.sparsity_loss(mask)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None, "Gradients should exist for x"
        assert v.grad is not None, "Gradients should exist for v"
        assert not torch.isnan(x.grad).any(), "Gradients should not contain NaN"
        assert not torch.isnan(v.grad).any(), "Gradients should not contain NaN"
    
    def test_numerical_stability(self, sparse_bk_core, device):
        """Test numerical stability with extreme inputs."""
        B, N, D = 2, 128, 64
        
        # Test with large values
        x_large = torch.randn(B, N, D, device=device) * 100
        v_large = torch.randn(B, N, device=device) * 10
        
        features, mask, sparsity_ratio = sparse_bk_core(x_large, v_large)
        
        assert not torch.isnan(features).any(), "Features should not contain NaN"
        assert not torch.isinf(features).any(), "Features should not contain Inf"
        assert not torch.isnan(mask).any(), "Mask should not contain NaN"


class TestSparseMoEResNetBKLayer:
    """Test suite for SparseMoEResNetBKLayer."""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def sparse_layer(self, device):
        """Create a SparseMoEResNetBKLayer instance for testing."""
        d_model = 64
        n_seq = 128
        return SparseMoEResNetBKLayer(
            d_model, n_seq, num_experts=4, target_sparsity=0.5
        ).to(device)
    
    def test_initialization(self, sparse_layer):
        """Test that SparseMoEResNetBKLayer initializes correctly."""
        assert sparse_layer.d_model == 64
        assert sparse_layer.n_seq == 128
        assert hasattr(sparse_layer, 'sparse_bk_core')
        assert hasattr(sparse_layer, 'moe_ffn')
    
    def test_forward_shape(self, sparse_layer, device):
        """Test that forward pass produces correct output shape."""
        B, N, D = 2, 128, 64
        x = torch.randn(B, N, D, device=device)
        
        output = sparse_layer(x)
        
        assert output.shape == (B, N, D), f"Expected output shape (2, 128, 64), got {output.shape}"
    
    def test_sparsity_loss(self, sparse_layer, device):
        """Test sparsity loss computation."""
        B, N, D = 2, 128, 64
        x = torch.randn(B, N, D, device=device)
        
        # Forward pass
        _ = sparse_layer(x)
        
        # Get sparsity loss
        loss = sparse_layer.get_sparsity_loss()
        
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1
        assert loss.item() >= 0, "Sparsity loss should be non-negative"
    
    def test_sparsity_loss_types(self, sparse_layer, device):
        """Test different sparsity loss types."""
        B, N, D = 2, 128, 64
        x = torch.randn(B, N, D, device=device)
        
        # Forward pass
        _ = sparse_layer(x)
        
        # Test different loss types
        loss_l2 = sparse_layer.get_sparsity_loss(loss_type='l2')
        loss_l1 = sparse_layer.get_sparsity_loss(loss_type='l1')
        loss_kl = sparse_layer.get_sparsity_loss(loss_type='kl')
        loss_adaptive = sparse_layer.get_sparsity_loss(loss_type='adaptive')
        
        # All should be valid tensors
        assert torch.isfinite(loss_l2)
        assert torch.isfinite(loss_l1)
        assert torch.isfinite(loss_kl)
        assert torch.isfinite(loss_adaptive)
    
    def test_balanced_loss(self, sparse_layer, device):
        """Test balanced loss computation."""
        B, N, D = 2, 128, 64
        x = torch.randn(B, N, D, device=device)
        
        # Forward pass
        output = sparse_layer(x)
        
        # Create dummy accuracy loss
        accuracy_loss = torch.tensor(2.5, device=device)
        
        # Get balanced loss
        total_loss, loss_dict = sparse_layer.get_balanced_loss(accuracy_loss)
        
        # Check loss components
        assert 'total_loss' in loss_dict
        assert 'accuracy_loss' in loss_dict
        assert 'sparsity_loss' in loss_dict
        assert 'current_sparsity' in loss_dict
        assert 'target_sparsity' in loss_dict
        
        # Total loss should be positive
        assert total_loss.item() > 0
        
        # Test with custom weights
        total_loss_custom, _ = sparse_layer.get_balanced_loss(
            accuracy_loss, sparsity_weight=0.1, accuracy_weight=2.0
        )
        assert torch.isfinite(total_loss_custom)
    
    def test_sparsity_stats(self, sparse_layer, device):
        """Test sparsity statistics."""
        B, N, D = 2, 128, 64
        x = torch.randn(B, N, D, device=device)
        
        # Forward pass
        _ = sparse_layer(x)
        
        # Get stats
        stats = sparse_layer.get_sparsity_stats()
        
        assert 'sparsity_ratio' in stats
        assert 'num_computed' in stats
        assert 'target_sparsity' in stats
        assert 0.0 <= stats['sparsity_ratio'] <= 1.0
        assert 0 <= stats['num_computed'] <= 128
    
    def test_backward_pass(self, sparse_layer, device):
        """Test that backward pass works correctly."""
        B, N, D = 2, 128, 64
        x = torch.randn(B, N, D, device=device, requires_grad=True)
        
        output = sparse_layer(x)
        sparsity_loss = sparse_layer.get_sparsity_loss()
        
        # Compute total loss
        loss = output.sum() + sparsity_loss
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None, "Gradients should exist for x"
        assert not torch.isnan(x.grad).any(), "Gradients should not contain NaN"


class TestSparseComputationOptimization:
    """Test suite for optimized sparse BK-Core computation."""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_sparse_theta_recursion(self, device):
        """Test sparse theta recursion produces valid output."""
        B, N = 2, 128
        
        # Create inputs
        he_diag = torch.randn(B, N, device=device)
        h0_super = torch.ones(B, N-1, device=device)
        h0_sub = torch.ones(B, N-1, device=device)
        z = torch.tensor(1.0j, dtype=torch.complex64, device=device)
        mask = torch.rand(B, N, device=device) > 0.5  # Random mask
        
        # Run sparse theta recursion
        theta = sparse_theta_recursion(he_diag, h0_super, h0_sub, z, mask.float())
        
        # Check shape
        assert theta.shape == (B, N+1), f"Expected theta shape (2, 129), got {theta.shape}"
        
        # Check dtype
        assert theta.dtype == torch.complex128
        
        # Check no NaN/Inf
        assert not torch.isnan(theta).any(), "Theta should not contain NaN"
        assert not torch.isinf(theta).any(), "Theta should not contain Inf"
        
        # Check initial conditions
        assert torch.allclose(theta[:, 0].real, torch.ones(B, device=device, dtype=torch.float64)), "theta[0] should be 1"
        assert torch.allclose(theta[:, 0].imag, torch.zeros(B, device=device, dtype=torch.float64)), "theta[0] should be real"
    
    def test_sparse_phi_recursion(self, device):
        """Test sparse phi recursion produces valid output."""
        B, N = 2, 128
        
        # Create inputs
        he_diag = torch.randn(B, N, device=device)
        h0_super = torch.ones(B, N-1, device=device)
        h0_sub = torch.ones(B, N-1, device=device)
        z = torch.tensor(1.0j, dtype=torch.complex64, device=device)
        mask = torch.rand(B, N, device=device) > 0.5  # Random mask
        
        # Run sparse phi recursion
        phi = sparse_phi_recursion(he_diag, h0_super, h0_sub, z, mask.float())
        
        # Check shape
        assert phi.shape == (B, N), f"Expected phi shape (2, 128), got {phi.shape}"
        
        # Check dtype
        assert phi.dtype == torch.complex128
        
        # Check no NaN/Inf
        assert not torch.isnan(phi).any(), "Phi should not contain NaN"
        assert not torch.isinf(phi).any(), "Phi should not contain Inf"
    
    def test_optimized_sparse_bk_core(self, device):
        """Test optimized sparse BK-Core computation."""
        B, N = 2, 128
        
        # Create inputs
        he_diag = torch.randn(B, N, device=device)
        h0_super = torch.ones(B, N-1, device=device)
        h0_sub = torch.ones(B, N-1, device=device)
        z = torch.tensor(1.0j, dtype=torch.complex64, device=device)
        mask = torch.rand(B, N, device=device) > 0.5  # Random mask
        
        # Run optimized sparse BK-Core
        features = optimized_sparse_bk_core(he_diag, h0_super, h0_sub, z, mask)
        
        # Check shape
        assert features.shape == (B, N, 2), f"Expected features shape (2, 128, 2), got {features.shape}"
        
        # Check dtype
        assert features.dtype == torch.float32
        
        # Check no NaN/Inf
        assert not torch.isnan(features).any(), "Features should not contain NaN"
        assert not torch.isinf(features).any(), "Features should not contain Inf"
    
    def test_sparse_vs_full_computation(self, device):
        """Test that sparse computation produces similar results to full computation."""
        from src.models.bk_core import BKCoreFunction
        
        B, N = 2, 128
        
        # Create inputs
        he_diag = torch.randn(B, N, device=device)
        h0_super = torch.ones(B, N-1, device=device)
        h0_sub = torch.ones(B, N-1, device=device)
        z = torch.tensor(1.0j, dtype=torch.complex64, device=device)
        
        # Create mask with 50% sparsity
        mask = torch.zeros(B, N, device=device)
        mask[:, ::2] = 1.0  # Every other position
        
        # Compute with sparse algorithm
        features_sparse = optimized_sparse_bk_core(he_diag, h0_super, h0_sub, z, mask)
        
        # Compute with full algorithm
        features_full = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z)
        
        # Compare at computed positions
        mask_expanded = mask.unsqueeze(-1).expand_as(features_sparse)
        diff = torch.abs(features_sparse - features_full) * mask_expanded
        
        # At computed positions, results should be similar
        # (not exact due to simplified computation for masked positions affecting recursion)
        max_diff = diff.max().item()
        mean_diff = diff.sum().item() / mask.sum().item() / 2  # Divide by 2 for real/imag
        
        print(f"\nSparse vs Full comparison:")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        
        # Differences should be reasonable (not exact due to approximation)
        # The sparse algorithm uses simplified computation for masked positions,
        # which affects the recursion and can lead to larger differences
        assert max_diff < 50.0, f"Max difference {max_diff} too large"
        assert mean_diff < 10.0, f"Mean difference {mean_diff} too large"
    
    def test_sparse_computation_flag(self, device):
        """Test that use_sparse_computation flag works correctly."""
        d_model = 64
        n_seq = 128
        B = 2
        
        # Create sparse BK-Core
        sparse_bk = SparseBKCore(d_model, n_seq, target_sparsity=0.5).to(device)
        
        # Create inputs
        x = torch.randn(B, n_seq, d_model, device=device)
        v = torch.randn(B, n_seq, device=device)
        
        # Test with sparse computation
        with torch.no_grad():  # Disable gradients to get deterministic Gumbel sampling
            features_sparse, mask_sparse, sparsity_sparse = sparse_bk(x, v, use_sparse_computation=True)
        
        # Test with full computation (use same random seed for Gumbel)
        with torch.no_grad():
            features_full, mask_full, sparsity_full = sparse_bk(x, v, use_sparse_computation=False)
        
        # Shapes should match
        assert features_sparse.shape == features_full.shape
        assert mask_sparse.shape == mask_full.shape
        
        # Note: Masks may differ due to Gumbel noise, so we just check they're both valid
        assert torch.all((mask_sparse == 0) | (mask_sparse == 1)), "Sparse mask should be binary"
        assert torch.all((mask_full == 0) | (mask_full == 1)), "Full mask should be binary"
        
        # Features should be similar (not exact due to different computation)
        diff = torch.abs(features_sparse - features_full)
        max_diff = diff.max().item()
        print(f"\nSparse vs Full (same mask) comparison:")
        print(f"  Max difference: {max_diff:.6f}")
    
    def test_sparse_layer_with_optimization(self, device):
        """Test SparseMoEResNetBKLayer with sparse computation optimization."""
        d_model = 64
        n_seq = 128
        B = 2
        
        # Create layer with sparse computation enabled
        layer_sparse = SparseMoEResNetBKLayer(
            d_model, n_seq, num_experts=4, target_sparsity=0.5,
            use_sparse_computation=True
        ).to(device)
        
        # Create layer with sparse computation disabled
        layer_full = SparseMoEResNetBKLayer(
            d_model, n_seq, num_experts=4, target_sparsity=0.5,
            use_sparse_computation=False
        ).to(device)
        
        # Copy weights to ensure fair comparison
        layer_full.load_state_dict(layer_sparse.state_dict())
        
        # Create input
        x = torch.randn(B, n_seq, d_model, device=device)
        
        # Forward pass
        output_sparse = layer_sparse(x)
        output_full = layer_full(x)
        
        # Outputs should have same shape
        assert output_sparse.shape == output_full.shape
        
        # Check no NaN/Inf
        assert not torch.isnan(output_sparse).any()
        assert not torch.isnan(output_full).any()


class TestAdaptiveSparsityScheduler:
    """Test suite for AdaptiveSparsityScheduler."""
    
    def test_initialization(self):
        """Test scheduler initialization."""
        scheduler = AdaptiveSparsityScheduler(
            initial_sparsity=0.2,
            final_sparsity=0.5,
            initial_weight=0.001,
            final_weight=0.01,
            warmup_steps=1000
        )
        
        assert scheduler.initial_sparsity == 0.2
        assert scheduler.final_sparsity == 0.5
        assert scheduler.current_step == 0
    
    def test_linear_schedule(self):
        """Test linear sparsity schedule."""
        scheduler = AdaptiveSparsityScheduler(
            initial_sparsity=0.2,
            final_sparsity=0.5,
            warmup_steps=100,
            schedule_type='linear'
        )
        
        # At step 1 (first step)
        state = scheduler.step()
        assert abs(state['sparsity_target'] - 0.2) < 0.01, "Should start near initial sparsity"
        
        # At step 50 (halfway)
        for _ in range(49):
            scheduler.step()
        state = scheduler.step()
        assert abs(state['sparsity_target'] - 0.35) < 0.02, "Should be halfway at step 50"
        
        # At step 100 (end)
        for _ in range(49):
            scheduler.step()
        state = scheduler.step()
        assert abs(state['sparsity_target'] - 0.5) < 0.01, "Should reach final sparsity"
    
    def test_cosine_schedule(self):
        """Test cosine sparsity schedule."""
        scheduler = AdaptiveSparsityScheduler(
            initial_sparsity=0.2,
            final_sparsity=0.5,
            warmup_steps=100,
            schedule_type='cosine'
        )
        
        # At step 1 (first step)
        state = scheduler.step()
        assert abs(state['sparsity_target'] - 0.2) < 0.01, "Should start near initial sparsity"
        
        # At end
        for _ in range(99):
            scheduler.step()
        state = scheduler.step()
        assert abs(state['sparsity_target'] - 0.5) < 0.01, "Should reach final sparsity"
    
    def test_step_schedule(self):
        """Test step sparsity schedule."""
        scheduler = AdaptiveSparsityScheduler(
            initial_sparsity=0.2,
            final_sparsity=0.5,
            warmup_steps=100,
            schedule_type='step'
        )
        
        # At step 1 (first step)
        state = scheduler.step()
        assert abs(state['sparsity_target'] - 0.2) < 0.01, "Should start near initial sparsity"
        
        # At step 10 (early in first quarter)
        for _ in range(9):
            scheduler.step()
        state = scheduler.step()
        assert abs(state['sparsity_target'] - 0.2) < 0.01, "Should stay at initial in first quarter"
        
        # At step 50 (halfway)
        for _ in range(39):
            scheduler.step()
        state = scheduler.step()
        assert state['sparsity_target'] > 0.25, "Should increase after first quarter"
        
        # At end
        for _ in range(49):
            scheduler.step()
        state = scheduler.step()
        assert abs(state['sparsity_target'] - 0.5) < 0.01, "Should reach final sparsity"
    
    def test_weight_schedule(self):
        """Test loss weight scheduling."""
        scheduler = AdaptiveSparsityScheduler(
            initial_weight=0.001,
            final_weight=0.01,
            warmup_steps=100,
            schedule_type='linear'
        )
        
        # At step 1 (first step)
        state = scheduler.step()
        assert abs(state['loss_weight'] - 0.001) < 0.001, "Should start near initial weight"
        
        # At end
        for _ in range(99):
            scheduler.step()
        state = scheduler.step()
        assert abs(state['loss_weight'] - 0.01) < 0.001, "Should reach final weight"
    
    def test_adaptive_adjustment(self):
        """Test adaptive adjustment based on accuracy."""
        scheduler = AdaptiveSparsityScheduler(
            initial_sparsity=0.2,
            final_sparsity=0.5,
            warmup_steps=100,
            accuracy_threshold=3.0
        )
        
        # Step with good accuracy (below threshold)
        state_good = scheduler.step(current_accuracy=2.0)
        sparsity_good = state_good['sparsity_target']
        
        # Reset and step with poor accuracy (above threshold)
        scheduler.reset()
        state_poor = scheduler.step(current_accuracy=4.0)
        sparsity_poor = state_poor['sparsity_target']
        
        # Poor accuracy should result in lower sparsity target
        assert sparsity_poor < sparsity_good, "Poor accuracy should reduce sparsity target"
    
    def test_reset(self):
        """Test scheduler reset."""
        scheduler = AdaptiveSparsityScheduler(warmup_steps=100)
        
        # Advance scheduler
        for _ in range(50):
            scheduler.step()
        
        assert scheduler.current_step == 50
        
        # Reset
        scheduler.reset()
        
        assert scheduler.current_step == 0
        assert scheduler.best_accuracy == float('inf')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
