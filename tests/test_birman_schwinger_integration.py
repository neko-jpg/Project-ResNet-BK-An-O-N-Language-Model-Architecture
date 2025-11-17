"""
Test Birman-Schwinger Integration with ResNet-BK

This script tests that the Birman-Schwinger core is properly integrated
into the ResNet-BK architecture and that stability monitoring works.
"""

import torch
import torch.nn as nn
from src.models.resnet_bk import LanguageModel


def test_birman_schwinger_integration():
    """Test that Birman-Schwinger core integrates properly with ResNet-BK."""
    
    print("Testing Birman-Schwinger Integration...")
    print("=" * 60)
    
    # Test configuration
    vocab_size = 1000
    d_model = 64
    n_layers = 2
    n_seq = 128
    batch_size = 4
    
    # Test 1: Model with Birman-Schwinger core
    print("\n1. Creating model with Birman-Schwinger core...")
    model_bs = LanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_seq=n_seq,
        num_experts=4,
        top_k=2,
        use_birman_schwinger=True,
        epsilon=1.0,
        use_mourre=True,
        use_lap=True,
        prime_bump_init=False,  # Test without prime bump first
    )
    print(f"   ✓ Model created with {sum(p.numel() for p in model_bs.parameters())} parameters")
    
    # Test 2: Forward pass
    print("\n2. Testing forward pass...")
    x = torch.randint(0, vocab_size, (batch_size, n_seq))
    
    try:
        logits = model_bs(x)
        assert logits.shape == (batch_size, n_seq, vocab_size), f"Expected shape {(batch_size, n_seq, vocab_size)}, got {logits.shape}"
        assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"
        print(f"   ✓ Forward pass successful, output shape: {logits.shape}")
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        return False
    
    # Test 3: Stability diagnostics
    print("\n3. Testing stability diagnostics...")
    try:
        diagnostics = model_bs.get_stability_diagnostics()
        assert isinstance(diagnostics, dict), "Diagnostics should be a dictionary"
        
        expected_keys = [
            'mean_schatten_s1', 'mean_schatten_s2',
            'mean_condition_number', 'max_condition_number',
            'mourre_verified_rate', 's1_bound_satisfied_rate',
            's2_bound_satisfied_rate', 'all_finite_rate',
            'precision_upgrades'
        ]
        
        for key in expected_keys:
            assert key in diagnostics, f"Missing diagnostic key: {key}"
        
        print(f"   ✓ Stability diagnostics available:")
        print(f"     - Mean Schatten S2: {diagnostics['mean_schatten_s2']:.4f}")
        print(f"     - Max Condition Number: {diagnostics['max_condition_number']:.2e}")
        print(f"     - Mourre Verified Rate: {diagnostics['mourre_verified_rate']:.1%}")
        print(f"     - All Finite Rate: {diagnostics['all_finite_rate']:.1%}")
        
    except Exception as e:
        print(f"   ✗ Stability diagnostics failed: {e}")
        return False
    
    # Test 4: Model with Prime-Bump initialization
    print("\n4. Creating model with Prime-Bump initialization...")
    model_pb = LanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_seq=n_seq,
        num_experts=4,
        top_k=2,
        use_birman_schwinger=True,
        epsilon=1.0,
        prime_bump_init=True,
        prime_bump_scale=0.02,
        k_max=3,
    )
    print(f"   ✓ Model created with Prime-Bump initialization")
    
    # Test 5: Forward pass with Prime-Bump
    print("\n5. Testing forward pass with Prime-Bump...")
    try:
        logits_pb = model_pb(x)
        assert logits_pb.shape == (batch_size, n_seq, vocab_size)
        assert torch.isfinite(logits_pb).all(), "Logits contain NaN or Inf"
        print(f"   ✓ Forward pass successful with Prime-Bump")
    except Exception as e:
        print(f"   ✗ Forward pass with Prime-Bump failed: {e}")
        return False
    
    # Test 6: Backward pass
    print("\n6. Testing backward pass...")
    try:
        criterion = nn.CrossEntropyLoss()
        target = torch.randint(0, vocab_size, (batch_size * n_seq,))
        loss = criterion(logits.view(-1, vocab_size), target)
        
        loss.backward()
        
        # Check gradients
        has_grad = False
        for name, param in model_bs.named_parameters():
            if param.grad is not None:
                has_grad = True
                assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains NaN or Inf"
        
        assert has_grad, "No gradients computed"
        print(f"   ✓ Backward pass successful, loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"   ✗ Backward pass failed: {e}")
        return False
    
    # Test 7: Compare with original BK-Core
    print("\n7. Comparing with original BK-Core...")
    model_original = LanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_seq=n_seq,
        num_experts=4,
        top_k=2,
        use_birman_schwinger=False,  # Use original BK-Core
    )
    
    try:
        logits_original = model_original(x)
        assert logits_original.shape == (batch_size, n_seq, vocab_size)
        assert torch.isfinite(logits_original).all()
        print(f"   ✓ Original BK-Core still works")
        
        # Check that diagnostics are empty for original core
        diagnostics_original = model_original.get_stability_diagnostics()
        assert diagnostics_original == {}, "Original core should have empty diagnostics"
        print(f"   ✓ Original core returns empty diagnostics as expected")
        
    except Exception as e:
        print(f"   ✗ Original BK-Core test failed: {e}")
        return False
    
    # Test 8: Epsilon parameter
    print("\n8. Testing epsilon parameter...")
    for epsilon in [1.0, 0.75, 0.5]:
        try:
            model_eps = LanguageModel(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=1,
                n_seq=64,
                use_birman_schwinger=True,
                epsilon=epsilon,
            )
            logits_eps = model_eps(x[:, :64])
            assert torch.isfinite(logits_eps).all()
            print(f"   ✓ Epsilon {epsilon} works")
        except Exception as e:
            print(f"   ✗ Epsilon {epsilon} failed: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("✓ All integration tests passed!")
    print("=" * 60)
    
    return True


def test_stability_monitoring_workflow():
    """Test the stability monitoring workflow during training."""
    
    print("\n\nTesting Stability Monitoring Workflow...")
    print("=" * 60)
    
    # Create model
    model = LanguageModel(
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        n_seq=128,
        use_birman_schwinger=True,
        epsilon=1.0,
        prime_bump_init=True,
    )
    
    # Simulate training steps
    print("\n1. Simulating training steps...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for step in range(5):
        # Forward pass
        x = torch.randint(0, 1000, (2, 128))
        logits = model(x)
        
        # Loss
        target = torch.randint(0, 1000, (2 * 128,))
        loss = criterion(logits.view(-1, 1000), target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Get diagnostics
        diagnostics = model.get_stability_diagnostics()
        
        print(f"   Step {step + 1}:")
        print(f"     Loss: {loss.item():.4f}")
        print(f"     Mean Schatten S2: {diagnostics.get('mean_schatten_s2', 0.0):.4f}")
        print(f"     Max Condition Number: {diagnostics.get('max_condition_number', 0.0):.2e}")
        print(f"     All Finite: {diagnostics.get('all_finite_rate', 1.0):.1%}")
    
    print("\n   ✓ Stability monitoring workflow successful")
    
    print("\n" + "=" * 60)
    print("✓ Stability monitoring workflow test passed!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    # Run tests
    success = True
    
    try:
        success = test_birman_schwinger_integration() and success
    except Exception as e:
        print(f"\n✗ Integration test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    try:
        success = test_stability_monitoring_workflow() and success
    except Exception as e:
        print(f"\n✗ Workflow test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    if success:
        print("\n" + "=" * 60)
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("=" * 60)
        exit(1)
