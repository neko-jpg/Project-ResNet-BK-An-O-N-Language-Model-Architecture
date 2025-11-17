"""
Tests for Hugging Face Integration

Tests the ResNet-BK Hugging Face integration including:
- Model configuration
- Model creation
- Forward pass
- Save/load functionality
- ONNX export (if available)
"""

import torch
import torch.nn as nn
import pytest
import os
import tempfile
import shutil


def test_resnet_bk_config():
    """Test ResNetBKConfig creation and serialization."""
    from src.models.hf_resnet_bk import ResNetBKConfig
    
    # Create config
    config = ResNetBKConfig(
        vocab_size=1000,
        d_model=128,
        n_layers=4,
        n_seq=256,
        num_experts=2,
        use_birman_schwinger=True,
        epsilon=0.75,
    )
    
    # Check attributes
    assert config.vocab_size == 1000
    assert config.d_model == 128
    assert config.n_layers == 4
    assert config.n_seq == 256
    assert config.num_experts == 2
    assert config.use_birman_schwinger == True
    assert config.epsilon == 0.75
    
    # Test serialization
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert config_dict['vocab_size'] == 1000
    
    print("✓ ResNetBKConfig test passed")


def test_resnet_bk_model_creation():
    """Test ResNetBKForCausalLM model creation."""
    from src.models.hf_resnet_bk import ResNetBKConfig, ResNetBKForCausalLM
    
    # Create small model
    config = ResNetBKConfig(
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        n_seq=128,
        num_experts=2,
    )
    
    model = ResNetBKForCausalLM(config)
    
    # Check model structure
    assert hasattr(model, 'model')
    assert hasattr(model, 'config')
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0
    
    print(f"✓ Model created with {num_params/1e6:.2f}M parameters")


def test_forward_pass():
    """Test forward pass through the model."""
    from src.models.hf_resnet_bk import ResNetBKConfig, ResNetBKForCausalLM
    
    # Create small model
    config = ResNetBKConfig(
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        n_seq=128,
        num_experts=2,
    )
    
    model = ResNetBKForCausalLM(config)
    model.eval()
    
    # Create dummy input (must match n_seq from config)
    batch_size = 2
    seq_length = 128  # Match config.n_seq
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
    
    # Check output shape
    assert hasattr(outputs, 'logits')
    assert outputs.logits.shape == (batch_size, seq_length, 1000)
    
    print(f"✓ Forward pass successful: {outputs.logits.shape}")


def test_forward_with_labels():
    """Test forward pass with labels (training mode)."""
    from src.models.hf_resnet_bk import ResNetBKConfig, ResNetBKForCausalLM
    
    # Create small model
    config = ResNetBKConfig(
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        n_seq=128,
        num_experts=2,
    )
    
    model = ResNetBKForCausalLM(config)
    
    # Create dummy input and labels (must match n_seq from config)
    batch_size = 2
    seq_length = 128  # Match config.n_seq
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    labels = torch.randint(0, 1000, (batch_size, seq_length))
    
    # Forward pass with labels
    outputs = model(input_ids, labels=labels)
    
    # Check loss
    assert hasattr(outputs, 'loss')
    assert outputs.loss is not None
    assert outputs.loss.item() > 0
    
    print(f"✓ Training forward pass successful: loss={outputs.loss.item():.4f}")


def test_save_and_load():
    """Test saving and loading model."""
    from src.models.hf_resnet_bk import ResNetBKConfig, ResNetBKForCausalLM
    
    # Create small model
    config = ResNetBKConfig(
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        n_seq=128,
        num_experts=2,
    )
    
    model = ResNetBKForCausalLM(config)
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save model
        model.save_pretrained(temp_dir)
        
        # Check files exist
        assert os.path.exists(os.path.join(temp_dir, "config.json"))
        assert os.path.exists(os.path.join(temp_dir, "pytorch_model.bin"))
        
        # Load model
        loaded_model = ResNetBKForCausalLM.from_pretrained(temp_dir)
        
        # Test loaded model (must match n_seq from config)
        model.eval()
        loaded_model.eval()
        input_ids = torch.randint(0, 1000, (1, 128))  # Match config.n_seq
        with torch.no_grad():
            original_output = model(input_ids)
            loaded_output = loaded_model(input_ids)
        
        # Check outputs are close (use larger tolerance due to numerical differences)
        assert torch.allclose(original_output.logits, loaded_output.logits, atol=1e-3)
        
        print("✓ Save and load test passed")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_create_resnet_bk_for_hf():
    """Test convenience function for creating models."""
    from src.models.hf_resnet_bk import create_resnet_bk_for_hf
    
    # Test different sizes
    sizes = ["1M", "10M"]
    
    for size in sizes:
        model = create_resnet_bk_for_hf(size)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Created {size} model with {num_params/1e6:.2f}M parameters")


def test_resize_token_embeddings():
    """Test resizing token embeddings."""
    from src.models.hf_resnet_bk import ResNetBKConfig, ResNetBKForCausalLM
    
    # Create model
    config = ResNetBKConfig(vocab_size=1000, d_model=64, n_layers=2, n_seq=128)
    model = ResNetBKForCausalLM(config)
    
    # Resize embeddings
    new_vocab_size = 1500
    model.resize_token_embeddings(new_vocab_size)
    
    # Check new size
    assert model.config.vocab_size == new_vocab_size
    assert model.get_input_embeddings().weight.shape[0] == new_vocab_size
    assert model.get_output_embeddings().weight.shape[0] == new_vocab_size
    
    # Test forward pass with new vocab size (must match n_seq from config)
    input_ids = torch.randint(0, new_vocab_size, (1, 128))  # Match config.n_seq
    with torch.no_grad():
        outputs = model(input_ids)
    
    assert outputs.logits.shape[-1] == new_vocab_size
    
    print(f"✓ Resize embeddings test passed: {1000} -> {new_vocab_size}")


def test_hubconf():
    """Test PyTorch Hub configuration."""
    import sys
    import importlib.util
    
    # Load hubconf module
    spec = importlib.util.spec_from_file_location("hubconf", "hubconf.py")
    hubconf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hubconf)
    
    # Test creating models
    model = hubconf.resnet_bk_1m(pretrained=False)
    assert model is not None
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Hub model created with {num_params/1e6:.2f}M parameters")
    
    # Test custom model
    custom_model = hubconf.resnet_bk_custom(
        d_model=128,
        n_layers=4,
        use_birman_schwinger=True,
    )
    assert custom_model is not None
    print("✓ Custom hub model created")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_forward():
    """Test forward pass on CUDA."""
    from src.models.hf_resnet_bk import ResNetBKConfig, ResNetBKForCausalLM
    
    # Create model
    config = ResNetBKConfig(
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        n_seq=128,
        num_experts=2,
    )
    
    model = ResNetBKForCausalLM(config).cuda()
    model.eval()
    
    # Create input on CUDA (must match n_seq from config)
    input_ids = torch.randint(0, 1000, (2, 128)).cuda()  # Match config.n_seq
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
    
    assert outputs.logits.is_cuda
    print("✓ CUDA forward pass successful")


def test_onnx_export_basic():
    """Test basic ONNX export functionality (without actual export)."""
    try:
        from src.models.onnx_export import export_to_onnx
        print("✓ ONNX export module imported successfully")
    except ImportError as e:
        print(f"⚠ ONNX export not available: {e}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ResNet-BK Hugging Face Integration Tests")
    print("="*60 + "\n")
    
    # Run tests
    test_resnet_bk_config()
    test_resnet_bk_model_creation()
    test_forward_pass()
    test_forward_with_labels()
    test_save_and_load()
    test_create_resnet_bk_for_hf()
    test_resize_token_embeddings()
    test_hubconf()
    
    if torch.cuda.is_available():
        test_cuda_forward()
    
    test_onnx_export_basic()
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60 + "\n")
