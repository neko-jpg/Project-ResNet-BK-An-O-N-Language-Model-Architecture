"""
PyTorch Hub Configuration for ResNet-BK Models

This file enables loading ResNet-BK models via torch.hub.load().

Example usage:
    ```python
    import torch
    
    # Load a pre-trained 1B parameter model
    model = torch.hub.load('username/resnet-bk', 'resnet_bk_1b', pretrained=True)
    
    # Load a model with custom configuration
    model = torch.hub.load('username/resnet-bk', 'resnet_bk_custom',
                          d_model=512, n_layers=12, use_birman_schwinger=True)
    ```
"""

dependencies = ['torch']

import torch
import torch.nn as nn
from typing import Optional
import os


def _load_pretrained_weights(model, model_name: str, force_reload: bool = False):
    """
    Load pretrained weights from Hugging Face Hub or local cache.
    
    Args:
        model: ResNet-BK model instance
        model_name: Name of the pretrained model
        force_reload: Force re-download even if cached
        
    Returns:
        Model with loaded weights
    """
    try:
        from huggingface_hub import hf_hub_download
        
        # Map model names to HF Hub repository
        repo_id = f"resnet-bk/{model_name}"
        
        # Download model weights
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename="pytorch_model.bin",
            force_download=force_reload,
        )
        
        # Load weights
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        print(f"Loaded pretrained weights for {model_name}")
        
    except Exception as e:
        print(f"Warning: Could not load pretrained weights for {model_name}: {e}")
        print("Using randomly initialized weights.")
    
    return model


def resnet_bk_1m(pretrained: bool = False, **kwargs):
    """
    ResNet-BK 1M parameter model.
    
    Args:
        pretrained: Load pretrained weights if True
        **kwargs: Additional configuration parameters
        
    Returns:
        ResNet-BK model with ~1M parameters
    """
    from src.models.hf_resnet_bk import create_resnet_bk_for_hf
    
    model = create_resnet_bk_for_hf("1M", **kwargs)
    
    if pretrained:
        model = _load_pretrained_weights(model, "resnet-bk-1m")
    
    return model


def resnet_bk_10m(pretrained: bool = False, **kwargs):
    """
    ResNet-BK 10M parameter model.
    
    Args:
        pretrained: Load pretrained weights if True
        **kwargs: Additional configuration parameters
        
    Returns:
        ResNet-BK model with ~10M parameters
    """
    from src.models.hf_resnet_bk import create_resnet_bk_for_hf
    
    model = create_resnet_bk_for_hf("10M", **kwargs)
    
    if pretrained:
        model = _load_pretrained_weights(model, "resnet-bk-10m")
    
    return model


def resnet_bk_100m(pretrained: bool = False, **kwargs):
    """
    ResNet-BK 100M parameter model.
    
    Args:
        pretrained: Load pretrained weights if True
        **kwargs: Additional configuration parameters
        
    Returns:
        ResNet-BK model with ~100M parameters
    """
    from src.models.hf_resnet_bk import create_resnet_bk_for_hf
    
    model = create_resnet_bk_for_hf("100M", **kwargs)
    
    if pretrained:
        model = _load_pretrained_weights(model, "resnet-bk-100m")
    
    return model


def resnet_bk_1b(pretrained: bool = False, **kwargs):
    """
    ResNet-BK 1B parameter model.
    
    Args:
        pretrained: Load pretrained weights if True
        **kwargs: Additional configuration parameters
        
    Returns:
        ResNet-BK model with ~1B parameters
    """
    from src.models.hf_resnet_bk import create_resnet_bk_for_hf
    
    model = create_resnet_bk_for_hf("1B", **kwargs)
    
    if pretrained:
        model = _load_pretrained_weights(model, "resnet-bk-1b")
    
    return model


def resnet_bk_10b(pretrained: bool = False, **kwargs):
    """
    ResNet-BK 10B parameter model.
    
    Args:
        pretrained: Load pretrained weights if True
        **kwargs: Additional configuration parameters
        
    Returns:
        ResNet-BK model with ~10B parameters
    """
    from src.models.hf_resnet_bk import create_resnet_bk_for_hf
    
    model = create_resnet_bk_for_hf("10B", **kwargs)
    
    if pretrained:
        model = _load_pretrained_weights(model, "resnet-bk-10b")
    
    return model


def resnet_bk_custom(
    d_model: int = 256,
    n_layers: int = 8,
    n_seq: int = 2048,
    num_experts: int = 4,
    top_k: int = 1,
    use_birman_schwinger: bool = False,
    use_scattering_router: bool = False,
    **kwargs
):
    """
    Create a custom ResNet-BK model with specified configuration.
    
    Args:
        d_model: Model dimension
        n_layers: Number of layers
        n_seq: Maximum sequence length
        num_experts: Number of MoE experts
        top_k: Number of experts to route to
        use_birman_schwinger: Use Birman-Schwinger core
        use_scattering_router: Use scattering-based routing
        **kwargs: Additional configuration parameters
        
    Returns:
        Custom ResNet-BK model
    """
    from src.models.hf_resnet_bk import ResNetBKConfig, ResNetBKForCausalLM
    
    config = ResNetBKConfig(
        d_model=d_model,
        n_layers=n_layers,
        n_seq=n_seq,
        num_experts=num_experts,
        top_k=top_k,
        use_birman_schwinger=use_birman_schwinger,
        use_scattering_router=use_scattering_router,
        **kwargs
    )
    
    model = ResNetBKForCausalLM(config)
    
    return model


# Alias for backward compatibility
resnet_bk = resnet_bk_1b
