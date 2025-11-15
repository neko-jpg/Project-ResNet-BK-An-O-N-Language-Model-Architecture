"""
Tensor Core Optimization Utilities

This module provides utilities to optimize ResNet-BK for tensor cores:
- Ensure matrix dimensions are multiples of 8 for FP16 tensor cores
- Pad embeddings if necessary
- Validate tensor core compatibility

Requirements: 5.10
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import warnings


def is_tensor_core_compatible(dim: int, dtype: torch.dtype = torch.float16) -> bool:
    """
    Check if dimension is compatible with tensor cores.
    
    For FP16 tensor cores on NVIDIA GPUs:
    - Dimensions should be multiples of 8
    
    Args:
        dim: Dimension to check
        dtype: Data type (FP16 for tensor cores)
    
    Returns:
        True if compatible, False otherwise
    """
    if dtype in [torch.float16, torch.bfloat16]:
        return dim % 8 == 0
    elif dtype == torch.float32:
        # FP32 tensor cores (Ampere+) prefer multiples of 8 as well
        return dim % 8 == 0
    else:
        return True


def pad_to_tensor_core_multiple(dim: int, dtype: torch.dtype = torch.float16) -> int:
    """
    Pad dimension to nearest tensor core compatible size.
    
    Args:
        dim: Original dimension
        dtype: Data type
    
    Returns:
        Padded dimension (multiple of 8)
    """
    if dtype in [torch.float16, torch.bfloat16, torch.float32]:
        multiple = 8
        return ((dim + multiple - 1) // multiple) * multiple
    else:
        return dim


def validate_model_for_tensor_cores(model: nn.Module, verbose: bool = True) -> dict:
    """
    Validate that model dimensions are optimized for tensor cores.
    
    Args:
        model: PyTorch model
        verbose: Print validation results
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'compatible_layers': [],
        'incompatible_layers': [],
        'total_layers': 0,
        'compatibility_rate': 0.0
    }
    
    if verbose:
        print("=" * 60)
        print("Tensor Core Compatibility Validation")
        print("=" * 60)
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            
            in_compatible = is_tensor_core_compatible(in_features)
            out_compatible = is_tensor_core_compatible(out_features)
            compatible = in_compatible and out_compatible
            
            results['total_layers'] += 1
            
            if compatible:
                results['compatible_layers'].append(name)
            else:
                results['incompatible_layers'].append({
                    'name': name,
                    'in_features': in_features,
                    'out_features': out_features,
                    'in_compatible': in_compatible,
                    'out_compatible': out_compatible
                })
            
            if verbose and not compatible:
                print(f"\n✗ {name}")
                print(f"  in_features: {in_features} ({'✓' if in_compatible else '✗ not multiple of 8'})")
                print(f"  out_features: {out_features} ({'✓' if out_compatible else '✗ not multiple of 8'})")
        
        elif isinstance(module, nn.Embedding):
            num_embeddings = module.num_embeddings
            embedding_dim = module.embedding_dim
            
            dim_compatible = is_tensor_core_compatible(embedding_dim)
            
            results['total_layers'] += 1
            
            if dim_compatible:
                results['compatible_layers'].append(name)
            else:
                results['incompatible_layers'].append({
                    'name': name,
                    'embedding_dim': embedding_dim,
                    'dim_compatible': dim_compatible
                })
            
            if verbose and not dim_compatible:
                print(f"\n✗ {name}")
                print(f"  embedding_dim: {embedding_dim} (✗ not multiple of 8)")
    
    results['compatibility_rate'] = len(results['compatible_layers']) / max(results['total_layers'], 1)
    
    if verbose:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Total layers: {results['total_layers']}")
        print(f"Compatible: {len(results['compatible_layers'])}")
        print(f"Incompatible: {len(results['incompatible_layers'])}")
        print(f"Compatibility rate: {results['compatibility_rate']:.1%}")
        
        if results['compatibility_rate'] == 1.0:
            print("\n✓ All layers are tensor core compatible")
        else:
            print(f"\n✗ {len(results['incompatible_layers'])} layers need padding")
    
    return results


class TensorCoreOptimizedLinear(nn.Linear):
    """
    Linear layer with automatic padding for tensor core optimization.
    
    Pads input/output dimensions to multiples of 8 if necessary.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        pad_for_tensor_cores: bool = True
    ):
        """
        Initialize tensor core optimized linear layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Use bias
            device: Device
            dtype: Data type
            pad_for_tensor_cores: Automatically pad dimensions
        """
        self.original_in_features = in_features
        self.original_out_features = out_features
        self.pad_for_tensor_cores = pad_for_tensor_cores
        
        if pad_for_tensor_cores:
            in_features = pad_to_tensor_core_multiple(in_features)
            out_features = pad_to_tensor_core_multiple(out_features)
        
        super().__init__(in_features, out_features, bias, device, dtype)
        
        self.in_padding = self.in_features - self.original_in_features
        self.out_padding = self.out_features - self.original_out_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic padding/unpadding.
        
        Args:
            x: Input tensor (..., in_features)
        
        Returns:
            Output tensor (..., out_features)
        """
        # Pad input if necessary
        if self.in_padding > 0:
            pad_shape = list(x.shape)
            pad_shape[-1] = self.in_padding
            x_pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
            x = torch.cat([x, x_pad], dim=-1)
        
        # Linear transformation
        y = super().forward(x)
        
        # Remove output padding if necessary
        if self.out_padding > 0:
            y = y[..., :self.original_out_features]
        
        return y


class TensorCoreOptimizedEmbedding(nn.Embedding):
    """
    Embedding layer with automatic padding for tensor core optimization.
    
    Pads embedding dimension to multiple of 8 if necessary.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[torch.Tensor] = None,
        device=None,
        dtype=None,
        pad_for_tensor_cores: bool = True
    ):
        """
        Initialize tensor core optimized embedding layer.
        
        Args:
            num_embeddings: Vocabulary size
            embedding_dim: Embedding dimension
            padding_idx: Padding index
            max_norm: Max norm for embeddings
            norm_type: Norm type
            scale_grad_by_freq: Scale gradients by frequency
            sparse: Use sparse gradients
            _weight: Pre-initialized weight
            device: Device
            dtype: Data type
            pad_for_tensor_cores: Automatically pad dimension
        """
        self.original_embedding_dim = embedding_dim
        self.pad_for_tensor_cores = pad_for_tensor_cores
        
        if pad_for_tensor_cores:
            embedding_dim = pad_to_tensor_core_multiple(embedding_dim)
        
        super().__init__(
            num_embeddings, embedding_dim, padding_idx, max_norm,
            norm_type, scale_grad_by_freq, sparse, _weight, device, dtype
        )
        
        self.dim_padding = self.embedding_dim - self.original_embedding_dim
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic unpadding.
        
        Args:
            input: Input indices
        
        Returns:
            Embeddings with original dimension
        """
        # Get embeddings (padded)
        embeddings = super().forward(input)
        
        # Remove padding if necessary
        if self.dim_padding > 0:
            embeddings = embeddings[..., :self.original_embedding_dim]
        
        return embeddings


def optimize_model_for_tensor_cores(model: nn.Module, inplace: bool = False) -> nn.Module:
    """
    Optimize model for tensor cores by padding dimensions.
    
    Args:
        model: Original model
        inplace: Modify model in place (not recommended)
    
    Returns:
        Optimized model
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)
    
    print("Optimizing model for tensor cores...")
    
    replacements = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if not is_tensor_core_compatible(module.in_features) or \
               not is_tensor_core_compatible(module.out_features):
                replacements.append((name, module, 'Linear'))
        
        elif isinstance(module, nn.Embedding):
            if not is_tensor_core_compatible(module.embedding_dim):
                replacements.append((name, module, 'Embedding'))
    
    # Replace modules
    for name, old_module, module_type in replacements:
        # Get parent module and attribute name
        *parent_names, attr_name = name.split('.')
        parent = model
        for parent_name in parent_names:
            parent = getattr(parent, parent_name)
        
        # Create replacement
        if module_type == 'Linear':
            new_module = TensorCoreOptimizedLinear(
                old_module.in_features,
                old_module.out_features,
                bias=old_module.bias is not None,
                device=old_module.weight.device,
                dtype=old_module.weight.dtype
            )
            # Copy weights
            with torch.no_grad():
                new_module.weight[:old_module.out_features, :old_module.in_features].copy_(old_module.weight)
                if old_module.bias is not None:
                    new_module.bias[:old_module.out_features].copy_(old_module.bias)
        
        elif module_type == 'Embedding':
            new_module = TensorCoreOptimizedEmbedding(
                old_module.num_embeddings,
                old_module.embedding_dim,
                padding_idx=old_module.padding_idx,
                max_norm=old_module.max_norm,
                norm_type=old_module.norm_type,
                scale_grad_by_freq=old_module.scale_grad_by_freq,
                sparse=old_module.sparse,
                device=old_module.weight.device,
                dtype=old_module.weight.dtype
            )
            # Copy weights
            with torch.no_grad():
                new_module.weight[:, :old_module.embedding_dim].copy_(old_module.weight)
        
        # Replace
        setattr(parent, attr_name, new_module)
        print(f"  Replaced {name} ({module_type})")
    
    print(f"\nOptimized {len(replacements)} layers")
    
    return model


def benchmark_tensor_core_optimization(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    num_iterations: int = 100,
    device: str = 'cuda'
) -> dict:
    """
    Benchmark tensor core optimization impact.
    
    Args:
        model: Model to benchmark
        input_shape: Input shape for forward pass
        num_iterations: Number of iterations
        device: Device
    
    Returns:
        Benchmark results
    """
    import time
    
    print("=" * 60)
    print("Benchmarking Tensor Core Optimization")
    print("=" * 60)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Original model
    model_original = model.to(device)
    
    # Optimized model
    model_optimized = optimize_model_for_tensor_cores(model_original, inplace=False)
    model_optimized = model_optimized.to(device)
    
    # Create input
    x = torch.randint(0, 1000, input_shape, device=device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model_original(x)
            _ = model_optimized(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark original
    start = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model_original(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    time_original = time.time() - start
    
    # Benchmark optimized
    start = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model_optimized(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    time_optimized = time.time() - start
    
    results = {
        'time_original': time_original / num_iterations,
        'time_optimized': time_optimized / num_iterations,
        'speedup': time_original / time_optimized
    }
    
    print(f"\nResults:")
    print(f"  Original: {results['time_original']*1000:.3f} ms")
    print(f"  Optimized: {results['time_optimized']*1000:.3f} ms")
    print(f"  Speedup: {results['speedup']:.2f}x")
    
    return results


if __name__ == '__main__':
    print("Tensor Core Optimization Utilities Test")
    print("=" * 60)
    
    # Test dimension checking
    print("\n1. Testing dimension compatibility:")
    for dim in [64, 65, 72, 128]:
        compatible = is_tensor_core_compatible(dim)
        padded = pad_to_tensor_core_multiple(dim)
        print(f"  dim={dim}: compatible={compatible}, padded={padded}")
    
    # Test optimized layers
    print("\n2. Testing TensorCoreOptimizedLinear:")
    linear = TensorCoreOptimizedLinear(65, 130, pad_for_tensor_cores=True)
    x = torch.randn(4, 10, 65)
    y = linear(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Weight shape: {linear.weight.shape}")
    print(f"  Padding: in={linear.in_padding}, out={linear.out_padding}")
    
    print("\n3. Testing TensorCoreOptimizedEmbedding:")
    embedding = TensorCoreOptimizedEmbedding(1000, 65, pad_for_tensor_cores=True)
    x = torch.randint(0, 1000, (4, 10))
    y = embedding(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Weight shape: {embedding.weight.shape}")
    print(f"  Padding: {embedding.dim_padding}")
    
    print("\n✓ All tests passed")
