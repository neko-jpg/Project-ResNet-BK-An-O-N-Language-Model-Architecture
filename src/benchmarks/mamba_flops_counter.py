"""
FLOPs and Memory Counter for Mamba Baseline

Comprehensive measurement of computational cost and memory usage for Mamba models.
Ensures fair comparison with ResNet-BK by counting all operations including:
- State updates
- Gating operations
- Normalization
- All buffers, activations, and optimizer states

Requirements: 11.5, 11.6, 11.7, 11.8, 11.10
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional
import json


@dataclass
class FLOPsCount:
    """Container for FLOPs measurements."""
    forward: int = 0
    backward: int = 0
    optimizer: int = 0
    
    @property
    def total(self) -> int:
        """Total FLOPs (forward + backward + optimizer)."""
        return self.forward + self.backward + self.optimizer
    
    def __add__(self, other):
        """Add two FLOPsCount objects."""
        return FLOPsCount(
            forward=self.forward + other.forward,
            backward=self.backward + other.backward,
            optimizer=self.optimizer + other.optimizer
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'forward': self.forward,
            'backward': self.backward,
            'optimizer': self.optimizer,
            'total': self.total
        }


@dataclass
class MemoryUsage:
    """Container for memory measurements."""
    parameters: int = 0  # Model parameters
    activations: int = 0  # Forward activations
    gradients: int = 0  # Gradient tensors
    optimizer_states: int = 0  # Optimizer state (momentum, variance)
    buffers: int = 0  # Other buffers (running stats, etc.)
    
    @property
    def total(self) -> int:
        """Total memory usage."""
        return self.parameters + self.activations + self.gradients + self.optimizer_states + self.buffers
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'parameters': self.parameters,
            'activations': self.activations,
            'gradients': self.gradients,
            'optimizer_states': self.optimizer_states,
            'buffers': self.buffers,
            'total': self.total
        }


class MambaFLOPsCounter:
    """
    Comprehensive FLOPs counter for Mamba models.
    
    Counts all operations including:
    - SSM state updates (discretization, scan)
    - Convolution operations
    - Linear projections
    - Gating operations (SiLU)
    - Normalization (LayerNorm)
    - Embeddings
    
    Usage:
        counter = MambaFLOPsCounter(model, batch_size=32, seq_len=128)
        flops = counter.count_total_flops()
        memory = counter.count_memory_usage()
    """
    
    def __init__(self, model: nn.Module, batch_size: int, seq_len: int):
        """
        Initialize FLOPs counter.
        
        Args:
            model: Mamba model
            batch_size: batch size for FLOPs calculation
            seq_len: sequence length
        """
        self.model = model
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        # Extract model configuration
        self.config = model.config
        self.d_model = self.config.d_model
        self.n_layers = self.config.n_layers
        self.vocab_size = self.config.vocab_size
        self.d_state = self.config.d_state
        self.d_inner = int(self.config.expand * self.d_model)
        self.d_conv = self.config.d_conv
        
        # Component-wise breakdown
        self.component_flops: Dict[str, FLOPsCount] = {}
    
    def count_linear_flops(self, in_features: int, out_features: int, has_bias: bool = False) -> FLOPsCount:
        """
        Count FLOPs for a linear layer.
        
        Args:
            in_features: input dimension
            out_features: output dimension
            has_bias: whether layer has bias
        
        Returns:
            FLOPsCount for this linear layer
        """
        B = self.batch_size
        L = self.seq_len
        
        # Forward: (B*L, in) @ (in, out) = B*L*in*out multiply-adds
        forward_flops = B * L * in_features * out_features * 2
        if has_bias:
            forward_flops += B * L * out_features  # Bias addition
        
        # Backward: gradient w.r.t. input + gradient w.r.t. weights
        backward_flops = 2 * forward_flops
        
        return FLOPsCount(forward=forward_flops, backward=backward_flops)
    
    def count_conv1d_flops(self, in_channels: int, out_channels: int, kernel_size: int, groups: int = 1) -> FLOPsCount:
        """
        Count FLOPs for 1D convolution.
        
        Args:
            in_channels: input channels
            out_channels: output channels
            kernel_size: convolution kernel size
            groups: number of groups (for depthwise conv)
        
        Returns:
            FLOPsCount for this conv layer
        """
        B = self.batch_size
        L = self.seq_len
        
        # Forward: B * L * out_channels * (in_channels/groups) * kernel_size * 2 (multiply-add)
        forward_flops = B * L * out_channels * (in_channels // groups) * kernel_size * 2
        
        # Backward: approximately 2× forward
        backward_flops = 2 * forward_flops
        
        return FLOPsCount(forward=forward_flops, backward=backward_flops)
    
    def count_ssm_flops(self) -> FLOPsCount:
        """
        Count FLOPs for SSM (State Space Model) operations.
        
        SSM operations include:
        1. Discretization: computing A_bar, B_bar from continuous parameters
        2. Selective scan: h_t = A_t * h_{t-1} + B_t * x_t
        3. Output projection: y_t = C_t * h_t
        
        Returns:
            FLOPsCount for SSM operations
        """
        B = self.batch_size
        L = self.seq_len
        D = self.d_inner
        N = self.d_state
        
        # 1. Discretization
        # - exp(Δ * A): L * D * N exponentials + multiplications
        discretization_flops = L * D * N * 3  # exp + multiply
        
        # - Δ * B: L * D * N multiplications
        discretization_flops += L * D * N
        
        # 2. Selective scan (sequential)
        # For each time step t:
        # - h_t = A_t * h_{t-1}: D * N multiplications
        # - h_t += B_t * x_t: D * N multiply-adds
        scan_flops_per_step = D * N * 2 + D * N * 2  # A*h + B*x
        scan_flops = L * scan_flops_per_step
        
        # 3. Output projection: y_t = C_t * h_t
        # - For each time step: D * N multiplications
        output_flops = L * D * N * 2
        
        # 4. Skip connection: y += D * x
        skip_flops = L * D
        
        forward_flops = B * (discretization_flops + scan_flops + output_flops + skip_flops)
        
        # Backward: gradient through scan is complex, approximately 3× forward
        # (gradient w.r.t. A, B, C, x, and state)
        backward_flops = 3 * forward_flops
        
        return FLOPsCount(forward=forward_flops, backward=backward_flops)
    
    def count_activation_flops(self, activation: str, num_elements: int) -> FLOPsCount:
        """
        Count FLOPs for activation functions.
        
        Args:
            activation: activation type ('silu', 'gelu', 'relu', etc.)
            num_elements: number of elements
        
        Returns:
            FLOPsCount for activation
        """
        B = self.batch_size
        L = self.seq_len
        
        if activation == 'silu':
            # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
            # exp + division + multiply: ~5 ops per element
            ops_per_element = 5
        elif activation == 'gelu':
            # GELU: more complex, ~8 ops per element
            ops_per_element = 8
        elif activation == 'relu':
            # ReLU: max(0, x), ~1 op per element
            ops_per_element = 1
        else:
            ops_per_element = 3  # Default
        
        forward_flops = B * L * num_elements * ops_per_element
        
        # Backward: gradient computation, similar complexity
        backward_flops = forward_flops
        
        return FLOPsCount(forward=forward_flops, backward=backward_flops)
    
    def count_layernorm_flops(self, normalized_shape: int) -> FLOPsCount:
        """
        Count FLOPs for LayerNorm.
        
        Operations: mean, variance, normalize, scale, shift
        
        Args:
            normalized_shape: dimension to normalize over
        
        Returns:
            FLOPsCount for LayerNorm
        """
        B = self.batch_size
        L = self.seq_len
        D = normalized_shape
        
        # Forward: mean (D ops) + variance (D ops) + normalize (D ops) + scale (D ops) + shift (D ops)
        forward_flops = B * L * D * 5
        
        # Backward: similar complexity
        backward_flops = B * L * D * 5
        
        return FLOPsCount(forward=forward_flops, backward=backward_flops)
    
    def count_embedding_flops(self) -> FLOPsCount:
        """
        Count FLOPs for embedding layer.
        
        Embedding lookup is essentially free (just indexing).
        """
        # Embedding lookup: negligible FLOPs (just memory access)
        return FLOPsCount(forward=0, backward=0)
    
    def count_mamba_block_flops(self) -> FLOPsCount:
        """
        Count FLOPs for a single Mamba block.
        
        Mamba block includes:
        1. LayerNorm
        2. Input projection (Linear: d_model -> 2*d_inner)
        3. Convolution (depthwise, kernel_size=d_conv)
        4. SiLU activation
        5. SSM operations
        6. Gating (SiLU)
        7. Output projection (Linear: d_inner -> d_model)
        
        Returns:
            FLOPsCount for one Mamba block
        """
        total = FLOPsCount()
        
        # 1. LayerNorm
        ln_flops = self.count_layernorm_flops(self.d_model)
        total = total + ln_flops
        
        # 2. Input projection
        in_proj_flops = self.count_linear_flops(self.d_model, 2 * self.d_inner, has_bias=False)
        total = total + in_proj_flops
        
        # 3. Convolution (depthwise)
        conv_flops = self.count_conv1d_flops(
            self.d_inner, self.d_inner, self.d_conv, groups=self.d_inner
        )
        total = total + conv_flops
        
        # 4. SiLU activation (after conv)
        silu1_flops = self.count_activation_flops('silu', self.d_inner)
        total = total + silu1_flops
        
        # 5. SSM operations
        ssm_flops = self.count_ssm_flops()
        total = total + ssm_flops
        
        # 6. Gating (SiLU on z branch + multiply)
        silu2_flops = self.count_activation_flops('silu', self.d_inner)
        gating_multiply = FLOPsCount(
            forward=self.batch_size * self.seq_len * self.d_inner,
            backward=self.batch_size * self.seq_len * self.d_inner
        )
        total = total + silu2_flops + gating_multiply
        
        # 7. Output projection
        out_proj_flops = self.count_linear_flops(self.d_inner, self.d_model, has_bias=False)
        total = total + out_proj_flops
        
        return total
    
    def count_forward_flops(self) -> FLOPsCount:
        """
        Count total forward pass FLOPs.
        
        Returns:
            FLOPsCount with forward FLOPs only
        """
        total = FLOPsCount()
        
        # Embeddings (negligible)
        embedding_flops = self.count_embedding_flops()
        total.forward += embedding_flops.forward
        self.component_flops['embedding'] = FLOPsCount(forward=embedding_flops.forward)
        
        # Mamba blocks
        for layer_idx in range(self.n_layers):
            block_flops = self.count_mamba_block_flops()
            total.forward += block_flops.forward
            self.component_flops[f'layer_{layer_idx}'] = FLOPsCount(forward=block_flops.forward)
        
        # Final LayerNorm
        final_ln_flops = self.count_layernorm_flops(self.d_model)
        total.forward += final_ln_flops.forward
        self.component_flops['final_layernorm'] = FLOPsCount(forward=final_ln_flops.forward)
        
        # LM Head
        lm_head_flops = self.count_linear_flops(self.d_model, self.vocab_size, has_bias=False)
        total.forward += lm_head_flops.forward
        self.component_flops['lm_head'] = FLOPsCount(forward=lm_head_flops.forward)
        
        return total
    
    def count_backward_flops(self) -> FLOPsCount:
        """
        Count total backward pass FLOPs.
        
        Returns:
            FLOPsCount with backward FLOPs only
        """
        total = FLOPsCount()
        
        # LM Head backward
        lm_head_flops = self.count_linear_flops(self.d_model, self.vocab_size, has_bias=False)
        total.backward += lm_head_flops.backward
        
        # Final LayerNorm backward
        final_ln_flops = self.count_layernorm_flops(self.d_model)
        total.backward += final_ln_flops.backward
        
        # Mamba blocks backward
        for _ in range(self.n_layers):
            block_flops = self.count_mamba_block_flops()
            total.backward += block_flops.backward
        
        # Embeddings backward (negligible)
        embedding_flops = self.count_embedding_flops()
        total.backward += embedding_flops.backward
        
        return total
    
    def count_optimizer_flops(self, optimizer_name: str = 'adamw') -> FLOPsCount:
        """
        Count FLOPs for optimizer step.
        
        Args:
            optimizer_name: 'sgd', 'adam', or 'adamw'
        
        Returns:
            FLOPsCount with optimizer FLOPs only
        """
        # Count total parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        if optimizer_name.lower() == 'sgd':
            # SGD: param = param - lr * grad
            optimizer_flops = num_params * 2
        elif optimizer_name.lower() in ['adam', 'adamw']:
            # Adam/AdamW: momentum, variance, bias correction, update
            optimizer_flops = num_params * 15
        else:
            optimizer_flops = num_params * 2
        
        return FLOPsCount(optimizer=optimizer_flops)
    
    def count_total_flops(self, optimizer_name: str = 'adamw') -> FLOPsCount:
        """
        Count total FLOPs for one training step.
        
        Args:
            optimizer_name: optimizer type
        
        Returns:
            FLOPsCount with forward, backward, and optimizer FLOPs
        """
        forward = self.count_forward_flops()
        backward = self.count_backward_flops()
        optimizer = self.count_optimizer_flops(optimizer_name)
        
        return FLOPsCount(
            forward=forward.forward,
            backward=backward.backward,
            optimizer=optimizer.optimizer
        )
    
    def count_memory_usage(self, optimizer_name: str = 'adamw', dtype: torch.dtype = torch.float32) -> MemoryUsage:
        """
        Count memory usage for training.
        
        Includes:
        - Model parameters
        - Activations (forward pass)
        - Gradients
        - Optimizer states
        - Buffers
        
        Args:
            optimizer_name: optimizer type
            dtype: data type for calculations
        
        Returns:
            MemoryUsage object
        """
        bytes_per_element = 4 if dtype == torch.float32 else 2  # FP32 or FP16
        
        B = self.batch_size
        L = self.seq_len
        
        # 1. Parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        parameters_memory = num_params * bytes_per_element
        
        # 2. Activations (forward pass)
        # - Embeddings: B * L * d_model
        # - Per layer: B * L * (d_model + 2*d_inner + d_inner + d_state*d_inner)
        # - Final: B * L * d_model + B * L * vocab_size
        activations_per_layer = B * L * (self.d_model + 3 * self.d_inner + self.d_state * self.d_inner)
        activations_memory = (
            B * L * self.d_model +  # Embeddings
            self.n_layers * activations_per_layer +  # Layers
            B * L * self.d_model +  # Final norm
            B * L * self.vocab_size  # LM head
        ) * bytes_per_element
        
        # 3. Gradients (same size as parameters)
        gradients_memory = parameters_memory
        
        # 4. Optimizer states
        if optimizer_name.lower() in ['adam', 'adamw']:
            # Adam: momentum (m) + variance (v) = 2× parameters
            optimizer_states_memory = 2 * parameters_memory
        else:
            # SGD: no extra states (or just momentum = 1× parameters)
            optimizer_states_memory = parameters_memory
        
        # 5. Buffers (A_log, D, etc.)
        # Approximate as 5% of parameters
        buffers_memory = int(0.05 * parameters_memory)
        
        return MemoryUsage(
            parameters=parameters_memory,
            activations=activations_memory,
            gradients=gradients_memory,
            optimizer_states=optimizer_states_memory,
            buffers=buffers_memory
        )
    
    def print_summary(self, optimizer_name: str = 'adamw'):
        """
        Print human-readable summary.
        
        Args:
            optimizer_name: optimizer type
        """
        flops = self.count_total_flops(optimizer_name)
        memory = self.count_memory_usage(optimizer_name)
        
        print("=" * 70)
        print(f"Mamba FLOPs and Memory Summary")
        print("=" * 70)
        print(f"Model: Mamba (d={self.d_model}, L={self.n_layers}, N={self.seq_len})")
        print(f"Batch Size: {self.batch_size}")
        print(f"SSM State Dimension: {self.d_state}")
        print(f"Inner Dimension: {self.d_inner}")
        print("-" * 70)
        print(f"FLOPs:")
        print(f"  Forward Pass:   {flops.forward:>15,} FLOPs ({flops.forward/1e9:.3f} GFLOPs)")
        print(f"  Backward Pass:  {flops.backward:>15,} FLOPs ({flops.backward/1e9:.3f} GFLOPs)")
        print(f"  Optimizer Step: {flops.optimizer:>15,} FLOPs ({flops.optimizer/1e9:.3f} GFLOPs)")
        print(f"  Total per Step: {flops.total:>15,} FLOPs ({flops.total/1e9:.3f} GFLOPs)")
        print("-" * 70)
        print(f"Memory Usage:")
        print(f"  Parameters:      {memory.parameters:>15,} bytes ({memory.parameters/1e6:.2f} MB)")
        print(f"  Activations:     {memory.activations:>15,} bytes ({memory.activations/1e6:.2f} MB)")
        print(f"  Gradients:       {memory.gradients:>15,} bytes ({memory.gradients/1e6:.2f} MB)")
        print(f"  Optimizer States:{memory.optimizer_states:>15,} bytes ({memory.optimizer_states/1e6:.2f} MB)")
        print(f"  Buffers:         {memory.buffers:>15,} bytes ({memory.buffers/1e6:.2f} MB)")
        print(f"  Total:           {memory.total:>15,} bytes ({memory.total/1e6:.2f} MB)")
        print("=" * 70)
    
    def save_to_json(self, filepath: str, optimizer_name: str = 'adamw'):
        """
        Save measurements to JSON file.
        
        Args:
            filepath: output JSON file path
            optimizer_name: optimizer type
        """
        flops = self.count_total_flops(optimizer_name)
        memory = self.count_memory_usage(optimizer_name)
        
        data = {
            'model_config': {
                'd_model': self.d_model,
                'n_layers': self.n_layers,
                'seq_len': self.seq_len,
                'batch_size': self.batch_size,
                'vocab_size': self.vocab_size,
                'd_state': self.d_state,
                'd_inner': self.d_inner,
                'd_conv': self.d_conv,
            },
            'flops': flops.to_dict(),
            'memory': memory.to_dict(),
            'optimizer': optimizer_name,
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Measurements saved to {filepath}")


if __name__ == '__main__':
    # Test Mamba FLOPs counter
    from src.models.mamba_baseline import MambaLM, MambaConfig
    
    config = MambaConfig(
        vocab_size=30000,
        d_model=256,
        n_layers=8,
        max_seq_len=2048
    )
    
    model = MambaLM(config)
    
    # Count FLOPs and memory
    counter = MambaFLOPsCounter(model, batch_size=32, seq_len=128)
    counter.print_summary()
    
    # Save to JSON
    counter.save_to_json('mamba_measurements.json')
