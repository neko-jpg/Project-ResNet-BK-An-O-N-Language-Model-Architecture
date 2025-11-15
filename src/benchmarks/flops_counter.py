"""
FLOPs Counter Infrastructure
Comprehensive FLOPs counting for ResNet-BK model components.

Tracks forward and backward FLOPs separately for:
- BK-Core (theta/phi recursions)
- MoE (expert computation and routing)
- Linear layers
- Optimizer steps
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional
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
    
    def __mul__(self, scalar):
        """Multiply FLOPs by scalar."""
        return FLOPsCount(
            forward=int(self.forward * scalar),
            backward=int(self.backward * scalar),
            optimizer=int(self.optimizer * scalar)
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'forward': self.forward,
            'backward': self.backward,
            'optimizer': self.optimizer,
            'total': self.total
        }


class FLOPsCounter:
    """
    Comprehensive FLOPs counter for ResNet-BK model.
    
    Counts FLOPs for:
    - BK-Core: theta/phi recursions, complex arithmetic
    - MoE: expert computation, routing, gating
    - Linear layers: matrix multiplications
    - Embeddings: lookup and addition
    - Optimizer: parameter updates (AdamW)
    
    Usage:
        counter = FLOPsCounter(model, batch_size=32, seq_len=128)
        flops = counter.count_forward_flops()
        flops_backward = counter.count_backward_flops()
        flops_total = counter.count_total_flops()
    """
    
    def __init__(self, model: nn.Module, batch_size: int, seq_len: int):
        """
        Initialize FLOPs counter.
        
        Args:
            model: ResNet-BK model
            batch_size: batch size for FLOPs calculation
            seq_len: sequence length
        """
        self.model = model
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        # Extract model dimensions
        # Handle both ConfigurableResNetBK (has .model) and LanguageModel (direct)
        if hasattr(model, 'model'):
            # ConfigurableResNetBK wrapper
            inner_model = model.model
            self.d_model = inner_model.d_model
            self.n_layers = len(inner_model.blocks)
            self.vocab_size = inner_model.lm_head.out_features
            first_block = inner_model.blocks[0]
        else:
            # Direct LanguageModel
            self.d_model = model.d_model
            self.n_layers = len(model.blocks)
            self.vocab_size = model.lm_head.out_features
            first_block = model.blocks[0]
        
        # Extract MoE configuration
        self.num_experts = first_block.bk_layer.moe_ffn.num_experts
        self.top_k = first_block.bk_layer.moe_ffn.top_k
        
        # Component-wise FLOPs breakdown
        self.component_flops: Dict[str, FLOPsCount] = {}
    
    def count_bk_core_flops(self) -> FLOPsCount:
        """
        Count FLOPs for BK-Core computation.
        
        Forward pass:
        - Theta recursion: N iterations, each with 2 complex multiplies + 1 complex add
        - Phi recursion: N iterations, each with 2 complex multiplies + 1 complex add
        - Final division: N complex divisions
        - Complex multiply: 6 real ops (4 muls, 2 adds)
        - Complex add: 2 real ops
        - Complex divide: 6 real ops (approximation)
        
        Backward pass:
        - Gradient computation: O(N) operations
        - G² computation: N complex multiplies
        - Gradient blending: N operations
        """
        N = self.seq_len
        B = self.batch_size
        
        # Forward FLOPs per sequence
        # Theta recursion: N iterations × (2 complex_mul + 1 complex_add)
        theta_flops = N * (2 * 6 + 2)  # 14 ops per iteration
        
        # Phi recursion: N iterations × (2 complex_mul + 1 complex_add)
        phi_flops = N * (2 * 6 + 2)  # 14 ops per iteration
        
        # Final division: N complex divisions
        division_flops = N * 6
        
        # Real/imag extraction: negligible
        
        forward_flops_per_seq = theta_flops + phi_flops + division_flops
        forward_flops = B * forward_flops_per_seq
        
        # Backward FLOPs per sequence
        # G² computation: N complex multiplies
        g_square_flops = N * 6
        
        # Gradient computation (theoretical + hypothesis-7): 2 × N operations
        grad_computation_flops = 2 * N * 10  # Approximate: complex ops + blending
        
        backward_flops_per_seq = g_square_flops + grad_computation_flops
        backward_flops = B * backward_flops_per_seq
        
        return FLOPsCount(forward=forward_flops, backward=backward_flops)
    
    def count_moe_flops(self) -> FLOPsCount:
        """
        Count FLOPs for MoE layer.
        
        Forward pass:
        - Gating network: Linear(D → E)
        - Expert computation: top_k experts × [Linear(D → 2D) + ReLU + Linear(2D → D)]
        - Routing: softmax over E experts
        
        Backward pass:
        - Gradient through experts: same as forward
        - Gradient through gating: same as forward
        """
        N = self.seq_len
        B = self.batch_size
        D = self.d_model
        E = self.num_experts
        K = self.top_k
        
        # Forward FLOPs
        # Gating network: (B*N, D) @ (D, E) = B*N*D*E multiplies
        gating_flops = B * N * D * E * 2  # 2 ops per multiply-add
        
        # Softmax: B*N*E operations (exp + sum + divide)
        softmax_flops = B * N * E * 3
        
        # Expert computation (per token, K experts selected)
        # Linear(D → 2D): D * 2D = 2D² operations
        # ReLU: 2D operations
        # Linear(2D → D): 2D * D = 2D² operations
        expert_flops_per_token = K * (2 * D * D * 2 + 2 * D + 2 * D * D * 2)
        expert_flops_per_token = K * (8 * D * D + 2 * D)
        expert_flops = B * N * expert_flops_per_token
        
        forward_flops = gating_flops + softmax_flops + expert_flops
        
        # Backward FLOPs (approximately 2× forward for standard backprop)
        backward_flops = 2 * forward_flops
        
        return FLOPsCount(forward=forward_flops, backward=backward_flops)
    
    def count_linear_flops(self, in_features: int, out_features: int) -> FLOPsCount:
        """
        Count FLOPs for a single linear layer.
        
        Args:
            in_features: input dimension
            out_features: output dimension
        
        Returns:
            FLOPsCount for this linear layer
        """
        B = self.batch_size
        N = self.seq_len
        
        # Forward: (B*N, in_features) @ (in_features, out_features)
        forward_flops = B * N * in_features * out_features * 2  # multiply-add
        
        # Backward: gradient w.r.t. input + gradient w.r.t. weights
        # grad_input: (B*N, out_features) @ (out_features, in_features)
        # grad_weight: (in_features, B*N) @ (B*N, out_features)
        backward_flops = 2 * forward_flops
        
        return FLOPsCount(forward=forward_flops, backward=backward_flops)
    
    def count_embedding_flops(self) -> FLOPsCount:
        """
        Count FLOPs for embedding layers.
        
        Token embedding + position embedding + addition.
        """
        B = self.batch_size
        N = self.seq_len
        D = self.d_model
        
        # Embedding lookup: negligible (just indexing)
        # Addition: B*N*D operations
        forward_flops = B * N * D
        
        # Backward: gradient propagation through embeddings
        backward_flops = B * N * D
        
        return FLOPsCount(forward=forward_flops, backward=backward_flops)
    
    def count_layernorm_flops(self) -> FLOPsCount:
        """
        Count FLOPs for LayerNorm.
        
        Operations: mean, variance, normalize, scale, shift
        """
        B = self.batch_size
        N = self.seq_len
        D = self.d_model
        
        # Forward: mean (D ops) + variance (D ops) + normalize (D ops) + scale (D ops) + shift (D ops)
        forward_flops = B * N * D * 5
        
        # Backward: similar complexity
        backward_flops = B * N * D * 5
        
        return FLOPsCount(forward=forward_flops, backward=backward_flops)
    
    def count_output_projection_flops(self) -> FLOPsCount:
        """
        Count FLOPs for BK-Core output projection.
        
        Linear(2 → D) projection from [real(G_ii), imag(G_ii)] to d_model.
        """
        return self.count_linear_flops(2, self.d_model)
    
    def count_lm_head_flops(self) -> FLOPsCount:
        """
        Count FLOPs for language modeling head.
        
        Linear(D → vocab_size) projection.
        """
        return self.count_linear_flops(self.d_model, self.vocab_size)
    
    def count_forward_flops(self) -> FLOPsCount:
        """
        Count total forward pass FLOPs.
        
        Returns:
            FLOPsCount with forward FLOPs only
        """
        total = FLOPsCount()
        
        # Embeddings
        embedding_flops = self.count_embedding_flops()
        total.forward += embedding_flops.forward
        self.component_flops['embedding'] = FLOPsCount(forward=embedding_flops.forward)
        
        # Per-layer computation
        for layer_idx in range(self.n_layers):
            layer_total = FLOPsCount()
            
            # LayerNorm
            ln_flops = self.count_layernorm_flops()
            layer_total.forward += ln_flops.forward
            
            # MoE
            moe_flops = self.count_moe_flops()
            layer_total.forward += moe_flops.forward
            
            # BK-Core
            bk_flops = self.count_bk_core_flops()
            layer_total.forward += bk_flops.forward
            
            # Output projection
            proj_flops = self.count_output_projection_flops()
            layer_total.forward += proj_flops.forward
            
            self.component_flops[f'layer_{layer_idx}'] = FLOPsCount(forward=layer_total.forward)
            total.forward += layer_total.forward
        
        # Final LayerNorm
        final_ln_flops = self.count_layernorm_flops()
        total.forward += final_ln_flops.forward
        self.component_flops['final_layernorm'] = FLOPsCount(forward=final_ln_flops.forward)
        
        # LM Head
        lm_head_flops = self.count_lm_head_flops()
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
        lm_head_flops = self.count_lm_head_flops()
        total.backward += lm_head_flops.backward
        
        # Final LayerNorm backward
        final_ln_flops = self.count_layernorm_flops()
        total.backward += final_ln_flops.backward
        
        # Per-layer backward
        for layer_idx in range(self.n_layers):
            # Output projection backward
            proj_flops = self.count_output_projection_flops()
            total.backward += proj_flops.backward
            
            # BK-Core backward (analytic gradient)
            bk_flops = self.count_bk_core_flops()
            total.backward += bk_flops.backward
            
            # MoE backward
            moe_flops = self.count_moe_flops()
            total.backward += moe_flops.backward
            
            # LayerNorm backward
            ln_flops = self.count_layernorm_flops()
            total.backward += ln_flops.backward
        
        # Embeddings backward
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
            optimizer_flops = num_params * 2  # 1 multiply, 1 add
        elif optimizer_name.lower() in ['adam', 'adamw']:
            # Adam/AdamW: 
            # - momentum update: m = beta1 * m + (1-beta1) * grad (3 ops)
            # - variance update: v = beta2 * v + (1-beta2) * grad² (4 ops)
            # - bias correction: m_hat, v_hat (4 ops)
            # - parameter update: param = param - lr * m_hat / (sqrt(v_hat) + eps) (4 ops)
            optimizer_flops = num_params * 15  # Approximate
        else:
            optimizer_flops = num_params * 2  # Default to SGD
        
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
    
    def get_breakdown(self) -> Dict[str, Dict]:
        """
        Get detailed FLOPs breakdown by component.
        
        Returns:
            Dictionary with component-wise FLOPs
        """
        # Ensure we've counted FLOPs
        if not self.component_flops:
            self.count_forward_flops()
        
        breakdown = {}
        for component, flops in self.component_flops.items():
            breakdown[component] = flops.to_dict()
        
        return breakdown
    
    def print_summary(self, optimizer_name: str = 'adamw'):
        """
        Print human-readable FLOPs summary.
        
        Args:
            optimizer_name: optimizer type
        """
        total = self.count_total_flops(optimizer_name)
        
        print("=" * 70)
        print(f"FLOPs Counter Summary")
        print("=" * 70)
        print(f"Model: ResNet-BK (d={self.d_model}, L={self.n_layers}, N={self.seq_len})")
        print(f"Batch Size: {self.batch_size}")
        print(f"MoE: {self.num_experts} experts, top-{self.top_k}")
        print("-" * 70)
        print(f"Forward Pass:   {total.forward:>15,} FLOPs ({total.forward/1e9:.3f} GFLOPs)")
        print(f"Backward Pass:  {total.backward:>15,} FLOPs ({total.backward/1e9:.3f} GFLOPs)")
        print(f"Optimizer Step: {total.optimizer:>15,} FLOPs ({total.optimizer/1e9:.3f} GFLOPs)")
        print("-" * 70)
        print(f"Total per Step: {total.total:>15,} FLOPs ({total.total/1e9:.3f} GFLOPs)")
        print("=" * 70)
        
        # Component breakdown
        print("\nComponent Breakdown (Forward Pass):")
        print("-" * 70)
        breakdown = self.get_breakdown()
        for component, flops in sorted(breakdown.items()):
            if flops['forward'] > 0:
                pct = 100 * flops['forward'] / total.forward
                print(f"  {component:20s}: {flops['forward']:>12,} FLOPs ({pct:5.1f}%)")
        print("=" * 70)
    
    def save_to_json(self, filepath: str, optimizer_name: str = 'adamw'):
        """
        Save FLOPs count to JSON file.
        
        Args:
            filepath: output JSON file path
            optimizer_name: optimizer type
        """
        total = self.count_total_flops(optimizer_name)
        breakdown = self.get_breakdown()
        
        data = {
            'model_config': {
                'd_model': self.d_model,
                'n_layers': self.n_layers,
                'seq_len': self.seq_len,
                'batch_size': self.batch_size,
                'vocab_size': self.vocab_size,
                'num_experts': self.num_experts,
                'top_k': self.top_k,
            },
            'total_flops': total.to_dict(),
            'component_breakdown': breakdown,
            'optimizer': optimizer_name,
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"FLOPs count saved to {filepath}")


def compare_models(model1: nn.Module, model2: nn.Module, 
                   batch_size: int, seq_len: int,
                   model1_name: str = "Model 1",
                   model2_name: str = "Model 2") -> Dict:
    """
    Compare FLOPs between two models.
    
    Args:
        model1: first model
        model2: second model (baseline)
        batch_size: batch size
        seq_len: sequence length
        model1_name: name for first model
        model2_name: name for second model (baseline)
    
    Returns:
        Dictionary with comparison results
    """
    counter1 = FLOPsCounter(model1, batch_size, seq_len)
    counter2 = FLOPsCounter(model2, batch_size, seq_len)
    
    flops1 = counter1.count_total_flops()
    flops2 = counter2.count_total_flops()
    
    # Calculate speedup
    speedup_forward = flops2.forward / flops1.forward if flops1.forward > 0 else 0
    speedup_backward = flops2.backward / flops1.backward if flops1.backward > 0 else 0
    speedup_total = flops2.total / flops1.total if flops1.total > 0 else 0
    
    comparison = {
        model1_name: flops1.to_dict(),
        model2_name: flops2.to_dict(),
        'speedup': {
            'forward': speedup_forward,
            'backward': speedup_backward,
            'total': speedup_total,
        }
    }
    
    print("=" * 70)
    print(f"Model Comparison: {model1_name} vs {model2_name}")
    print("=" * 70)
    print(f"{model1_name}:")
    print(f"  Forward:  {flops1.forward:>15,} FLOPs ({flops1.forward/1e9:.3f} GFLOPs)")
    print(f"  Backward: {flops1.backward:>15,} FLOPs ({flops1.backward/1e9:.3f} GFLOPs)")
    print(f"  Total:    {flops1.total:>15,} FLOPs ({flops1.total/1e9:.3f} GFLOPs)")
    print()
    print(f"{model2_name} (Baseline):")
    print(f"  Forward:  {flops2.forward:>15,} FLOPs ({flops2.forward/1e9:.3f} GFLOPs)")
    print(f"  Backward: {flops2.backward:>15,} FLOPs ({flops2.backward/1e9:.3f} GFLOPs)")
    print(f"  Total:    {flops2.total:>15,} FLOPs ({flops2.total/1e9:.3f} GFLOPs)")
    print()
    print(f"Speedup:")
    print(f"  Forward:  {speedup_forward:.2f}×")
    print(f"  Backward: {speedup_backward:.2f}×")
    print(f"  Total:    {speedup_total:.2f}×")
    print("=" * 70)
    
    return comparison


if __name__ == '__main__':
    # Example usage
    from src.models.configurable_resnet_bk import ConfigurableResNetBK, BASELINE_CONFIG
    
    # Create model
    config = BASELINE_CONFIG
    model = ConfigurableResNetBK(config)
    
    # Count FLOPs
    counter = FLOPsCounter(model, batch_size=32, seq_len=128)
    counter.print_summary()
    
    # Save to JSON
    counter.save_to_json('flops_count.json')
