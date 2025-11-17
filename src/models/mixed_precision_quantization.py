"""
Mixed-Precision Quantization for ResNet-BK

This module implements Task 14 and 14.1 from mamba-killer-ultra-scale spec:
- Task 14: Mixed-precision quantization (INT4 for MoE, INT8 for BK-Core, FP16 for output)
- Task 14.1: Dynamic quantization based on layer importance

Requirements:
- 7.10: Implement mixed-precision quantization: INT4 for MoE, INT8 for BK-Core, FP16 for output
- 7.11: Achieve 6× model size reduction with < 8% PPL degradation
- 7.12: Implement dynamic quantization: adjust precision based on layer importance
- 7.13: Achieve better accuracy-size trade-off than uniform quantization

Mathematical Foundation:
- Layer importance measured by gradient magnitude and activation variance
- Sensitive layers (high importance) get higher precision (INT8 or FP16)
- Less sensitive layers (low importance) get lower precision (INT4)
- Mixed precision: INT4 (0.5 bytes) + INT8 (1 byte) + FP16 (2 bytes) + FP32 (4 bytes)

Architecture:
- MoE Experts: INT4 (most parameters, less sensitive)
- BK-Core: INT8 (critical for numerical stability)
- Output Layers: FP16 (final projection, needs precision)
- Embeddings: FP16 (vocabulary mapping)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np

from .complex_quantization import ComplexQuantizer
from .quantized_birman_schwinger import QuantizationConfig, GroupWiseQuantizer


class LayerImportanceAnalyzer:
    """
    Analyze layer importance for dynamic quantization.
    
    Measures importance based on:
    1. Gradient magnitude (sensitivity to weight changes)
    2. Activation variance (information content)
    3. Weight magnitude (parameter significance)
    
    Args:
        model: PyTorch model to analyze
        num_samples: Number of samples for importance estimation
    """
    
    def __init__(self, model: nn.Module, num_samples: int = 100):
        self.model = model
        self.num_samples = num_samples
        
        # Storage for importance metrics
        self.layer_importance = {}
        self.gradient_magnitudes = {}
        self.activation_variances = {}
        self.weight_magnitudes = {}
        
        # Hooks for activation tracking
        self.activation_hooks = []
        self.activations = {}
    
    def register_hooks(self):
        """Register forward hooks to capture activations."""
        def get_activation_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activations[name] = output.detach()
            return hook
        
        # Register hooks for all layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                hook = module.register_forward_hook(get_activation_hook(name))
                self.activation_hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.activation_hooks:
            hook.remove()
        self.activation_hooks = []
    
    def compute_gradient_importance(self, loss: torch.Tensor):
        """
        Compute gradient-based importance for each layer.
        
        Args:
            loss: Loss tensor to backpropagate
        """
        # Backpropagate to compute gradients
        loss.backward(retain_graph=True)
        
        # Collect gradient magnitudes
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_mag = param.grad.abs().mean().item()
                
                if name not in self.gradient_magnitudes:
                    self.gradient_magnitudes[name] = []
                self.gradient_magnitudes[name].append(grad_mag)
    
    def compute_activation_importance(self):
        """Compute activation-based importance from captured activations."""
        for name, activation in self.activations.items():
            # Compute variance as measure of information content
            var = activation.var().item()
            
            if name not in self.activation_variances:
                self.activation_variances[name] = []
            self.activation_variances[name].append(var)
        
        # Clear activations to save memory
        self.activations = {}
    
    def compute_weight_importance(self):
        """Compute weight-based importance from parameter magnitudes."""
        for name, param in self.model.named_parameters():
            weight_mag = param.abs().mean().item()
            
            if name not in self.weight_magnitudes:
                self.weight_magnitudes[name] = []
            self.weight_magnitudes[name].append(weight_mag)
    
    def analyze(self, dataloader, num_batches: Optional[int] = None):
        """
        Analyze layer importance using sample data.
        
        Args:
            dataloader: DataLoader providing sample batches
            num_batches: Number of batches to analyze (None = all)
        
        Returns:
            layer_importance: Dict mapping layer names to importance scores
        """
        self.model.eval()
        self.register_hooks()
        
        print(f"Analyzing layer importance with {num_batches or 'all'} batches...")
        
        batch_count = 0
        for batch_idx, batch in enumerate(dataloader):
            if num_batches is not None and batch_idx >= num_batches:
                break
            
            # Forward pass
            if isinstance(batch, dict):
                inputs = batch['input_ids']
                targets = batch.get('labels', batch.get('target_ids'))
            elif isinstance(batch, (tuple, list)):
                inputs = batch[0]
                targets = batch[1] if len(batch) > 1 else None
            else:
                inputs = batch
                targets = None
            
            # Move to device
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            if targets is not None:
                targets = targets.to(device)
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Compute loss if targets available
            if targets is not None:
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Compute cross-entropy loss
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-100
                )
                
                # Compute gradient importance
                self.compute_gradient_importance(loss)
                
                # Zero gradients
                self.model.zero_grad()
            
            # Compute activation importance
            self.compute_activation_importance()
            
            # Compute weight importance
            self.compute_weight_importance()
            
            batch_count += 1
            
            if batch_count % 10 == 0:
                print(f"  Processed {batch_count} batches...")
        
        self.remove_hooks()
        
        # Aggregate importance scores
        self._aggregate_importance()
        
        print(f"Layer importance analysis complete ({batch_count} batches)")
        return self.layer_importance
    
    def _aggregate_importance(self):
        """Aggregate importance metrics into final scores."""
        # Normalize each metric
        all_grad_mags = []
        all_act_vars = []
        all_weight_mags = []
        
        for name in self.gradient_magnitudes.keys():
            all_grad_mags.extend(self.gradient_magnitudes[name])
        for name in self.activation_variances.keys():
            all_act_vars.extend(self.activation_variances[name])
        for name in self.weight_magnitudes.keys():
            all_weight_mags.extend(self.weight_magnitudes[name])
        
        # Compute global statistics
        grad_mean = np.mean(all_grad_mags) if all_grad_mags else 1.0
        grad_std = np.std(all_grad_mags) if all_grad_mags else 1.0
        act_mean = np.mean(all_act_vars) if all_act_vars else 1.0
        act_std = np.std(all_act_vars) if all_act_vars else 1.0
        weight_mean = np.mean(all_weight_mags) if all_weight_mags else 1.0
        weight_std = np.std(all_weight_mags) if all_weight_mags else 1.0
        
        # Compute normalized importance for each layer
        all_layer_names = set()
        all_layer_names.update(self.gradient_magnitudes.keys())
        all_layer_names.update(self.activation_variances.keys())
        all_layer_names.update(self.weight_magnitudes.keys())
        
        for name in all_layer_names:
            # Gradient importance (normalized)
            grad_imp = 0.0
            if name in self.gradient_magnitudes:
                grad_avg = np.mean(self.gradient_magnitudes[name])
                grad_imp = (grad_avg - grad_mean) / (grad_std + 1e-8)
            
            # Activation importance (normalized)
            act_imp = 0.0
            if name in self.activation_variances:
                act_avg = np.mean(self.activation_variances[name])
                act_imp = (act_avg - act_mean) / (act_std + 1e-8)
            
            # Weight importance (normalized)
            weight_imp = 0.0
            if name in self.weight_magnitudes:
                weight_avg = np.mean(self.weight_magnitudes[name])
                weight_imp = (weight_avg - weight_mean) / (weight_std + 1e-8)
            
            # Combined importance (weighted average)
            # Gradient is most important, then activation, then weight
            importance = 0.5 * grad_imp + 0.3 * act_imp + 0.2 * weight_imp
            
            self.layer_importance[name] = importance
        
        # Print top 10 most important layers
        sorted_layers = sorted(
            self.layer_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        print("\nTop 10 most important layers:")
        for name, importance in sorted_layers[:10]:
            print(f"  {name}: {importance:.4f}")


class DynamicQuantizationPolicy:
    """
    Dynamic quantization policy based on layer importance.
    
    Assigns quantization precision to each layer based on importance:
    - High importance (top 20%): FP16 or INT8
    - Medium importance (20-60%): INT8
    - Low importance (bottom 40%): INT4
    
    Args:
        layer_importance: Dict mapping layer names to importance scores
        high_precision_ratio: Ratio of layers to keep in high precision (default: 0.2)
        low_precision_ratio: Ratio of layers to use low precision (default: 0.4)
    """
    
    def __init__(
        self,
        layer_importance: Dict[str, float],
        high_precision_ratio: float = 0.2,
        low_precision_ratio: float = 0.4,
    ):
        self.layer_importance = layer_importance
        self.high_precision_ratio = high_precision_ratio
        self.low_precision_ratio = low_precision_ratio
        
        # Compute thresholds
        importance_values = sorted(layer_importance.values(), reverse=True)
        num_layers = len(importance_values)
        
        # Calculate indices (use max to ensure at least 1 layer in each category if possible)
        high_idx = max(1, int(num_layers * high_precision_ratio))
        low_idx = max(high_idx + 1, int(num_layers * (1 - low_precision_ratio)))
        
        # Ensure indices are within bounds
        high_idx = min(high_idx, num_layers - 1)
        low_idx = min(low_idx, num_layers)
        
        # Set thresholds (use slightly lower value to ensure correct assignment)
        if high_idx < num_layers:
            self.high_threshold = importance_values[high_idx - 1] if high_idx > 0 else importance_values[0]
        else:
            self.high_threshold = float('inf')
        
        if low_idx < num_layers:
            self.low_threshold = importance_values[low_idx - 1] if low_idx > 0 else importance_values[0]
        else:
            self.low_threshold = float('-inf')
        
        # Assign precision to each layer
        self.layer_precision = {}
        for name, importance in layer_importance.items():
            if importance >= self.high_threshold:
                self.layer_precision[name] = 'fp16'  # or 'int8' for very high importance
            elif importance >= self.low_threshold:
                self.layer_precision[name] = 'int8'
            else:
                self.layer_precision[name] = 'int4'
        
        # Print summary
        fp16_count = sum(1 for p in self.layer_precision.values() if p == 'fp16')
        int8_count = sum(1 for p in self.layer_precision.values() if p == 'int8')
        int4_count = sum(1 for p in self.layer_precision.values() if p == 'int4')
        
        print(f"\nDynamic quantization policy:")
        print(f"  FP16: {fp16_count} layers ({fp16_count/num_layers*100:.1f}%)")
        print(f"  INT8: {int8_count} layers ({int8_count/num_layers*100:.1f}%)")
        print(f"  INT4: {int4_count} layers ({int4_count/num_layers*100:.1f}%)")
    
    def get_precision(self, layer_name: str) -> str:
        """Get quantization precision for a layer."""
        return self.layer_precision.get(layer_name, 'int8')  # Default to INT8


class MixedPrecisionQuantizer:
    """
    Mixed-precision quantizer for ResNet-BK model.
    
    Implements Task 14: Mixed-precision quantization
    - INT4 for MoE experts (most parameters, less sensitive)
    - INT8 for BK-Core (critical for numerical stability)
    - FP16 for output layers (final projection needs precision)
    
    Implements Task 14.1: Dynamic quantization
    - Adjust precision based on layer importance
    - High importance layers: FP16 or INT8
    - Low importance layers: INT4
    
    Args:
        model: ResNet-BK model to quantize
        policy: Dynamic quantization policy (optional)
        group_size: Group size for INT4 quantization (default: 128)
    """
    
    def __init__(
        self,
        model: nn.Module,
        policy: Optional[DynamicQuantizationPolicy] = None,
        group_size: int = 128,
    ):
        self.model = model
        self.policy = policy
        self.group_size = group_size
        
        # Quantizers for different components
        self.quantizers = {}
        
        # Calibration state
        self.calibrated = False
        self.calibration_samples = {}
    
    def _identify_component_type(self, name: str, module: nn.Module) -> str:
        """
        Identify component type for mixed-precision assignment.
        
        Returns:
            'moe_expert': MoE expert network (INT4)
            'bk_core': BK-Core component (INT8)
            'output': Output layer (FP16)
            'embedding': Embedding layer (FP16)
            'other': Other components (INT8 default)
        """
        name_lower = name.lower()
        
        # MoE experts
        if 'expert' in name_lower and 'moe' in name_lower:
            return 'moe_expert'
        
        # BK-Core components
        if any(x in name_lower for x in ['bk_core', 'birman', 'schwinger', 'resolvent']):
            return 'bk_core'
        
        # Output layers
        if any(x in name_lower for x in ['lm_head', 'output_proj', 'final']):
            return 'output'
        
        # Embeddings
        if any(x in name_lower for x in ['embed', 'token', 'position']):
            return 'embedding'
        
        return 'other'
    
    def _get_precision_for_layer(self, name: str, module: nn.Module) -> str:
        """
        Get quantization precision for a layer.
        
        Priority:
        1. Dynamic policy (if provided)
        2. Component type (MoE=INT4, BK-Core=INT8, Output=FP16)
        3. Default (INT8)
        
        Returns:
            'fp32', 'fp16', 'int8', or 'int4'
        """
        # Check dynamic policy first
        if self.policy is not None:
            return self.policy.get_precision(name)
        
        # Use component-based assignment
        component_type = self._identify_component_type(name, module)
        
        if component_type == 'moe_expert':
            return 'int4'  # Requirement 7.10: INT4 for MoE
        elif component_type == 'bk_core':
            return 'int8'  # Requirement 7.10: INT8 for BK-Core
        elif component_type in ['output', 'embedding']:
            return 'fp16'  # Requirement 7.10: FP16 for output
        else:
            return 'int8'  # Default
    
    def create_quantizers(self):
        """Create quantizers for each layer based on precision assignment."""
        print("Creating mixed-precision quantizers...")
        
        precision_counts = {'fp32': 0, 'fp16': 0, 'int8': 0, 'int4': 0}
        
        for name, module in self.model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                continue
            
            precision = self._get_precision_for_layer(name, module)
            precision_counts[precision] += 1
            
            # Create appropriate quantizer
            if precision == 'int4':
                # INT4 with group-wise quantization
                if hasattr(module, 'out_features'):
                    num_channels = module.out_features
                elif hasattr(module, 'out_channels'):
                    num_channels = module.out_channels
                else:
                    num_channels = 128  # Default
                
                quantizer = GroupWiseQuantizer(
                    num_channels=num_channels,
                    group_size=self.group_size,
                    bits=4,
                )
            elif precision == 'int8':
                # INT8 with per-channel quantization
                if hasattr(module, 'out_features'):
                    num_channels = module.out_features
                elif hasattr(module, 'out_channels'):
                    num_channels = module.out_channels
                else:
                    num_channels = 128  # Default
                
                quantizer = GroupWiseQuantizer(
                    num_channels=num_channels,
                    group_size=num_channels,  # Per-channel = group_size = num_channels
                    bits=8,
                )
            elif precision == 'fp16':
                # FP16: no quantizer needed, just cast
                quantizer = None
            else:
                # FP32: no quantization
                quantizer = None
            
            self.quantizers[name] = {
                'precision': precision,
                'quantizer': quantizer,
                'module': module,
            }
        
        print(f"Mixed-precision quantizers created:")
        print(f"  FP32: {precision_counts['fp32']} layers")
        print(f"  FP16: {precision_counts['fp16']} layers")
        print(f"  INT8: {precision_counts['int8']} layers")
        print(f"  INT4: {precision_counts['int4']} layers")
    
    def start_calibration(self):
        """Start calibration mode to collect samples."""
        self.calibration_samples = {name: [] for name in self.quantizers.keys()}
        print("Started calibration for mixed-precision quantization")
    
    def calibrate_layer(self, name: str, output: torch.Tensor):
        """Collect calibration sample for a layer."""
        if name in self.calibration_samples:
            self.calibration_samples[name].append(output.detach().cpu())
    
    def end_calibration(self):
        """End calibration and compute quantization parameters."""
        print("Calibrating mixed-precision quantizers...")
        
        for name, info in self.quantizers.items():
            quantizer = info['quantizer']
            if quantizer is None:
                continue  # FP16/FP32, no calibration needed
            
            samples = self.calibration_samples.get(name, [])
            if len(samples) == 0:
                print(f"  Warning: No samples for {name}")
                continue
            
            # Stack samples
            samples_tensor = torch.cat(samples, dim=0)
            
            # Calibrate
            quantizer.calibrate(samples_tensor)
        
        self.calibrated = True
        self.calibration_samples = {}  # Clear to save memory
        
        print("Mixed-precision calibration complete")
    
    def quantize_model(self):
        """
        Apply quantization to model weights.
        
        This modifies the model in-place, replacing FP32 weights with quantized versions.
        """
        if not self.calibrated:
            raise RuntimeError("Must calibrate before quantizing. Call start_calibration(), forward(), end_calibration().")
        
        print("Applying mixed-precision quantization to model...")
        
        for name, info in self.quantizers.items():
            precision = info['precision']
            quantizer = info['quantizer']
            module = info['module']
            
            if precision == 'fp16':
                # Convert to FP16
                module.half()
            elif precision in ['int8', 'int4']:
                # Quantize weights
                if hasattr(module, 'weight'):
                    weight_fp32 = module.weight.data
                    
                    # Quantize
                    weight_int = quantizer.quantize(weight_fp32)
                    weight_quant = quantizer.dequantize(weight_int)
                    
                    # Replace weight
                    module.weight.data = weight_quant
        
        print("Mixed-precision quantization applied")
    
    def estimate_model_size(self) -> Dict[str, float]:
        """
        Estimate model size with mixed-precision quantization.
        
        Returns:
            Dictionary with size estimates
        """
        total_params = 0
        fp32_size = 0
        mixed_size = 0
        
        for name, info in self.quantizers.items():
            precision = info['precision']
            module = info['module']
            
            # Count parameters
            num_params = sum(p.numel() for p in module.parameters())
            total_params += num_params
            
            # FP32 size
            fp32_size += num_params * 4
            
            # Mixed precision size
            if precision == 'fp32':
                mixed_size += num_params * 4
            elif precision == 'fp16':
                mixed_size += num_params * 2
            elif precision == 'int8':
                mixed_size += num_params * 1
            elif precision == 'int4':
                mixed_size += num_params * 0.5
        
        # Add quantization parameters (scales and zero points)
        for name, info in self.quantizers.items():
            quantizer = info['quantizer']
            if quantizer is not None:
                # Each group needs scale (FP32) + zero_point (INT32)
                num_groups = quantizer.num_groups
                mixed_size += num_groups * 2 * 4  # 2 params × 4 bytes
        
        compression_ratio = fp32_size / mixed_size if mixed_size > 0 else 0.0
        
        return {
            'total_parameters': total_params,
            'fp32_bytes': fp32_size,
            'mixed_precision_bytes': mixed_size,
            'compression_ratio': compression_ratio,
            'target_compression': 6.0,  # Requirement 7.11: 6× reduction
            'meets_target': compression_ratio >= 6.0,
        }


def create_mixed_precision_quantizer(
    model: nn.Module,
    dataloader: Optional[torch.utils.data.DataLoader] = None,
    use_dynamic_policy: bool = True,
    num_importance_batches: int = 50,
    group_size: int = 128,
) -> MixedPrecisionQuantizer:
    """
    Factory function to create mixed-precision quantizer.
    
    Args:
        model: ResNet-BK model to quantize
        dataloader: DataLoader for importance analysis (required if use_dynamic_policy=True)
        use_dynamic_policy: Use dynamic quantization based on layer importance
        num_importance_batches: Number of batches for importance analysis
        group_size: Group size for INT4 quantization
    
    Returns:
        MixedPrecisionQuantizer instance
    
    Examples:
        # Static mixed-precision (component-based)
        >>> quantizer = create_mixed_precision_quantizer(model, use_dynamic_policy=False)
        
        # Dynamic mixed-precision (importance-based)
        >>> quantizer = create_mixed_precision_quantizer(
        ...     model, dataloader=train_loader, use_dynamic_policy=True
        ... )
    """
    policy = None
    
    if use_dynamic_policy:
        if dataloader is None:
            raise ValueError("dataloader required for dynamic quantization policy")
        
        # Analyze layer importance
        analyzer = LayerImportanceAnalyzer(model, num_samples=num_importance_batches)
        layer_importance = analyzer.analyze(dataloader, num_batches=num_importance_batches)
        
        # Create dynamic policy
        policy = DynamicQuantizationPolicy(layer_importance)
    
    # Create quantizer
    quantizer = MixedPrecisionQuantizer(
        model=model,
        policy=policy,
        group_size=group_size,
    )
    
    quantizer.create_quantizers()
    
    return quantizer
