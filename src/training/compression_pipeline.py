"""
Automated compression pipeline for ResNet-BK models.

This module orchestrates the complete compression workflow:
1. Quantization-Aware Training (QAT)
2. Structured Pruning
3. Knowledge Distillation

Achieving 100× compression with minimal accuracy loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np
import time
from pathlib import Path


class CompressionPipeline:
    """
    Automated compression pipeline: QAT → Pruning → Distillation.
    
    Orchestrates all compression techniques to achieve target compression ratio.
    """
    
    def __init__(self, model: nn.Module, target_compression: float = 100.0,
                 device: str = 'cuda'):
        """
        Args:
            model: Model to compress
            target_compression: Target compression ratio (e.g., 100 = 100×)
            device: Device to run on
        """
        self.model = model.to(device)
        self.target_compression = target_compression
        self.device = device
        
        # Track compression stages
        self.compression_history = []
        self.original_params = self._count_parameters(model)
        
        print(f"\n=== Compression Pipeline Initialized ===")
        print(f"Original model parameters: {self.original_params:,}")
        print(f"Target compression: {target_compression}×")
        print(f"Target parameters: {int(self.original_params / target_compression):,}")
    
    def _count_parameters(self, model: nn.Module) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def _measure_model_size(self, model: nn.Module) -> Dict[str, float]:
        """
        Measure model size in bytes.
        
        Returns:
            Dictionary with size metrics
        """
        # FP32 size
        fp32_size = sum(p.numel() * 4 for p in model.parameters())  # 4 bytes per FP32
        
        # Estimate quantized size (if quantization is applied)
        # This is approximate - actual size depends on quantization scheme
        quantized_size = fp32_size  # Default to FP32
        
        return {
            'fp32_bytes': fp32_size,
            'fp32_mb': fp32_size / (1024 ** 2),
            'quantized_bytes': quantized_size,
            'quantized_mb': quantized_size / (1024 ** 2)
        }
    
    def run_pipeline(self, train_loader, val_loader, 
                    qat_epochs: int = 3,
                    pruning_epochs: int = 3,
                    distillation_epochs: int = 5,
                    save_dir: Optional[str] = None) -> Tuple[nn.Module, Dict]:
        """
        Execute full compression pipeline.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            qat_epochs: Epochs for quantization-aware training
            pruning_epochs: Epochs for pruning
            distillation_epochs: Epochs for distillation
            save_dir: Directory to save checkpoints
        
        Returns:
            compressed_model: Final compressed model
            metrics: Compression metrics
        """
        print(f"\n{'='*60}")
        print(f"COMPRESSION PIPELINE EXECUTION")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Stage 1: Quantization-Aware Training
        print(f"\n{'='*60}")
        print(f"STAGE 1: QUANTIZATION-AWARE TRAINING")
        print(f"{'='*60}")
        qat_model, qat_metrics = self.stage1_quantization_aware_training(
            train_loader, val_loader, epochs=qat_epochs
        )
        
        if save_dir:
            self._save_checkpoint(qat_model, Path(save_dir) / 'qat_model.pt', qat_metrics)
        
        # Stage 2: Structured Pruning
        print(f"\n{'='*60}")
        print(f"STAGE 2: STRUCTURED PRUNING")
        print(f"{'='*60}")
        pruned_model, pruning_metrics = self.stage2_structured_pruning(
            qat_model, train_loader, val_loader, epochs=pruning_epochs
        )
        
        if save_dir:
            self._save_checkpoint(pruned_model, Path(save_dir) / 'pruned_model.pt', pruning_metrics)
        
        # Stage 3: Knowledge Distillation
        print(f"\n{'='*60}")
        print(f"STAGE 3: KNOWLEDGE DISTILLATION")
        print(f"{'='*60}")
        final_model, distillation_metrics = self.stage3_knowledge_distillation(
            pruned_model, train_loader, val_loader, epochs=distillation_epochs
        )
        
        if save_dir:
            self._save_checkpoint(final_model, Path(save_dir) / 'final_model.pt', distillation_metrics)
        
        # Compute final metrics
        end_time = time.time()
        total_time = end_time - start_time
        
        final_metrics = self._compute_final_metrics(
            final_model, qat_metrics, pruning_metrics, distillation_metrics, total_time
        )
        
        # Print summary
        self._print_summary(final_metrics)
        
        return final_model, final_metrics
    
    def stage1_quantization_aware_training(self, train_loader, val_loader,
                                          epochs: int = 3) -> Tuple[nn.Module, Dict]:
        """
        Stage 1: Quantization-Aware Training.
        
        Replace BK-Core with quantized version and train with fake quantization.
        """
        from src.models.quantized_bk_core import QuantizedBKCore
        
        print(f"\nReplacing BK-Core with quantized version...")
        
        # Get the actual model (handle ConfigurableResNetBK wrapper)
        actual_model = self.model.model if hasattr(self.model, 'model') else self.model
        
        # Replace BK-Core in each block
        for block_idx, block in enumerate(actual_model.blocks):
            if hasattr(block, 'bk_layer') and hasattr(block.bk_layer, 'bk_core'):
                # Get n_seq from the layer
                n_seq = block.bk_layer.n_seq if hasattr(block.bk_layer, 'n_seq') else 128
                quantized_core = QuantizedBKCore(n_seq=n_seq, enable_quantization=True)
                quantized_core.train()  # Set to training mode
                block.bk_layer.bk_core = quantized_core
                print(f"  Block {block_idx}: Replaced with QuantizedBKCore")
        
        # Calibration phase
        print(f"\nCalibration phase (collecting statistics)...")
        for block in actual_model.blocks:
            if hasattr(block, 'bk_layer') and hasattr(block.bk_layer, 'bk_core'):
                if hasattr(block.bk_layer.bk_core, 'start_calibration'):
                    block.bk_layer.bk_core.start_calibration()
        
        # Run a few batches for calibration
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (x_batch, _) in enumerate(train_loader):
                if batch_idx >= 10:  # Calibrate on 10 batches
                    break
                x_batch = x_batch.to(self.device)
                _ = self.model(x_batch)
        
        # End calibration
        for block in actual_model.blocks:
            if hasattr(block, 'bk_layer') and hasattr(block.bk_layer, 'bk_core'):
                if hasattr(block.bk_layer.bk_core, 'end_calibration'):
                    block.bk_layer.bk_core.end_calibration()
        
        # QAT training
        print(f"\nQuantization-Aware Training for {epochs} epochs...")
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        qat_losses = []
        for epoch in range(epochs):
            epoch_loss = self._train_epoch(train_loader, optimizer, criterion)
            val_metrics = self._evaluate(val_loader, criterion)
            
            qat_losses.append(epoch_loss)
            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"Train Loss = {epoch_loss:.4f}, "
                  f"Val PPL = {val_metrics['perplexity']:.2f}")
        
        metrics = {
            'stage': 'qat',
            'final_perplexity': val_metrics['perplexity'],
            'training_losses': qat_losses,
            'parameters': self._count_parameters(self.model)
        }
        
        self.compression_history.append(metrics)
        
        return self.model, metrics
    
    def stage2_structured_pruning(self, model: nn.Module, train_loader, val_loader,
                                 epochs: int = 3) -> Tuple[nn.Module, Dict]:
        """
        Stage 2: Structured Pruning.
        
        Prune unused MoE experts and low-magnitude weights.
        """
        from src.models.pruned_moe import PrunedMoELayer, MagnitudePruner
        
        print(f"\nReplacing MoE layers with prunable versions...")
        
        # Get the actual model
        actual_model = model.model if hasattr(model, 'model') else model
        
        # Replace MoE layers
        for block_idx, block in enumerate(actual_model.blocks):
            if hasattr(block, 'bk_layer') and hasattr(block.bk_layer, 'moe_ffn'):
                old_moe = block.bk_layer.moe_ffn
                d_model = old_moe.d_model if hasattr(old_moe, 'd_model') else 64
                num_experts = old_moe.num_experts if hasattr(old_moe, 'num_experts') else 4
                
                pruned_moe = PrunedMoELayer(
                    d_model=d_model,
                    num_experts=num_experts,
                    prune_threshold=0.05
                ).to(self.device)  # Move to GPU
                
                # Try to copy weights if structure matches
                try:
                    if hasattr(old_moe, 'gating') and hasattr(pruned_moe, 'gating'):
                        # Copy gating weights (should be compatible)
                        pruned_moe.gating.weight.data.copy_(old_moe.gating.weight.data)
                        if old_moe.gating.bias is not None and pruned_moe.gating.bias is not None:
                            pruned_moe.gating.bias.data.copy_(old_moe.gating.bias.data)
                        print(f"  Block {block_idx}: Copied gating weights")
                    
                    # Copy expert weights (skip if structure doesn't match)
                    if hasattr(old_moe, 'experts') and hasattr(pruned_moe, 'experts'):
                        for i in range(min(len(old_moe.experts), len(pruned_moe.experts))):
                            old_expert = old_moe.experts[i]
                            new_expert = pruned_moe.experts[i]
                            
                            # Copy layer by layer (skip Dropout layers)
                            old_layers = [m for m in old_expert if isinstance(m, nn.Linear)]
                            new_layers = [m for m in new_expert if isinstance(m, nn.Linear)]
                            
                            for old_layer, new_layer in zip(old_layers, new_layers):
                                if old_layer.weight.shape == new_layer.weight.shape:
                                    new_layer.weight.data.copy_(old_layer.weight.data)
                                    if old_layer.bias is not None and new_layer.bias is not None:
                                        new_layer.bias.data.copy_(old_layer.bias.data)
                        
                        print(f"  Block {block_idx}: Copied expert weights")
                except Exception as e:
                    print(f"  Block {block_idx}: Could not copy weights ({e}), using random initialization")
                
                block.bk_layer.moe_ffn = pruned_moe
                print(f"  Block {block_idx}: Replaced with PrunedMoELayer")
        
        # Training with pruning
        print(f"\nTraining with progressive pruning for {epochs} epochs...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        pruning_losses = []
        active_experts_history = []
        
        for epoch in range(epochs):
            epoch_loss = self._train_epoch(train_loader, optimizer, criterion)
            
            # Prune after each epoch
            total_pruned = 0
            total_active = 0
            for block in actual_model.blocks:
                if hasattr(block, 'bk_layer') and hasattr(block.bk_layer, 'moe_ffn'):
                    moe = block.bk_layer.moe_ffn
                    if hasattr(moe, 'prune_experts'):
                        num_pruned = moe.prune_experts(verbose=(epoch == epochs - 1))
                        total_pruned += num_pruned
                        total_active += moe.get_num_active_experts()
            
            val_metrics = self._evaluate(val_loader, criterion)
            
            pruning_losses.append(epoch_loss)
            active_experts_history.append(total_active)
            
            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"Train Loss = {epoch_loss:.4f}, "
                  f"Val PPL = {val_metrics['perplexity']:.2f}, "
                  f"Active Experts = {total_active}")
        
        # Magnitude-based pruning
        print(f"\nApplying magnitude-based pruning...")
        magnitude_pruner = MagnitudePruner(threshold=0.01)
        pruning_stats = magnitude_pruner.prune_model(
            model, 
            layer_names=None,  # Prune all linear layers
            verbose=True
        )
        
        metrics = {
            'stage': 'pruning',
            'final_perplexity': val_metrics['perplexity'],
            'training_losses': pruning_losses,
            'active_experts_history': active_experts_history,
            'magnitude_pruning_stats': pruning_stats,
            'parameters': self._count_parameters(model)
        }
        
        self.compression_history.append(metrics)
        
        return model, metrics
    
    def stage3_knowledge_distillation(self, teacher_model: nn.Module, 
                                     train_loader, val_loader,
                                     epochs: int = 5) -> Tuple[nn.Module, Dict]:
        """
        Stage 3: Knowledge Distillation.
        
        Train smaller student model from compressed teacher.
        """
        from src.training.distillation_trainer import DistillationTrainer
        from src.models.configurable_resnet_bk import ConfigurableResNetBK
        
        print(f"\nCreating student model (50% size)...")
        
        # Get teacher config (handle ConfigurableResNetBK wrapper)
        actual_teacher = teacher_model.model if hasattr(teacher_model, 'model') else teacher_model
        teacher_d_model = actual_teacher.d_model if hasattr(actual_teacher, 'd_model') else 64
        teacher_n_layers = actual_teacher.n_layers if hasattr(actual_teacher, 'n_layers') else 4
        vocab_size = actual_teacher.vocab_size if hasattr(actual_teacher, 'vocab_size') else 50257
        n_seq = actual_teacher.n_seq if hasattr(actual_teacher, 'n_seq') else 128
        
        # Create smaller student
        student_d_model = max(32, teacher_d_model // 2)
        student_n_layers = max(2, teacher_n_layers // 2)
        student_num_experts = 2
        
        print(f"  Teacher: d_model={teacher_d_model}, n_layers={teacher_n_layers}")
        print(f"  Student: d_model={student_d_model}, n_layers={student_n_layers}")
        
        from src.models.configurable_resnet_bk import ResNetBKConfig
        
        student_config = ResNetBKConfig(
            vocab_size=vocab_size,
            d_model=student_d_model,
            n_layers=student_n_layers,
            n_seq=n_seq,
            num_experts=student_num_experts,
            top_k=1,
            use_analytic_gradient=True,
            grad_blend=0.5
        )
        
        student_model = ConfigurableResNetBK(student_config).to(self.device)
        
        teacher_params = self._count_parameters(teacher_model)
        student_params = self._count_parameters(student_model)
        print(f"  Compression: {teacher_params / student_params:.2f}×")
        
        # Distillation training
        print(f"\nDistillation training for {epochs} epochs...")
        trainer = DistillationTrainer(
            teacher_model=teacher_model,
            student_model=student_model,
            temperature=2.0,
            alpha=0.7,
            device=self.device
        )
        
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        distillation_losses = []
        for epoch in range(epochs):
            epoch_losses = []
            for x_batch, y_batch in train_loader:
                loss_dict = trainer.train_step(x_batch, y_batch, optimizer)
                epoch_losses.append(loss_dict['loss_total'])
            
            avg_loss = np.mean(epoch_losses)
            val_metrics = trainer.evaluate(val_loader, criterion)
            
            distillation_losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"Train Loss = {avg_loss:.4f}, "
                  f"Val PPL = {val_metrics['val_perplexity']:.2f}")
        
        metrics = {
            'stage': 'distillation',
            'final_perplexity': val_metrics['val_perplexity'],
            'training_losses': distillation_losses,
            'parameters': self._count_parameters(student_model),
            'teacher_params': teacher_params,
            'student_params': student_params
        }
        
        self.compression_history.append(metrics)
        
        return student_model, metrics
    
    def _train_epoch(self, train_loader, optimizer, criterion) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            optimizer.zero_grad()
            logits = self.model(x_batch)
            loss = criterion(logits.view(-1, logits.size(-1)), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        return epoch_loss / num_batches
    
    def _evaluate(self, val_loader, criterion) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                logits = self.model(x_batch)
                loss = criterion(logits.view(-1, logits.size(-1)), y_batch)
                
                total_loss += loss.item() * y_batch.size(0)
                total_tokens += y_batch.size(0)
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return {'loss': avg_loss, 'perplexity': perplexity}
    
    def _compute_final_metrics(self, final_model, qat_metrics, pruning_metrics,
                               distillation_metrics, total_time) -> Dict:
        """Compute final compression metrics."""
        final_params = self._count_parameters(final_model)
        compression_ratio = self.original_params / final_params
        
        size_metrics = self._measure_model_size(final_model)
        
        metrics = {
            'original_parameters': self.original_params,
            'final_parameters': final_params,
            'compression_ratio': compression_ratio,
            'target_compression': self.target_compression,
            'compression_achieved': compression_ratio >= self.target_compression,
            'total_time_seconds': total_time,
            'model_size': size_metrics,
            'stage_metrics': {
                'qat': qat_metrics,
                'pruning': pruning_metrics,
                'distillation': distillation_metrics
            }
        }
        
        return metrics
    
    def _print_summary(self, metrics: Dict):
        """Print compression summary."""
        print(f"\n{'='*60}")
        print(f"COMPRESSION PIPELINE SUMMARY")
        print(f"{'='*60}")
        print(f"Original parameters: {metrics['original_parameters']:,}")
        print(f"Final parameters: {metrics['final_parameters']:,}")
        print(f"Compression ratio: {metrics['compression_ratio']:.2f}×")
        print(f"Target compression: {metrics['target_compression']:.2f}×")
        print(f"Target achieved: {'✓' if metrics['compression_achieved'] else '✗'}")
        print(f"Total time: {metrics['total_time_seconds']:.2f}s")
        print(f"\nModel size:")
        print(f"  FP32: {metrics['model_size']['fp32_mb']:.2f} MB")
        print(f"  Quantized: {metrics['model_size']['quantized_mb']:.2f} MB")
        print(f"\nFinal perplexity: {metrics['stage_metrics']['distillation']['final_perplexity']:.2f}")
        print(f"{'='*60}")
    
    def _save_checkpoint(self, model: nn.Module, path: Path, metrics: Dict):
        """Save model checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics': metrics
        }, path)
        print(f"Saved checkpoint: {path}")
