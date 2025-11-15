"""
Knowledge distillation trainer for ResNet-BK models.

This module implements distillation from a large teacher model to a smaller student,
using both soft targets (teacher logits) and feature matching (BK-Core outputs).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class DistillationTrainer:
    """
    Knowledge distillation: train small student from large teacher.
    
    Combines:
    - Soft targets: KL divergence between teacher and student logits
    - Hard targets: Cross-entropy with ground truth labels
    - Feature distillation: MSE between intermediate BK-Core features
    """
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module,
                 temperature: float = 2.0, alpha: float = 0.5, 
                 feature_weight: float = 0.1, device: str = 'cuda'):
        """
        Args:
            teacher_model: Pre-trained teacher model
            student_model: Student model to train
            temperature: Temperature for soft targets (higher = softer)
            alpha: Balance between soft (alpha) and hard (1-alpha) targets
            feature_weight: Weight for feature distillation loss
            device: Device to run on
        """
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.temperature = temperature
        self.alpha = alpha
        self.feature_weight = feature_weight
        self.device = device
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        # Storage for intermediate features
        self.teacher_features = []
        self.student_features = []
        
        # Register hooks to capture BK-Core features
        self._register_feature_hooks()
    
    def _register_feature_hooks(self):
        """Register forward hooks to capture intermediate features."""
        def get_teacher_hook(layer_idx):
            def hook(module, input, output):
                # Store BK-Core features (G_ii)
                if hasattr(module, 'output_features'):
                    self.teacher_features.append(module.output_features.detach())
            return hook
        
        def get_student_hook(layer_idx):
            def hook(module, input, output):
                # Store BK-Core features (G_ii)
                if hasattr(module, 'output_features'):
                    self.student_features.append(module.output_features)
            return hook
        
        # Get actual models (handle ConfigurableResNetBK wrapper)
        actual_teacher = self.teacher.model if hasattr(self.teacher, 'model') else self.teacher
        actual_student = self.student.model if hasattr(self.student, 'model') else self.student
        
        # Register hooks for each ResNet-BK block
        if hasattr(actual_teacher, 'blocks'):
            for idx, block in enumerate(actual_teacher.blocks):
                if hasattr(block, 'bk_layer'):
                    block.bk_layer.register_forward_hook(get_teacher_hook(idx))
        
        if hasattr(actual_student, 'blocks'):
            for idx, block in enumerate(actual_student.blocks):
                if hasattr(block, 'bk_layer'):
                    block.bk_layer.register_forward_hook(get_student_hook(idx))
    
    def distillation_loss(self, student_logits: torch.Tensor, 
                         teacher_logits: torch.Tensor,
                         targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined distillation loss.
        
        Args:
            student_logits: (B*N, vocab_size) - student predictions
            teacher_logits: (B*N, vocab_size) - teacher predictions
            targets: (B*N,) - ground truth labels
        
        Returns:
            loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        # Soft targets loss (KL divergence)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        loss_soft = F.kl_div(soft_student, soft_targets, reduction='batchmean') * (self.temperature ** 2)
        
        # Hard targets loss (cross-entropy)
        loss_hard = F.cross_entropy(student_logits, targets)
        
        # Combined logit loss
        loss_logits = self.alpha * loss_soft + (1 - self.alpha) * loss_hard
        
        # Feature distillation loss
        loss_features = self._feature_distillation_loss()
        
        # Total loss
        loss = loss_logits + self.feature_weight * loss_features
        
        loss_dict = {
            'loss_total': loss.item(),
            'loss_soft': loss_soft.item(),
            'loss_hard': loss_hard.item(),
            'loss_features': loss_features.item()
        }
        
        return loss, loss_dict
    
    def _feature_distillation_loss(self) -> torch.Tensor:
        """
        Compute feature distillation loss (MSE between BK-Core features).
        
        Returns:
            Feature matching loss
        """
        if len(self.teacher_features) == 0 or len(self.student_features) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Match features from corresponding layers
        num_layers = min(len(self.teacher_features), len(self.student_features))
        
        loss = 0.0
        for i in range(num_layers):
            teacher_feat = self.teacher_features[i]
            student_feat = self.student_features[i]
            
            # Handle dimension mismatch (teacher may be larger)
            if teacher_feat.shape != student_feat.shape:
                # Interpolate or project to match dimensions
                if teacher_feat.shape[-1] != student_feat.shape[-1]:
                    # Different feature dimensions - use projection
                    # For simplicity, skip this layer
                    continue
                
                # Match sequence length if different
                if teacher_feat.shape[1] != student_feat.shape[1]:
                    # Interpolate teacher features to match student
                    teacher_feat = F.interpolate(
                        teacher_feat.transpose(1, 2),
                        size=student_feat.shape[1],
                        mode='linear',
                        align_corners=False
                    ).transpose(1, 2)
            
            # MSE loss
            loss += F.mse_loss(student_feat, teacher_feat)
        
        if num_layers > 0:
            loss = loss / num_layers
        
        return loss
    
    def train_step(self, x_batch: torch.Tensor, y_batch: torch.Tensor,
                   optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """
        Single training step with distillation.
        
        Args:
            x_batch: (B, N) - input token IDs
            y_batch: (B*N,) - target token IDs (flattened)
            optimizer: Optimizer for student model
        
        Returns:
            Dictionary of loss values
        """
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        
        # Clear feature storage
        self.teacher_features = []
        self.student_features = []
        
        # Teacher forward (no gradients)
        with torch.no_grad():
            teacher_logits = self.teacher(x_batch)
            teacher_logits_flat = teacher_logits.view(-1, teacher_logits.size(-1))
        
        # Student forward
        optimizer.zero_grad()
        student_logits = self.student(x_batch)
        student_logits_flat = student_logits.view(-1, student_logits.size(-1))
        
        # Compute distillation loss
        loss, loss_dict = self.distillation_loss(student_logits_flat, teacher_logits_flat, y_batch)
        
        # Backward and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        optimizer.step()
        
        return loss_dict
    
    def evaluate(self, val_loader, criterion: nn.Module) -> Dict[str, float]:
        """
        Evaluate student model on validation set.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss criterion (e.g., CrossEntropyLoss)
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.student.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Student forward
                logits = self.student(x_batch)
                logits_flat = logits.view(-1, logits.size(-1))
                
                # Compute loss
                loss = criterion(logits_flat, y_batch)
                
                total_loss += loss.item() * y_batch.size(0)
                total_tokens += y_batch.size(0)
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        self.student.train()
        
        return {
            'val_loss': avg_loss,
            'val_perplexity': perplexity
        }


class ProgressiveDistillation:
    """
    Progressive distillation: train sequence of smaller models.
    
    Each student learns from the previous teacher in a cascade.
    """
    
    def __init__(self, model_sizes: list, device: str = 'cuda'):
        """
        Args:
            model_sizes: List of (d_model, n_layers, num_experts) tuples
            device: Device to run on
        """
        self.model_sizes = model_sizes
        self.device = device
        self.models = []
        self.trainers = []
    
    def create_model(self, d_model: int, n_layers: int, num_experts: int,
                    vocab_size: int, n_seq: int) -> nn.Module:
        """
        Create a ResNet-BK model with specified size.
        
        Args:
            d_model: Model dimension
            n_layers: Number of layers
            num_experts: Number of MoE experts
            vocab_size: Vocabulary size
            n_seq: Sequence length
        
        Returns:
            Model instance
        """
        # Import here to avoid circular dependency
        from src.models.configurable_resnet_bk import ConfigurableResNetBK, ResNetBKConfig
        
        config = ResNetBKConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_seq=n_seq,
            num_experts=num_experts,
            top_k=1,
            use_analytic_gradient=True,
            grad_blend=0.5
        )
        
        model = ConfigurableResNetBK(config)
        
        return model.to(self.device)
    
    def train_cascade(self, initial_teacher: nn.Module, train_loader, val_loader,
                     vocab_size: int, n_seq: int, epochs_per_stage: int = 5,
                     temperature: float = 2.0, alpha: float = 0.7) -> list:
        """
        Train cascade of progressively smaller models.
        
        Args:
            initial_teacher: Pre-trained teacher model
            train_loader: Training data loader
            val_loader: Validation data loader
            vocab_size: Vocabulary size
            n_seq: Sequence length
            epochs_per_stage: Epochs to train each student
            temperature: Distillation temperature
            alpha: Soft target weight
        
        Returns:
            List of trained models
        """
        self.models = [initial_teacher]
        current_teacher = initial_teacher
        
        print(f"\n=== Progressive Distillation Cascade ===")
        print(f"Stages: {len(self.model_sizes)}")
        
        for stage_idx, (d_model, n_layers, num_experts) in enumerate(self.model_sizes):
            print(f"\n--- Stage {stage_idx + 1}/{len(self.model_sizes)} ---")
            print(f"Target size: d_model={d_model}, n_layers={n_layers}, num_experts={num_experts}")
            
            # Create student model
            student = self.create_model(d_model, n_layers, num_experts, vocab_size, n_seq)
            
            # Count parameters
            teacher_params = sum(p.numel() for p in current_teacher.parameters())
            student_params = sum(p.numel() for p in student.parameters())
            compression_ratio = teacher_params / student_params
            
            print(f"Teacher params: {teacher_params:,}")
            print(f"Student params: {student_params:,}")
            print(f"Compression ratio: {compression_ratio:.2f}×")
            
            # Create distillation trainer
            trainer = DistillationTrainer(
                teacher_model=current_teacher,
                student_model=student,
                temperature=temperature,
                alpha=alpha,
                device=self.device
            )
            
            # Train student
            optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(epochs_per_stage):
                # Training
                epoch_losses = []
                for x_batch, y_batch in train_loader:
                    loss_dict = trainer.train_step(x_batch, y_batch, optimizer)
                    epoch_losses.append(loss_dict['loss_total'])
                
                avg_train_loss = np.mean(epoch_losses)
                
                # Validation
                val_metrics = trainer.evaluate(val_loader, criterion)
                
                print(f"Epoch {epoch + 1}/{epochs_per_stage}: "
                      f"Train Loss = {avg_train_loss:.4f}, "
                      f"Val PPL = {val_metrics['val_perplexity']:.2f}")
            
            # Save student and use as next teacher
            self.models.append(student)
            self.trainers.append(trainer)
            current_teacher = student
        
        print(f"\n=== Progressive Distillation Complete ===")
        print(f"Trained {len(self.models)} models")
        
        return self.models
