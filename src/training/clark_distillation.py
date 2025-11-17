"""
Knowledge Distillation with Clark Measure Loss

This module implements knowledge distillation for model compression
using Clark measure preservation as an additional loss term.

Loss: L = L_CE + λ_Clark · ||μ_teacher - μ_student||²

Requirements: 4.9, 4.10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging
from dataclasses import dataclass

from src.models.clark_measure import ClarkMeasureComputer, ClarkMeasureResult

logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Configuration for Clark measure distillation."""
    temperature: float = 2.0  # Temperature for soft targets
    alpha_ce: float = 0.5  # Weight for cross-entropy loss
    alpha_kd: float = 0.5  # Weight for KD loss (soft targets)
    lambda_clark: float = 0.1  # Weight for Clark measure loss
    
    # Clark measure computation
    lambda_min: float = -10.0
    lambda_max: float = 10.0
    num_points: int = 500
    
    # Training
    compute_clark_every_n_steps: int = 100  # Expensive, compute periodically


class ClarkMeasureLoss(nn.Module):
    """
    Clark measure matching loss: ||μ_teacher - μ_student||².
    
    This loss ensures that the compressed model preserves the
    spectral distribution of the teacher model.
    
    Args:
        lambda_min: Minimum spectral value
        lambda_max: Maximum spectral value
        num_points: Number of grid points
    """
    
    def __init__(
        self,
        lambda_min: float = -10.0,
        lambda_max: float = 10.0,
        num_points: int = 500
    ):
        super().__init__()
        self.clark_computer = ClarkMeasureComputer(
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            num_points=num_points
        )
    
    def forward(
        self,
        G_ii_teacher: torch.Tensor,
        G_ii_student: torch.Tensor,
        epsilon_teacher: float,
        epsilon_student: float
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute Clark measure matching loss.
        
        Args:
            G_ii_teacher: Teacher resolvent diagonal (B, N)
            G_ii_student: Student resolvent diagonal (B, N)
            epsilon_teacher: Teacher ε parameter
            epsilon_student: Student ε parameter
        
        Returns:
            loss: Clark measure L2 distance
            info: Dictionary with diagnostic information
        """
        # Compute Clark measures
        measure_teacher = self.clark_computer.compute_measure(
            G_ii_teacher, epsilon_teacher
        )
        measure_student = self.clark_computer.compute_measure(
            G_ii_student, epsilon_student
        )
        
        # L2 distance between measures
        # ||μ_teacher - μ_student||² = ∫ (μ_teacher(λ) - μ_student(λ))² dλ
        diff = measure_teacher.measure_values - measure_student.measure_values
        l2_distance_sq = torch.tensor(
            (diff ** 2).sum() * self.clark_computer.dlambda,
            dtype=torch.float32,
            device=G_ii_teacher.device
        )
        
        # Also compute TV distance for monitoring
        tv_distance = self.clark_computer.compute_total_variation_distance(
            measure_teacher, measure_student
        )
        
        info = {
            'clark_l2_loss': l2_distance_sq.item(),
            'clark_tv_distance': tv_distance,
            'teacher_total_mass': measure_teacher.total_mass,
            'student_total_mass': measure_student.total_mass,
            'teacher_valid': measure_teacher.is_valid,
            'student_valid': measure_student.is_valid,
        }
        
        return l2_distance_sq, info


class ClarkDistillationTrainer:
    """
    Trainer for knowledge distillation with Clark measure preservation.
    
    Implements:
    L = α_CE * L_CE(y, y_student) + α_KD * L_KD(y_teacher, y_student) + λ_Clark * L_Clark
    
    where:
    - L_CE: Standard cross-entropy loss
    - L_KD: KL divergence between teacher and student logits (soft targets)
    - L_Clark: Clark measure matching loss
    
    Args:
        teacher_model: Teacher model (larger ε)
        student_model: Student model (smaller ε)
        config: Distillation configuration
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        config: DistillationConfig
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.config = config
        
        # Freeze teacher
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
        
        # Clark measure loss
        self.clark_loss_fn = ClarkMeasureLoss(
            lambda_min=config.lambda_min,
            lambda_max=config.lambda_max,
            num_points=config.num_points
        )
        
        self.step_count = 0
    
    def compute_distillation_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        compute_clark: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined distillation loss.
        
        Args:
            input_ids: Input token IDs (B, N)
            labels: Target labels (B, N)
            compute_clark: Whether to compute Clark measure loss (expensive)
        
        Returns:
            total_loss: Combined loss
            info: Dictionary with loss components
        """
        # Student forward pass
        student_logits = self.student_model(input_ids)
        
        # Teacher forward pass (no grad)
        with torch.no_grad():
            teacher_logits = self.teacher_model(input_ids)
        
        # 1. Cross-entropy loss (hard targets)
        loss_ce = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        # 2. Knowledge distillation loss (soft targets)
        # KL divergence between teacher and student distributions
        T = self.config.temperature
        
        # Soften logits with temperature
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        
        # KL divergence
        loss_kd = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (T ** 2)  # Scale by T^2 as per Hinton et al.
        
        # 3. Clark measure loss (optional, computed periodically)
        loss_clark = torch.tensor(0.0, device=input_ids.device)
        clark_info = {}
        
        if compute_clark:
            # Extract G_ii from both models
            G_ii_teacher = self._extract_resolvent_diagonal(
                self.teacher_model, input_ids
            )
            G_ii_student = self._extract_resolvent_diagonal(
                self.student_model, input_ids
            )
            
            if G_ii_teacher is not None and G_ii_student is not None:
                # Get epsilon values
                epsilon_teacher = self._get_epsilon(self.teacher_model)
                epsilon_student = self._get_epsilon(self.student_model)
                
                loss_clark, clark_info = self.clark_loss_fn(
                    G_ii_teacher,
                    G_ii_student,
                    epsilon_teacher,
                    epsilon_student
                )
        
        # Combined loss
        total_loss = (
            self.config.alpha_ce * loss_ce +
            self.config.alpha_kd * loss_kd +
            self.config.lambda_clark * loss_clark
        )
        
        # Collect info
        info = {
            'loss_total': total_loss.item(),
            'loss_ce': loss_ce.item(),
            'loss_kd': loss_kd.item(),
            'loss_clark': loss_clark.item() if isinstance(loss_clark, torch.Tensor) else 0.0,
            **clark_info
        }
        
        return total_loss, info
    
    def _extract_resolvent_diagonal(
        self,
        model: nn.Module,
        input_ids: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Extract G_ii (resolvent diagonal) from model.
        
        This assumes the model has BirmanSchwingerCore that stores last_G_ii.
        """
        # Try to find BirmanSchwingerCore in model
        for module in model.modules():
            if hasattr(module, 'last_G_ii'):
                return module.last_G_ii
        
        # If not found, try to get via method
        if hasattr(model, 'get_resolvent_diagonal'):
            with torch.no_grad():
                return model.get_resolvent_diagonal(input_ids)
        
        logger.warning("Could not extract G_ii from model")
        return None
    
    def _get_epsilon(self, model: nn.Module) -> float:
        """Get epsilon parameter from model."""
        # Try to find epsilon in model config or modules
        if hasattr(model, 'epsilon'):
            return model.epsilon
        
        for module in model.modules():
            if hasattr(module, 'epsilon'):
                return module.epsilon
        
        # Default
        logger.warning("Could not find epsilon in model, using default 1.0")
        return 1.0
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict:
        """
        Single training step with distillation.
        
        Args:
            input_ids: Input token IDs
            labels: Target labels
            optimizer: Optimizer
        
        Returns:
            Dictionary with loss information
        """
        self.student_model.train()
        
        # Decide whether to compute Clark loss this step
        compute_clark = (
            self.step_count % self.config.compute_clark_every_n_steps == 0
        )
        
        # Forward pass and loss computation
        loss, info = self.compute_distillation_loss(
            input_ids, labels, compute_clark=compute_clark
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        self.step_count += 1
        info['step'] = self.step_count
        
        return info
    
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_steps: Optional[int] = None
    ) -> Dict:
        """
        Evaluate student model with distillation metrics.
        
        Args:
            dataloader: Evaluation dataloader
            max_steps: Maximum number of steps (None = full dataset)
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.student_model.eval()
        
        total_loss = 0.0
        total_loss_ce = 0.0
        total_loss_kd = 0.0
        total_loss_clark = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if max_steps is not None and i >= max_steps:
                    break
                
                input_ids = batch['input_ids']
                labels = batch['labels']
                
                # Compute loss (with Clark measure every 10 batches)
                compute_clark = (i % 10 == 0)
                loss, info = self.compute_distillation_loss(
                    input_ids, labels, compute_clark=compute_clark
                )
                
                total_loss += info['loss_total']
                total_loss_ce += info['loss_ce']
                total_loss_kd += info['loss_kd']
                total_loss_clark += info['loss_clark']
                num_batches += 1
        
        return {
            'eval_loss': total_loss / num_batches,
            'eval_loss_ce': total_loss_ce / num_batches,
            'eval_loss_kd': total_loss_kd / num_batches,
            'eval_loss_clark': total_loss_clark / num_batches,
        }


def progressive_compression(
    model: nn.Module,
    epsilon_schedule: list = None,
    train_dataloader = None,
    eval_dataloader = None,
    num_epochs_per_stage: int = 5,
    base_lr: float = 1e-4,
    device: str = 'cuda'
) -> Dict[float, nn.Module]:
    """
    Progressive compression: ε = 1.0 → 0.75 → 0.5 → 0.25 → 0.1.
    
    Each stage uses the previous model as teacher and trains a new
    student with smaller ε.
    
    Args:
        model: Initial model (ε=1.0)
        epsilon_schedule: List of ε values (default: [1.0, 0.75, 0.5, 0.25, 0.1])
        train_dataloader: Training data
        eval_dataloader: Evaluation data
        num_epochs_per_stage: Epochs per compression stage
        base_lr: Base learning rate
        device: Device to use
    
    Returns:
        Dictionary mapping ε to trained model
    """
    if epsilon_schedule is None:
        epsilon_schedule = [1.0, 0.75, 0.5, 0.25, 0.1]
    
    models = {}
    current_model = model
    models[epsilon_schedule[0]] = current_model
    
    logger.info(f"Starting progressive compression: {epsilon_schedule}")
    
    for i in range(len(epsilon_schedule) - 1):
        eps_teacher = epsilon_schedule[i]
        eps_student = epsilon_schedule[i + 1]
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Compression Stage: ε={eps_teacher} → ε={eps_student}")
        logger.info(f"{'='*70}")
        
        # Create student model with smaller epsilon
        # (In practice, would clone and modify epsilon parameter)
        student_model = _create_student_model(current_model, eps_student)
        student_model = student_model.to(device)
        
        # Setup distillation trainer
        config = DistillationConfig(
            lambda_clark=0.1,
            compute_clark_every_n_steps=100
        )
        
        trainer = ClarkDistillationTrainer(
            teacher_model=current_model,
            student_model=student_model,
            config=config
        )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            student_model.parameters(),
            lr=base_lr
        )
        
        # Train for specified epochs
        for epoch in range(num_epochs_per_stage):
            logger.info(f"Epoch {epoch+1}/{num_epochs_per_stage}")
            
            # Training loop
            for batch_idx, batch in enumerate(train_dataloader):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                info = trainer.train_step(input_ids, labels, optimizer)
                
                if batch_idx % 100 == 0:
                    logger.info(
                        f"  Step {batch_idx}: "
                        f"loss={info['loss_total']:.4f}, "
                        f"ce={info['loss_ce']:.4f}, "
                        f"kd={info['loss_kd']:.4f}, "
                        f"clark={info['loss_clark']:.4f}"
                    )
            
            # Evaluation
            if eval_dataloader is not None:
                eval_metrics = trainer.evaluate(eval_dataloader, max_steps=50)
                logger.info(f"  Eval: {eval_metrics}")
        
        # Save student model
        models[eps_student] = student_model
        current_model = student_model
        
        logger.info(f"✓ Compression stage {eps_teacher} → {eps_student} complete")
    
    logger.info(f"\n{'='*70}")
    logger.info("Progressive compression complete!")
    logger.info(f"Trained models at ε = {list(models.keys())}")
    logger.info(f"{'='*70}")
    
    return models


def _create_student_model(teacher_model: nn.Module, epsilon: float) -> nn.Module:
    """
    Create student model with smaller epsilon.
    
    This is a placeholder - in practice, would clone teacher and modify epsilon.
    """
    import copy
    student = copy.deepcopy(teacher_model)
    
    # Update epsilon in all modules
    for module in student.modules():
        if hasattr(module, 'epsilon'):
            module.epsilon = epsilon
    
    return student
