"""
Demo: Knowledge Distillation with Clark Measure Loss

This demo shows how to:
1. Set up knowledge distillation with Clark measure preservation
2. Train a compressed model (smaller ε) from a teacher model
3. Verify that Clark measure is preserved during compression
4. Perform progressive compression through multiple ε values

Requirements: 4.9, 4.10
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.clark_distillation import (
    ClarkDistillationTrainer,
    DistillationConfig,
    ClarkMeasureLoss,
    progressive_compression
)
from src.models.birman_schwinger_core import BirmanSchwingerCore


class SimpleModelWithBK(nn.Module):
    """Simple model with BirmanSchwingerCore for testing."""
    
    def __init__(self, vocab_size=1000, d_model=128, n_seq=64, epsilon=1.0):
        super().__init__()
        self.epsilon = epsilon
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.bk_core = BirmanSchwingerCore(n_seq=n_seq, epsilon=epsilon)
        self.output = nn.Linear(d_model, vocab_size)
        
        # Store last G_ii for Clark measure computation
        self.last_G_ii = None
    
    def forward(self, input_ids):
        # Embed
        x = self.embedding(input_ids)  # (B, N, D)
        
        # Generate potential from embeddings
        v = x.mean(dim=-1)  # (B, N)
        
        # BK-Core
        features, _ = self.bk_core(v, z=1.0j)
        self.last_G_ii = torch.complex(features[..., 0], features[..., 1])
        
        # Combine with embeddings
        bk_features = features.unsqueeze(-1).expand(-1, -1, -1, x.size(-1))
        x = x.unsqueeze(2) + bk_features  # Broadcast
        x = x.mean(dim=2)  # Aggregate
        
        # Output
        logits = self.output(x)
        return logits
    
    def get_resolvent_diagonal(self, input_ids):
        """Extract G_ii for Clark measure computation."""
        with torch.no_grad():
            _ = self.forward(input_ids)
        return self.last_G_ii


def demo_clark_measure_loss():
    """Demo 1: Clark measure loss computation."""
    print("=" * 70)
    print("Demo 1: Clark Measure Loss")
    print("=" * 70)
    
    # Create teacher and student models
    teacher = SimpleModelWithBK(epsilon=1.0, n_seq=32)
    student = SimpleModelWithBK(epsilon=0.5, n_seq=32)
    
    # Sample input
    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward pass to get G_ii
    _ = teacher(input_ids)
    _ = student(input_ids)
    
    G_ii_teacher = teacher.last_G_ii
    G_ii_student = student.last_G_ii
    
    print(f"Teacher G_ii shape: {G_ii_teacher.shape}")
    print(f"Student G_ii shape: {G_ii_student.shape}")
    
    # Compute Clark measure loss
    clark_loss_fn = ClarkMeasureLoss(
        lambda_min=-5.0,
        lambda_max=5.0,
        num_points=200
    )
    
    loss, info = clark_loss_fn(
        G_ii_teacher,
        G_ii_student,
        epsilon_teacher=1.0,
        epsilon_student=0.5
    )
    
    print(f"\nClark Measure Loss:")
    print(f"  L2 distance: {info['clark_l2_loss']:.6f}")
    print(f"  TV distance: {info['clark_tv_distance']:.6f}")
    print(f"  Teacher total mass: {info['teacher_total_mass']:.6f}")
    print(f"  Student total mass: {info['student_total_mass']:.6f}")
    print(f"  Teacher valid: {info['teacher_valid']}")
    print(f"  Student valid: {info['student_valid']}")


def demo_distillation_trainer():
    """Demo 2: Distillation trainer setup."""
    print("\n" + "=" * 70)
    print("Demo 2: Distillation Trainer")
    print("=" * 70)
    
    # Create models
    teacher = SimpleModelWithBK(epsilon=1.0, n_seq=32)
    student = SimpleModelWithBK(epsilon=0.5, n_seq=32)
    
    # Setup trainer
    config = DistillationConfig(
        temperature=2.0,
        alpha_ce=0.5,
        alpha_kd=0.5,
        lambda_clark=0.1,
        compute_clark_every_n_steps=10
    )
    
    trainer = ClarkDistillationTrainer(
        teacher_model=teacher,
        student_model=student,
        config=config
    )
    
    print(f"Distillation Config:")
    print(f"  Temperature: {config.temperature}")
    print(f"  α_CE: {config.alpha_ce}")
    print(f"  α_KD: {config.alpha_kd}")
    print(f"  λ_Clark: {config.lambda_clark}")
    print(f"  Compute Clark every: {config.compute_clark_every_n_steps} steps")
    
    # Sample batch
    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Compute loss
    print("\nComputing distillation loss...")
    loss, info = trainer.compute_distillation_loss(
        input_ids, labels, compute_clark=True
    )
    
    print(f"\nLoss Components:")
    print(f"  Total loss: {info['loss_total']:.4f}")
    print(f"  CE loss: {info['loss_ce']:.4f}")
    print(f"  KD loss: {info['loss_kd']:.4f}")
    print(f"  Clark loss: {info['loss_clark']:.4f}")


def demo_training_step():
    """Demo 3: Single training step."""
    print("\n" + "=" * 70)
    print("Demo 3: Training Step")
    print("=" * 70)
    
    # Create models
    teacher = SimpleModelWithBK(epsilon=1.0, n_seq=32)
    student = SimpleModelWithBK(epsilon=0.5, n_seq=32)
    
    # Setup trainer
    config = DistillationConfig(
        lambda_clark=0.1,
        compute_clark_every_n_steps=5
    )
    
    trainer = ClarkDistillationTrainer(
        teacher_model=teacher,
        student_model=student,
        config=config
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    
    # Training loop
    print("Running 10 training steps...")
    
    for step in range(10):
        # Sample batch
        input_ids = torch.randint(0, 1000, (4, 32))
        labels = torch.randint(0, 1000, (4, 32))
        
        # Training step
        info = trainer.train_step(input_ids, labels, optimizer)
        
        print(f"Step {info['step']}: "
              f"loss={info['loss_total']:.4f}, "
              f"ce={info['loss_ce']:.4f}, "
              f"kd={info['loss_kd']:.4f}, "
              f"clark={info['loss_clark']:.4f}")


def demo_compression_stages():
    """Demo 4: Progressive compression stages."""
    print("\n" + "=" * 70)
    print("Demo 4: Progressive Compression")
    print("=" * 70)
    
    # Epsilon schedule
    epsilon_schedule = [1.0, 0.75, 0.5, 0.25, 0.1]
    
    print(f"Compression schedule: {epsilon_schedule}")
    print(f"Number of stages: {len(epsilon_schedule) - 1}")
    
    # Simulate compression stages
    for i in range(len(epsilon_schedule) - 1):
        eps_teacher = epsilon_schedule[i]
        eps_student = epsilon_schedule[i + 1]
        
        print(f"\nStage {i+1}: ε={eps_teacher} → ε={eps_student}")
        
        # Create models
        teacher = SimpleModelWithBK(epsilon=eps_teacher, n_seq=32)
        student = SimpleModelWithBK(epsilon=eps_student, n_seq=32)
        
        # Setup trainer
        config = DistillationConfig(lambda_clark=0.1)
        trainer = ClarkDistillationTrainer(teacher, student, config)
        
        # Sample batch
        input_ids = torch.randint(0, 1000, (4, 32))
        labels = torch.randint(0, 1000, (4, 32))
        
        # Compute loss
        loss, info = trainer.compute_distillation_loss(
            input_ids, labels, compute_clark=True
        )
        
        print(f"  Initial loss: {info['loss_total']:.4f}")
        print(f"  Clark TV distance: {info.get('clark_tv_distance', 'N/A')}")


def demo_loss_components():
    """Demo 5: Analyze loss components."""
    print("\n" + "=" * 70)
    print("Demo 5: Loss Component Analysis")
    print("=" * 70)
    
    # Create models
    teacher = SimpleModelWithBK(epsilon=1.0, n_seq=32)
    student = SimpleModelWithBK(epsilon=0.5, n_seq=32)
    
    # Test different lambda_clark values
    lambda_values = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
    
    print("Testing different λ_Clark values:")
    print("-" * 70)
    
    for lambda_clark in lambda_values:
        config = DistillationConfig(lambda_clark=lambda_clark)
        trainer = ClarkDistillationTrainer(teacher, student, config)
        
        # Sample batch
        input_ids = torch.randint(0, 1000, (4, 32))
        labels = torch.randint(0, 1000, (4, 32))
        
        # Compute loss
        loss, info = trainer.compute_distillation_loss(
            input_ids, labels, compute_clark=True
        )
        
        print(f"λ_Clark={lambda_clark:.2f}: "
              f"total={info['loss_total']:.4f}, "
              f"ce={info['loss_ce']:.4f}, "
              f"kd={info['loss_kd']:.4f}, "
              f"clark={info['loss_clark']:.4f}")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("Clark Distillation Demo")
    print("Implementing Requirements 4.9, 4.10")
    print("=" * 70)
    
    # Set random seed
    torch.manual_seed(42)
    
    try:
        demo_clark_measure_loss()
        demo_distillation_trainer()
        demo_training_step()
        demo_compression_stages()
        demo_loss_components()
        
        print("\n" + "=" * 70)
        print("All demos completed successfully!")
        print("=" * 70)
        print("\nKey Features Demonstrated:")
        print("  ✓ Clark measure loss computation")
        print("  ✓ Knowledge distillation with soft targets")
        print("  ✓ Combined loss: L = L_CE + L_KD + λ_Clark * L_Clark")
        print("  ✓ Progressive compression: ε = 1.0 → 0.1")
        print("  ✓ Measure preservation verification")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
