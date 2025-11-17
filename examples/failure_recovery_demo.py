"""
Failure Recovery and Monitoring Demo

Demonstrates the complete failure recovery system for Mamba-Killer ResNet-BK:
- StabilityMonitor: Real-time health monitoring
- AutoRecovery: Automatic failure detection and recovery
- ColabTimeoutHandler: Colab timeout handling and emergency checkpointing

Based on Requirement 12: 失敗モード分析と自動リカバリ
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.stability_monitor import StabilityMonitor
from src.training.auto_recovery import AutoRecovery
from src.training.colab_timeout_handler import ColabTimeoutHandler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_dummy_model():
    """Create a simple model for demonstration."""
    return nn.Sequential(
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 100)
    )


def create_dummy_dataloader(batch_size=32, num_batches=100):
    """Create dummy data for demonstration."""
    X = torch.randn(batch_size * num_batches, 100)
    y = torch.randn(batch_size * num_batches, 100)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def simulate_training_with_failures(
    model,
    optimizer,
    dataloader,
    monitor,
    recovery,
    timeout_handler,
    num_epochs=5,
    inject_failures=True
):
    """
    Simulate training with potential failures.
    
    Args:
        model: Model to train
        optimizer: Optimizer
        dataloader: Training data
        monitor: StabilityMonitor instance
        recovery: AutoRecovery instance
        timeout_handler: ColabTimeoutHandler instance
        num_epochs: Number of epochs to train
        inject_failures: Whether to inject artificial failures for testing
    """
    logger.info("="*60)
    logger.info("Starting Training with Failure Recovery")
    logger.info("="*60)
    
    global_step = 0
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        logger.info("-"*60)
        
        epoch_loss = 0.0
        
        for batch_idx, (X, y) in enumerate(dataloader):
            global_step += 1
            
            # Forward pass
            optimizer.zero_grad()
            output = model(X)
            loss = nn.functional.mse_loss(output, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Check stability
            metrics = monitor.check_step(model, loss, optimizer, step=global_step)
            
            # Detect failures
            failure_type = recovery.detect_failure(metrics)
            
            if failure_type:
                logger.warning(f"⚠️ Failure detected: {failure_type}")
                
                # Attempt recovery
                success, action = recovery.recover(
                    failure_type=failure_type,
                    model=model,
                    optimizer=optimizer,
                    scheduler=None,
                    current_step=global_step,
                    current_epoch=epoch,
                    dataloader=dataloader
                )
                
                if not success:
                    logger.error("❌ Recovery failed. Halting training.")
                    return False
                
                logger.info(f"✓ Recovery successful: {action}")
            
            # Check for Colab timeout
            timeout_info = timeout_handler.check_timeout()
            
            if timeout_info['should_save']:
                logger.warning("⚠️ Colab timeout imminent. Saving emergency checkpoint...")
                
                checkpoint_path = timeout_handler.save_emergency_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=None,
                    step=global_step,
                    epoch=epoch,
                    metrics={
                        'loss': loss.item(),
                        'epoch_loss': epoch_loss / (batch_idx + 1)
                    }
                )
                
                if checkpoint_path:
                    logger.info(f"✓ Emergency checkpoint saved: {checkpoint_path}")
                    logger.info("Training can be resumed from this checkpoint.")
                    return True
            
            # Save regular checkpoint every 50 steps
            if global_step % 50 == 0:
                checkpoint_path = recovery.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=None,
                    step=global_step,
                    epoch=epoch,
                    metrics={'loss': loss.item()},
                    is_stable=(metrics.overall_health == "healthy")
                )
            
            # Inject artificial failures for testing
            if inject_failures:
                # Inject NaN at step 30
                if global_step == 30:
                    logger.info("\n🧪 Injecting NaN for testing...")
                    with torch.no_grad():
                        for param in model.parameters():
                            param.data[0] = float('nan')
                            break
                
                # Inject gradient explosion at step 80
                elif global_step == 80:
                    logger.info("\n🧪 Injecting gradient explosion for testing...")
                    with torch.no_grad():
                        for param in model.parameters():
                            if param.grad is not None:
                                param.grad.data *= 1000
                            break
            
            # Log progress
            if global_step % 20 == 0:
                logger.info(f"Step {global_step}: Loss = {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")
    
    logger.info("\n" + "="*60)
    logger.info("Training Completed Successfully")
    logger.info("="*60)
    
    return True


def main():
    """Main demonstration function."""
    print("\n" + "="*60)
    print("Failure Recovery and Monitoring Demo")
    print("="*60)
    
    # Create checkpoint directory
    checkpoint_dir = Path("./checkpoints/failure_recovery_demo")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    logger.info("\n1. Initializing components...")
    
    monitor = StabilityMonitor(
        check_interval=5,
        gradient_window=50,
        loss_window=50,
        enable_detailed_logging=True
    )
    
    recovery = AutoRecovery(
        checkpoint_dir=str(checkpoint_dir),
        max_retries=3,
        enable_auto_adjustment=True
    )
    
    timeout_handler = ColabTimeoutHandler(
        checkpoint_dir=str(checkpoint_dir),
        session_duration_hours=12.0,
        warning_threshold_minutes=30,
        enable_auto_resume=True
    )
    
    logger.info("✓ All components initialized")
    
    # Check for incomplete training
    logger.info("\n2. Checking for incomplete training...")
    resume_info = timeout_handler.auto_resume(None, None, None)
    
    if resume_info:
        logger.info("✓ Resuming from previous session")
        start_epoch = resume_info['epoch']
        start_step = resume_info['step']
    else:
        logger.info("ℹ️ Starting fresh training")
        start_epoch = 0
        start_step = 0
    
    # Create model and optimizer
    logger.info("\n3. Creating model and optimizer...")
    model = create_dummy_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Create dataloader
    logger.info("\n4. Creating dataloader...")
    dataloader = create_dummy_dataloader(batch_size=32, num_batches=20)
    
    # Run training with failure recovery
    logger.info("\n5. Starting training...")
    success = simulate_training_with_failures(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        monitor=monitor,
        recovery=recovery,
        timeout_handler=timeout_handler,
        num_epochs=3,
        inject_failures=True  # Enable failure injection for testing
    )
    
    # Print final statistics
    logger.info("\n" + "="*60)
    logger.info("Final Statistics")
    logger.info("="*60)
    
    # Health dashboard
    dashboard = monitor.get_health_dashboard()
    logger.info("\nHealth Dashboard:")
    logger.info(f"  Health Score: {dashboard['health_score']:.1f}/100")
    logger.info(f"  NaN Count: {dashboard['nan_count']}")
    logger.info(f"  Gradient Explosion Count: {dashboard['gradient_explosion_count']}")
    logger.info(f"  Loss Divergence Count: {dashboard['loss_divergence_count']}")
    logger.info(f"  Current Gradient Norm: {dashboard['current_gradient_norm']:.4f}")
    logger.info(f"  Loss Trend: {dashboard['loss_trend']}")
    
    # Recovery report
    recovery_report = recovery.get_recovery_report()
    logger.info("\nRecovery Report:")
    logger.info(f"  Total Recoveries: {recovery_report['total_recoveries']}")
    logger.info(f"  Rollback Count: {recovery_report['rollback_count']}")
    logger.info(f"  LR Reduction Count: {recovery_report['lr_reduction_count']}")
    logger.info(f"  Epsilon Increase Count: {recovery_report['epsilon_increase_count']}")
    logger.info(f"  Last Recovery Action: {recovery_report['last_recovery_action']}")
    
    # Timeout handler status
    timeout_status = timeout_handler.get_status()
    logger.info("\nTimeout Handler Status:")
    logger.info(f"  Is Colab: {timeout_status['is_colab']}")
    logger.info(f"  Elapsed Time: {timeout_status['elapsed_hours']:.2f} hours")
    logger.info(f"  Remaining Time: {timeout_status['remaining_minutes']:.1f} minutes")
    logger.info(f"  Emergency Checkpoint Saved: {timeout_status['emergency_checkpoint_saved']}")
    
    # Export metrics
    logger.info("\n6. Exporting metrics...")
    metrics_path = checkpoint_dir / "training_metrics.json"
    monitor.export_metrics(str(metrics_path))
    logger.info(f"✓ Metrics exported to {metrics_path}")
    
    logger.info("\n" + "="*60)
    if success:
        logger.info("✓ Demo completed successfully!")
    else:
        logger.info("⚠️ Demo completed with failures")
    logger.info("="*60)


if __name__ == '__main__':
    main()
