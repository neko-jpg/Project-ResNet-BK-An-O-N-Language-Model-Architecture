# Failure Recovery and Monitoring - Quick Reference

## Overview

Complete failure detection and automatic recovery system for Mamba-Killer ResNet-BK training, with special support for Google Colab timeout handling.

**Implementation Status:** ✅ Complete (Task 22)

**Requirements:** 12.1-12.20 (失敗モード分析と自動リカバリ)

## Components

### 1. StabilityMonitor

Real-time monitoring of training health with 20+ metrics.

**Features:**
- NaN/Inf detection in all tensors (Requirement 12.1)
- Gradient explosion detection (>10× median) (Requirement 12.3)
- Loss divergence detection (>50% increase over 100 steps) (Requirement 12.5)
- Numerical stability monitoring (condition numbers, Schatten norms) (Requirement 12.15)
- Training health dashboard (Requirement 12.17)

**Usage:**
```python
from src.training import StabilityMonitor

# Initialize monitor
monitor = StabilityMonitor(
    check_interval=10,           # Check every 10 steps
    gradient_window=100,         # Track last 100 gradient norms
    loss_window=100,             # Track last 100 losses
    gradient_explosion_threshold=10.0,  # 10× median
    loss_divergence_threshold=0.5,      # 50% increase
    condition_number_threshold=1e6,
    schatten_threshold=100.0
)

# During training loop
metrics = monitor.check_step(model, loss, optimizer, step=current_step)

# Check health status
if metrics.overall_health == "critical":
    print(f"Errors: {metrics.errors}")
elif metrics.overall_health == "warning":
    print(f"Warnings: {metrics.warnings}")

# Get comprehensive dashboard
dashboard = monitor.get_health_dashboard()
print(f"Health Score: {dashboard['health_score']}/100")
print(f"NaN Count: {dashboard['nan_count']}")
print(f"Gradient Explosion Count: {dashboard['gradient_explosion_count']}")

# Export metrics
monitor.export_metrics("training_metrics.json")
```

### 2. AutoRecovery

Automatic failure detection and recovery with multiple strategies.

**Features:**
- Rollback to last stable checkpoint on NaN/Inf (Requirement 12.2)
- Learning rate reduction (10×) on gradient explosion (Requirement 12.4)
- Epsilon increase on loss divergence (Requirement 12.6)
- Batch size reduction on OOM (Requirement 12.7, 12.8)
- Checkpoint corruption detection (Requirement 12.9)
- Automatic hyperparameter adjustment (Requirement 12.19)

**Usage:**
```python
from src.training import AutoRecovery

# Initialize recovery system
recovery = AutoRecovery(
    checkpoint_dir="./checkpoints",
    max_retries=3,
    enable_auto_adjustment=True,
    min_lr=1e-7,
    max_epsilon=1.0,
    min_batch_size=1
)

# During training loop
metrics = monitor.check_step(model, loss, optimizer, step)
failure_type = recovery.detect_failure(metrics)

if failure_type:
    success, action = recovery.recover(
        failure_type=failure_type,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        current_step=step,
        current_epoch=epoch,
        dataloader=dataloader
    )
    
    if not success:
        print("Recovery failed. Halting training.")
        break

# Save checkpoints regularly
if step % 100 == 0:
    checkpoint_path = recovery.save_checkpoint(
        model, optimizer, scheduler, step, epoch,
        metrics={'loss': loss.item()},
        is_stable=(metrics.overall_health == "healthy")
    )

# Get recovery report
report = recovery.get_recovery_report()
print(f"Total Recoveries: {report['total_recoveries']}")
print(f"Rollback Count: {report['rollback_count']}")
print(f"LR Reduction Count: {report['lr_reduction_count']}")
```

### 3. ColabTimeoutHandler

Google Colab timeout detection and emergency checkpointing.

**Features:**
- Timeout detection (<30 min remaining) (Requirement 12.11)
- Emergency checkpoint with full state (Requirement 12.12)
- Automatic resume from checkpoint (Requirement 12.13, 12.14)
- Random state preservation for reproducibility
- Session state tracking

**Usage:**
```python
from src.training import ColabTimeoutHandler

# Initialize timeout handler
timeout_handler = ColabTimeoutHandler(
    checkpoint_dir="./checkpoints",
    session_duration_hours=12.0,      # Colab free tier
    warning_threshold_minutes=30,     # Save when <30 min left
    check_interval_seconds=60,        # Check every minute
    enable_auto_resume=True
)

# At training start: check for incomplete training
resume_info = timeout_handler.auto_resume(model, optimizer, scheduler)
if resume_info:
    start_step = resume_info['step']
    start_epoch = resume_info['epoch']
    print(f"Resumed from step {start_step}, epoch {start_epoch}")

# During training loop
timeout_info = timeout_handler.check_timeout()

if timeout_info['should_save']:
    checkpoint_path = timeout_handler.save_emergency_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=current_step,
        epoch=current_epoch,
        metrics={'loss': loss.item(), 'ppl': ppl},
        random_state=None,  # Auto-captured
        additional_state={'custom_data': custom_data}
    )
    print(f"Emergency checkpoint saved: {checkpoint_path}")
    break  # Exit training gracefully

# Get status
status = timeout_handler.get_status()
print(f"Elapsed: {status['elapsed_hours']:.2f} hours")
print(f"Remaining: {status['remaining_minutes']:.1f} minutes")
```

## Complete Training Loop Example

```python
from src.training import StabilityMonitor, AutoRecovery, ColabTimeoutHandler

# Initialize all components
monitor = StabilityMonitor(check_interval=10)
recovery = AutoRecovery(checkpoint_dir="./checkpoints", max_retries=3)
timeout_handler = ColabTimeoutHandler(checkpoint_dir="./checkpoints")

# Check for incomplete training
resume_info = timeout_handler.auto_resume(model, optimizer, scheduler)
start_step = resume_info['step'] if resume_info else 0
start_epoch = resume_info['epoch'] if resume_info else 0

# Training loop
for epoch in range(start_epoch, num_epochs):
    for batch_idx, (X, y) in enumerate(dataloader):
        step = epoch * len(dataloader) + batch_idx
        
        if step < start_step:
            continue  # Skip already processed steps
        
        # Forward and backward
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # 1. Check stability
        metrics = monitor.check_step(model, loss, optimizer, step)
        
        # 2. Detect and recover from failures
        failure_type = recovery.detect_failure(metrics)
        if failure_type:
            success, action = recovery.recover(
                failure_type, model, optimizer, scheduler,
                step, epoch, dataloader=dataloader
            )
            if not success:
                print("Training halted due to unrecoverable failure")
                break
        
        # 3. Check for Colab timeout
        timeout_info = timeout_handler.check_timeout()
        if timeout_info['should_save']:
            timeout_handler.save_emergency_checkpoint(
                model, optimizer, scheduler, step, epoch,
                metrics={'loss': loss.item()}
            )
            print("Emergency checkpoint saved. Training can be resumed.")
            break
        
        # 4. Save regular checkpoints
        if step % 100 == 0:
            recovery.save_checkpoint(
                model, optimizer, scheduler, step, epoch,
                metrics={'loss': loss.item()},
                is_stable=(metrics.overall_health == "healthy")
            )

# Final statistics
dashboard = monitor.get_health_dashboard()
recovery_report = recovery.get_recovery_report()
timeout_status = timeout_handler.get_status()

print(f"Health Score: {dashboard['health_score']}/100")
print(f"Total Recoveries: {recovery_report['total_recoveries']}")
print(f"Session Duration: {timeout_status['elapsed_hours']:.2f} hours")
```

## Recovery Actions

| Failure Type | Recovery Action | Details |
|--------------|----------------|---------|
| `nan_detected` | Rollback checkpoint | Load last stable checkpoint |
| `inf_detected` | Rollback checkpoint | Load last stable checkpoint |
| `gradient_explosion` | Reduce LR 10× | Multiply learning rate by 0.1 |
| `loss_divergence` | Increase epsilon | Multiply ε by 1.5 (up to 1.0) |
| `oom` | Reduce batch size | Multiply batch size by 0.5 |
| `condition_number_high` | Upgrade precision | Switch to complex128 |

## Health Metrics

The StabilityMonitor tracks 20+ metrics:

**Failure Counts:**
- `nan_count`: Number of NaN detections
- `inf_count`: Number of Inf detections
- `gradient_explosion_count`: Number of gradient explosions
- `loss_divergence_count`: Number of loss divergences
- `oom_count`: Number of OOM errors

**Gradient Statistics:**
- `current_gradient_norm`: Current gradient norm
- `gradient_norm_mean`: Mean gradient norm
- `gradient_norm_std`: Standard deviation
- `gradient_norm_median`: Median gradient norm
- `gradient_norm_max/min`: Max/min gradient norms

**Loss Statistics:**
- `current_loss`: Current loss value
- `loss_mean`: Mean loss
- `loss_std`: Standard deviation
- `loss_median`: Median loss
- `loss_trend`: "improving", "stable", or "degrading"

**Numerical Stability:**
- `condition_numbers`: Condition numbers by module
- `schatten_norms`: Schatten norms by module
- `eigenvalue_stats`: Eigenvalue statistics

**Overall Health:**
- `overall_health`: "healthy", "warning", or "critical"
- `health_score`: 0-100 score (100 = perfect health)

## Colab-Specific Features

### Session Duration Tracking
```python
elapsed = timeout_handler.get_elapsed_time()
remaining = timeout_handler.get_remaining_time()
print(f"Elapsed: {elapsed.total_seconds()/3600:.2f} hours")
print(f"Remaining: {remaining.total_seconds()/60:.1f} minutes")
```

### Emergency Checkpoint Contents
- Model state dict
- Optimizer state dict
- Scheduler state dict (if provided)
- Training step and epoch
- Metrics history
- Random states (Python, NumPy, PyTorch, CUDA)
- Custom additional state
- Session information (elapsed/remaining time)

### Automatic Resume
```python
# Detects incomplete training automatically
resume_info = timeout_handler.auto_resume(model, optimizer, scheduler)

if resume_info:
    # Training state fully restored
    step = resume_info['step']
    epoch = resume_info['epoch']
    metrics = resume_info['metrics']
    # Random states are automatically restored
```

## Testing

Run the demo to test all features:
```bash
python examples/failure_recovery_demo.py
```

The demo includes:
- Artificial failure injection (NaN, gradient explosion)
- Automatic recovery testing
- Checkpoint saving and loading
- Timeout simulation
- Comprehensive statistics

## Files

**Core Implementation:**
- `src/training/stability_monitor.py` - Real-time health monitoring
- `src/training/auto_recovery.py` - Automatic failure recovery
- `src/training/colab_timeout_handler.py` - Colab timeout handling

**Examples:**
- `examples/failure_recovery_demo.py` - Complete demonstration

**Documentation:**
- `FAILURE_RECOVERY_QUICK_REFERENCE.md` - This file

## Requirements Mapping

| Requirement | Component | Status |
|-------------|-----------|--------|
| 12.1 | StabilityMonitor.check_nan_inf() | ✅ |
| 12.2 | AutoRecovery._rollback_checkpoint() | ✅ |
| 12.3 | StabilityMonitor.compute_gradient_norm() | ✅ |
| 12.4 | AutoRecovery._reduce_learning_rate() | ✅ |
| 12.5 | StabilityMonitor (loss divergence) | ✅ |
| 12.6 | AutoRecovery._increase_epsilon() | ✅ |
| 12.7 | AutoRecovery._reduce_batch_size() | ✅ |
| 12.8 | AutoRecovery (OOM recovery) | ✅ |
| 12.9 | AutoRecovery.verify_checkpoint() | ✅ |
| 12.10 | AutoRecovery (checkpoint queue) | ✅ |
| 12.11 | ColabTimeoutHandler.should_save_emergency_checkpoint() | ✅ |
| 12.12 | ColabTimeoutHandler.save_emergency_checkpoint() | ✅ |
| 12.13 | ColabTimeoutHandler.detect_incomplete_training() | ✅ |
| 12.14 | ColabTimeoutHandler.resume_from_checkpoint() | ✅ |
| 12.15 | StabilityMonitor.check_numerical_stability() | ✅ |
| 12.16 | AutoRecovery._upgrade_precision() | ✅ |
| 12.17 | StabilityMonitor.get_health_dashboard() | ✅ |
| 12.18 | StabilityMonitor.suggest_recovery() | ✅ |
| 12.19 | AutoRecovery (auto adjustment) | ✅ |
| 12.20 | AutoRecovery (max retries + halt) | ✅ |

## Best Practices

1. **Check Interval**: Use `check_interval=10` for development, `check_interval=50` for production
2. **Max Retries**: Set `max_retries=3` to prevent infinite recovery loops
3. **Checkpoint Frequency**: Save checkpoints every 50-100 steps
4. **Colab Sessions**: Always enable `enable_auto_resume=True` for Colab
5. **Emergency Checkpoints**: Include all necessary state for full recovery
6. **Health Monitoring**: Export metrics regularly for post-training analysis
7. **Recovery Testing**: Test recovery mechanisms before long training runs

## Performance Impact

- **StabilityMonitor**: ~0.1% overhead (checks every N steps)
- **AutoRecovery**: Negligible (only active during failures)
- **ColabTimeoutHandler**: ~0.01% overhead (checks every 60 seconds)
- **Checkpoint Saving**: ~1-2 seconds per checkpoint (depends on model size)

## Troubleshooting

**Q: Recovery keeps failing after 3 attempts**
A: Check the failure type and adjust thresholds. Some failures may require manual intervention.

**Q: Colab timeout not detected**
A: Ensure `session_duration_hours` matches your Colab tier (12h free, 24h Pro).

**Q: Checkpoint loading fails**
A: Use `verify_checkpoint()` to check integrity. May need to use earlier checkpoint.

**Q: Health score always low**
A: Review failure counts and adjust detection thresholds if too sensitive.

**Q: Random state not restored correctly**
A: Ensure all random seeds are set before calling `auto_resume()`.

## Related Documentation

- [Long Context Training](LONG_CONTEXT_TRAINING_QUICK_REFERENCE.md)
- [Reproducibility Package](REPRODUCIBILITY_QUICK_REFERENCE.md)
- [Stability Graph Generation](LONG_CONTEXT_STABILITY_GRAPH_QUICK_REFERENCE.md)

## Citation

If you use this failure recovery system, please cite:

```bibtex
@software{resnetbk_failure_recovery,
  title={Failure Recovery and Monitoring for Mamba-Killer ResNet-BK},
  author={ResNet-BK Team},
  year={2024},
  url={https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture}
}
```
