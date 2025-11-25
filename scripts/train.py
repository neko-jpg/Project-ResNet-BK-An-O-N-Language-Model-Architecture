"""
ResNet-BK Training Script

Train ResNet-BK models with configurable optimizations.

Usage:
    python train.py --config-preset baseline
    python train.py --config-preset full --epochs 5
    python train.py --help
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import math
from pathlib import Path
import warnings
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn

from src.models.configurable_resnet_bk import ConfigurableResNetBK
import subprocess
from src.utils import (
    parse_args,
    get_config_from_args,
    get_data_loader,
    get_mixed_data_loader,
    TrainingMetrics,
    MetricsLogger,
    WandBLogger,
)
from src.training.curriculum import CurriculumScheduler
from src.eval.skill_bench import SkillEvaluator
from scripts.calibration import MuseCalibrator

# Silence noisy warnings early
warnings.filterwarnings("ignore", message=".*_register_pytree_node is deprecated.*")
warnings.filterwarnings("ignore", message=".*torch.utils._pytree.*deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Triton kernel failed or unstable.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Current implementation only support single tensor input.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*To copy construct from a tensor.*")


def train():
    """Main training function."""
    # Silence noisy deprecation warnings from pytree/transformers
    warnings.filterwarnings("ignore", message=".*_register_pytree_node is deprecated.*")
    warnings.filterwarnings("ignore", message=".*torch.utils._pytree.*deprecated.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Triton kernel failed or unstable.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Current implementation only support single tensor input.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*To copy construct from a tensor.*")
    # Parse arguments
    args = parse_args()

    # Strict Triton Check for Training
    # We must ensure Triton is available if we are running the O(N) kernel on GPU
    # If device is CPU, MuseCalibrator handles it gracefully (returns True)
    # If device is CUDA, strict=True will exit if Triton is missing.
    cal = MuseCalibrator()
    if args.device != 'cpu': # Only check strict if potentially using GPU
        cal.check_triton(strict=True)

    # Build config early so that sequence length and related params propagate to data loaders
    config = get_config_from_args(args)
    if hasattr(config, "n_seq"):
        args.n_seq = config.n_seq
    # Prefer CUDA-friendly defaults when using Phase 4
    if torch.cuda.is_available():
        if not getattr(config, "use_mixed_precision", False):
            config.use_mixed_precision = True
            if hasattr(args, "use_mixed_precision"):
                args.use_mixed_precision = True
            print("Enabling mixed precision for CUDA to reduce VRAM pressure.")
        if getattr(config, "use_symplectic", False) and not getattr(config, "use_gradient_checkpointing", False):
            config.use_gradient_checkpointing = True
            if hasattr(args, "use_gradient_checkpointing"):
                args.use_gradient_checkpointing = True
            print("Enabling gradient checkpointing for Symplectic mode.")
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")

    # Pre-flight VRAM guard (conservative) to avoid CUDA OOM
    if device.type == 'cuda' and cal.vram_total > 0:
        target_limit = cal.vram_total * 0.78  # keep more headroom for activations
        safety_kwargs = {
            'use_symplectic': getattr(config, 'use_symplectic', False),
            'use_gradient_checkpointing': getattr(config, 'use_gradient_checkpointing', False),
            'use_bitnet': getattr(config, 'use_bitnet', False),
        }
        est_mem, _ = cal.predict(args.batch_size, args.n_seq, args.d_model, args.n_layers, **safety_kwargs)
        if est_mem > target_limit:
            print(f"Preflight estimate {est_mem:.0f}MB > safe limit {target_limit:.0f}MB. Shrinking config...")
            # Reduce batch size first
            while est_mem > target_limit and args.batch_size > 1:
                args.batch_size = max(1, args.batch_size // 2)
                config.batch_size = args.batch_size
                est_mem, _ = cal.predict(args.batch_size, args.n_seq, args.d_model, args.n_layers, **safety_kwargs)
            # Reduce d_model next
            while est_mem > target_limit and args.d_model > 256:
                args.d_model = max(256, args.d_model - 128)
                if getattr(config, 'use_symplectic', False) and args.d_model % 2 != 0:
                    args.d_model -= 1
                config.d_model = args.d_model
                est_mem, _ = cal.predict(args.batch_size, args.n_seq, args.d_model, args.n_layers, **safety_kwargs)
            # Reduce depth if still heavy
            while est_mem > target_limit and args.n_layers > 2:
                args.n_layers = max(2, args.n_layers - 1)
                config.n_layers = args.n_layers
                est_mem, _ = cal.predict(args.batch_size, args.n_seq, args.d_model, args.n_layers, **safety_kwargs)
            # Shorten sequence length last
            while est_mem > target_limit and args.n_seq > 256:
                args.n_seq = max(256, args.n_seq - 128)
                config.n_seq = args.n_seq
                est_mem, _ = cal.predict(args.batch_size, args.n_seq, args.d_model, args.n_layers, **safety_kwargs)
            print(f"Using safe config: batch_size={args.batch_size}, d_model={args.d_model}, n_layers={args.n_layers}, n_seq={args.n_seq} (est. {est_mem:.0f}MB)")
    
    # Load data
    print("\nLoading data...")
    use_mixed = args.use_mixed_datasets or args.dataset in ("mixed", "phase4", "phase4-mix") or args.dataset.endswith(".yaml")

    val_loader = None
    val_steps_per_epoch = 0

    if use_mixed:
        config_path = args.dataset_mix_config if not args.dataset.endswith(".yaml") else args.dataset
        mixed_loader, vocab, steps_per_epoch = get_mixed_data_loader(
            config_path=config_path,
            batch_size=args.batch_size,
            n_seq=args.n_seq,
            total_tokens=args.data_limit,
            seed=args.seed,
            vocab_size=args.vocab_size,
            split='train'
        )
        train_data = None
        get_batch = None
        training_tokens = steps_per_epoch * args.batch_size * args.n_seq
        print(f"Vocabulary size: {vocab['vocab_size']}")
        print(f"Training tokens (per epoch): {training_tokens}")

        # Try to load validation data
        try:
            val_loader, _, val_steps_per_epoch = get_mixed_data_loader(
                config_path=config_path,
                batch_size=args.batch_size,
                n_seq=args.n_seq,
                total_tokens=max(10000, args.data_limit // 10), # 10% size or min 10k
                seed=args.seed + 999, # Different seed
                vocab_size=vocab['vocab_size'], # Use training vocab size
                split='validation'
            )
            print(f"Validation enabled. Steps per epoch: {val_steps_per_epoch}")
        except Exception as e:
            print(f"Validation loader skipped: {e}")
            val_loader = None
    else:
        train_data, vocab, get_batch = get_data_loader(
            batch_size=args.batch_size,
            n_seq=args.n_seq,
            dataset_name=args.dataset,
            data_limit=args.data_limit
        )
        
        if train_data is None:
            print("Failed to load data. Exiting.")
            return
        
        training_tokens = train_data.numel()
        print(f"Vocabulary size: {vocab['vocab_size']}")
        print(f"Training tokens: {training_tokens}")
    
    # Finalize model configuration now that vocab size is known
    config.vocab_size = vocab['vocab_size']
    config.n_seq = args.n_seq
    args.vocab_size = config.vocab_size
    
    # Create model
    print("\nCreating model...")
    model_type = getattr(config, 'model_type', 'phase3')

    if model_type == 'phase7':
        from src.models.phase7.hybrid_attention import HybridHyperbolicAttention

        class Phase7Model(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.embedding = nn.Embedding(config.vocab_size, config.d_model)
                self.core_model = HybridHyperbolicAttention(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    local_window_size=config.local_window_size
                )
                self.to_logits = nn.Linear(config.d_model, config.vocab_size)
                self.config = config

            def forward(self, x):
                x = self.embedding(x)
                # The hybrid model now returns diagnostics, which we can log
                # For now, we ignore them in the main forward pass for loss calculation
                x, diagnostics = self.core_model(x, return_diagnostics=True)
                return self.to_logits(x)

            def get_config_summary(self):
                 return {
                    "Model Type": "Phase 7 Hybrid Hyperbolic",
                    "d_model": self.config.d_model,
                    "n_layers": "N/A (Hybrid)",
                    "num_heads": self.config.num_heads,
                    "local_window_size": self.config.local_window_size,
                }

        model = Phase7Model(config).to(device)

    else: # Phase 3
        model = ConfigurableResNetBK(config).to(device)
    
    # Print model summary
    summary = model.get_config_summary()
    print("\nModel Configuration:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Resume from checkpoint
    if args.resume_from:
        print(f"Loading checkpoint: {args.resume_from}")
        try:
            checkpoint = torch.load(args.resume_from, map_location=device)
            model.model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ Resume successful")
        except Exception as e:
            print(f"Error resuming from checkpoint: {e}")
            return
    
    criterion = nn.CrossEntropyLoss()
    
    if use_mixed:
        num_total_steps = steps_per_epoch * args.epochs
    else:
        num_total_steps = (train_data.size(0) // args.n_seq) * args.epochs
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_total_steps,
        eta_min=args.lr / 10
    )
    use_amp = getattr(config, "use_mixed_precision", False) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # Curriculum Scheduler (only for mixed data)
    curriculum = None
    if use_mixed:
        curriculum = CurriculumScheduler(mixed_loader, window_size=50, threshold=0.001)
        print("Enable Curriculum Optimizer: ON")

    # Setup logging
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger = MetricsLogger(
        log_dir=str(save_dir / "logs"),
        experiment_name=f"resnet_bk_{args.config_preset}"
    )

    # Skill Evaluator (Future IQ)
    skill_evaluator = SkillEvaluator(device=device)
    
    # Optional W&B logging
    wandb_logger = WandBLogger(
        project="resnet-bk",
        name=f"{args.config_preset}_d{args.d_model}_l{args.n_layers}",
        config=vars(args),
        enabled=False  # Set to True to enable W&B
    )
    
    print(f"\nTotal training steps: {num_total_steps}")
    print(f"Logging to: {save_dir / 'logs'}")
    print("\nStarting training...\n")
    
    # Training loop
    model.train()
    global_step = 0
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        total_loss = 0.0
        num_batches = 0

        if use_mixed:
            batch_iter = mixed_loader.iter_epoch(epoch)
            steps_in_epoch = mixed_loader.steps_per_epoch
        else:
            batch_iter = range(0, train_data.size(0) - 1, args.n_seq)
            steps_in_epoch = max(1, train_data.size(0) // args.n_seq)

        progress = Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True,
        )

        with progress:
            task = progress.add_task(f"Epoch {epoch}/{args.epochs}", total=steps_in_epoch)

            for step_idx, batch_item in enumerate(batch_iter, start=1):
                step_start = time.time()

                if use_mixed:
                    x_batch, y_batch = batch_item
                else:
                    x_batch, y_batch = get_batch(train_data, batch_item)
                    x_batch = x_batch.t().contiguous()

                    if x_batch.size(1) != args.n_seq:
                        progress.update(task, advance=1)
                        continue

                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                try:
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        logits = model(x_batch)
                        loss = criterion(logits.view(-1, logits.size(-1)), y_batch)

                    # Skip if loss is NaN/Inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        progress.update(task, advance=1)
                        continue

                    if use_amp:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            args.grad_clip
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            args.grad_clip
                        )
                        optimizer.step()
                    scheduler.step()
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print("CUDA OOM detected. Clearing cache and skipping this step.")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        progress.update(task, advance=1)
                        continue
                    else:
                        raise

                # Curriculum Step
                if curriculum:
                    curriculum.step(loss.item())

                global_step += 1

                step_time = time.time() - step_start
                total_loss += loss.item()
                num_batches += 1

                # Log metrics
                if global_step % args.log_interval == 0:
                    routing_entropy = getattr(model.model, "last_routing_entropy", None)

                    metrics = TrainingMetrics(
                        step=global_step,
                        epoch=epoch,
                        loss=loss.item(),
                        learning_rate=scheduler.get_last_lr()[0],
                        step_time=step_time,
                        grad_norm=grad_norm.item(),
                        routing_entropy=float(routing_entropy) if routing_entropy is not None else 0.0,
                    )

                    # Run Skill Evaluation (Every 100 steps to avoid slowdown)
                    skill_scores = {}
                    if global_step % 100 == 0:
                        skill_scores = skill_evaluator.evaluate(model)

                        # Log to CSV
                        skills_log_path = save_dir / "logs" / "skills.csv"
                        params_exist = skills_log_path.exists()
                        with open(skills_log_path, "a") as f:
                            if not params_exist:
                                f.write("step," + ",".join(skill_scores.keys()) + "\n")
                            f.write(f"{global_step}," + ",".join([f"{v:.2f}" for v in skill_scores.values()]) + "\n")

                    logger.log(metrics)

                    # Get stability diagnostics if using Birman-Schwinger
                    stability_diagnostics = {}
                    if hasattr(model.model, 'get_stability_diagnostics'):
                        stability_diagnostics = model.model.get_stability_diagnostics()

                    # Log to W&B
                    wandb_log_dict = {
                        'loss': loss.item(),
                        'perplexity': metrics.perplexity,
                        'learning_rate': metrics.learning_rate,
                        'grad_norm': grad_norm.item(),
                    }

                    # Log Skills
                    if skill_scores:
                        for skill, score in skill_scores.items():
                            wandb_log_dict[f'skills/{skill}'] = score

                    # Add stability diagnostics to W&B logging
                    if stability_diagnostics:
                        wandb_log_dict.update({
                            'stability/mean_schatten_s1': stability_diagnostics.get('mean_schatten_s1', 0.0),
                            'stability/mean_schatten_s2': stability_diagnostics.get('mean_schatten_s2', 0.0),
                            'stability/max_schatten_s1': stability_diagnostics.get('max_schatten_s1', 0.0),
                            'stability/max_schatten_s2': stability_diagnostics.get('max_schatten_s2', 0.0),
                            'stability/mean_condition_number': stability_diagnostics.get('mean_condition_number', 0.0),
                            'stability/max_condition_number': stability_diagnostics.get('max_condition_number', 0.0),
                            'stability/mourre_verified_rate': stability_diagnostics.get('mourre_verified_rate', 0.0),
                            'stability/s1_bound_satisfied_rate': stability_diagnostics.get('s1_bound_satisfied_rate', 0.0),
                            'stability/s2_bound_satisfied_rate': stability_diagnostics.get('s2_bound_satisfied_rate', 0.0),
                            'stability/all_finite_rate': stability_diagnostics.get('all_finite_rate', 1.0),
                            'stability/precision_upgrades': stability_diagnostics.get('precision_upgrades', 0),
                        })

                    wandb_logger.log(wandb_log_dict, step=global_step)

                progress.update(task, advance=1)
        
        # Validation Loop
        val_metrics = {}
        if val_loader:
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            val_iter = val_loader.iter_epoch(epoch)

            print("Running Validation...")
            with torch.no_grad():
                for vx, vy in val_iter:
                    vx, vy = vx.to(device), vy.to(device)
                    # Handle potential shape mismatch
                    if vx.size(1) != args.n_seq:
                        continue

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        logits = model(vx)
                        v_loss = criterion(logits.view(-1, logits.size(-1)), vy)

                    if not torch.isnan(v_loss) and not torch.isinf(v_loss):
                        val_loss_sum += v_loss.item()
                        val_batches += 1

            if val_batches > 0:
                avg_val_loss = val_loss_sum / val_batches
                val_perplexity = math.exp(min(avg_val_loss, 20))
                val_metrics = {
                    'val_loss': avg_val_loss,
                    'val_perplexity': val_perplexity
                }

            model.train()

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / max(1, num_batches)
        perplexity = math.exp(min(avg_loss, 20))
        
        print(f"\nEpoch {epoch}/{args.epochs} Summary:")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")

        if val_metrics:
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val PPL:  {val_metrics['val_perplexity']:.2f}")
        
        # Print stability diagnostics if using Birman-Schwinger
        if hasattr(model.model, 'get_stability_diagnostics'):
            stability_diagnostics = model.model.get_stability_diagnostics()
            if stability_diagnostics:
                print(f"  Stability Diagnostics:")
                print(f"    Mean Schatten S2: {stability_diagnostics.get('mean_schatten_s2', 0.0):.4f}")
                print(f"    Max Condition Number: {stability_diagnostics.get('max_condition_number', 0.0):.2e}")
                print(f"    Mourre Verified: {stability_diagnostics.get('mourre_verified_rate', 0.0):.1%}")
                print(f"    All Finite: {stability_diagnostics.get('all_finite_rate', 1.0):.1%}")
                if stability_diagnostics.get('precision_upgrades', 0) > 0:
                    print(f"    Precision Upgrades: {stability_diagnostics.get('precision_upgrades', 0)}")
        print()
    
    # Save final metrics and model
    logger.save_json()
    logger.print_summary()
    
    # Get final stability diagnostics
    final_stability_diagnostics = {}
    if hasattr(model.model, 'get_stability_diagnostics'):
        final_stability_diagnostics = model.model.get_stability_diagnostics()
    
    checkpoint_path = save_dir / f"resnet_bk_{args.config_preset}_final.pt"
    checkpoint = {
        'model_state_dict': model.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'args': vars(args),
        'metrics': logger.get_summary(),
        'stability_diagnostics': final_stability_diagnostics,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"\nCheckpoint saved to: {checkpoint_path}")
    
    # Print final stability summary
    if final_stability_diagnostics:
        print("\nFinal Stability Summary:")
        print(f"  Mean Schatten S1: {final_stability_diagnostics.get('mean_schatten_s1', 0.0):.4f}")
        print(f"  Mean Schatten S2: {final_stability_diagnostics.get('mean_schatten_s2', 0.0):.4f}")
        print(f"  Mean Condition Number: {final_stability_diagnostics.get('mean_condition_number', 0.0):.2e}")
        print(f"  Max Condition Number: {final_stability_diagnostics.get('max_condition_number', 0.0):.2e}")
        print(f"  Mourre Verified Rate: {final_stability_diagnostics.get('mourre_verified_rate', 0.0):.1%}")
        print(f"  S1 Bound Satisfied Rate: {final_stability_diagnostics.get('s1_bound_satisfied_rate', 0.0):.1%}")
        print(f"  S2 Bound Satisfied Rate: {final_stability_diagnostics.get('s2_bound_satisfied_rate', 0.0):.1%}")
        print(f"  All Finite Rate: {final_stability_diagnostics.get('all_finite_rate', 1.0):.1%}")
        print(f"  Total Precision Upgrades: {final_stability_diagnostics.get('precision_upgrades', 0)}")
    
    wandb_logger.finish()
    print("\n✓ Training complete!")

    # Notifier
    try:
        subprocess.run(["python3", "scripts/muse_utils.py", "notify", "--message", f"Training Complete: {args.config_preset}"])
    except:
        pass


if __name__ == "__main__":
    train()
