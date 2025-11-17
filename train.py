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

from src.models.configurable_resnet_bk import ConfigurableResNetBK
from src.utils import (
    parse_args,
    get_config_from_args,
    get_data_loader,
    TrainingMetrics,
    MetricsLogger,
    WandBLogger,
)


def train():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    train_data, vocab, get_batch = get_data_loader(
        batch_size=args.batch_size,
        n_seq=args.n_seq,
        dataset_name=args.dataset,
        data_limit=args.data_limit
    )
    
    if train_data is None:
        print("Failed to load data. Exiting.")
        return
    
    print(f"Vocabulary size: {vocab['vocab_size']}")
    print(f"Training tokens: {train_data.numel()}")
    
    # Create model configuration
    config = get_config_from_args(args)
    config.vocab_size = vocab['vocab_size']
    
    # Create model
    print("\nCreating model...")
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
    
    criterion = nn.CrossEntropyLoss()
    
    num_total_steps = (train_data.size(0) // args.n_seq) * args.epochs
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_total_steps,
        eta_min=args.lr / 10
    )
    
    # Setup logging
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger = MetricsLogger(
        log_dir=str(save_dir / "logs"),
        experiment_name=f"resnet_bk_{args.config_preset}"
    )
    
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
        
        for i in range(0, train_data.size(0) - 1, args.n_seq):
            step_start = time.time()
            
            x_batch, y_batch = get_batch(train_data, i)
            x_batch = x_batch.t().contiguous()
            
            if x_batch.size(1) != args.n_seq:
                continue
            
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits.view(-1, logits.size(-1)), y_batch)
            
            # Skip if loss is NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss at step {global_step}, skipping")
                continue
            
            # Backward pass
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip
            )
            optimizer.step()
            scheduler.step()
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
                logger.log(metrics)
                
                # Log to W&B
                wandb_logger.log({
                    'loss': loss.item(),
                    'perplexity': metrics.perplexity,
                    'learning_rate': metrics.learning_rate,
                    'grad_norm': grad_norm.item(),
                }, step=global_step)
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / max(1, num_batches)
        perplexity = math.exp(min(avg_loss, 20))
        
        print(f"\nEpoch {epoch}/{args.epochs} Summary:")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}\n")
    
    # Save final metrics and model
    logger.save_json()
    logger.print_summary()
    
    checkpoint_path = save_dir / f"resnet_bk_{args.config_preset}_final.pt"
    checkpoint = {
        'model_state_dict': model.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'args': vars(args),
        'metrics': logger.get_summary(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"\nCheckpoint saved to: {checkpoint_path}")
    
    wandb_logger.finish()
    print("\n✓ Training complete!")


if __name__ == "__main__":
    train()
