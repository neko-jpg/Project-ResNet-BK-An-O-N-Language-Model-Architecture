"""
Distributed Training Optimizations

Implements distributed training optimizations including:
- Overlapping communication and computation in DDP
- ZeRO optimizer (stage 1)

Based on Step 7 design for achieving 10× cost reduction through data efficiency.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional, Dict, List
import os


class ZeROOptimizer:
    """
    ZeRO (Zero Redundancy Optimizer) Stage 1 implementation.
    
    Implements Requirement 7.19:
    - Partition optimizer states across GPUs
    - Reduce memory usage while maintaining performance
    
    Stage 1: Partition optimizer states (momentum, variance)
    """
    
    def __init__(
        self,
        optimizer_class,
        model_params,
        world_size: int,
        rank: int,
        **optimizer_kwargs
    ):
        """
        Args:
            optimizer_class: Optimizer class (e.g., torch.optim.AdamW)
            model_params: Model parameters
            world_size: Number of processes
            rank: Current process rank
            **optimizer_kwargs: Optimizer arguments (lr, betas, etc.)
        """
        self.world_size = world_size
        self.rank = rank
        
        # Partition parameters across ranks
        all_params = list(model_params)
        self.all_params = all_params
        
        # Determine which parameters this rank is responsible for
        params_per_rank = len(all_params) // world_size
        start_idx = rank * params_per_rank
        end_idx = start_idx + params_per_rank if rank < world_size - 1 else len(all_params)
        
        self.local_params = all_params[start_idx:end_idx]
        
        # Create optimizer for local parameters only
        self.optimizer = optimizer_class(self.local_params, **optimizer_kwargs)
        
        print(f"[Rank {rank}] ZeRO Stage 1: Managing {len(self.local_params)}/{len(all_params)} parameters")
    
    def zero_grad(self):
        """Zero gradients for all parameters."""
        for param in self.all_params:
            if param.grad is not None:
                param.grad.zero_()
    
    def step(self):
        """
        Perform optimizer step.
        
        1. Each rank updates its local parameters
        2. Broadcast updated parameters to all ranks
        """
        # Update local parameters
        self.optimizer.step()
        
        # Broadcast updated parameters to all ranks
        for param in self.local_params:
            dist.broadcast(param.data, src=self.rank)
    
    def state_dict(self):
        """Return optimizer state dict."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load optimizer state dict."""
        self.optimizer.load_state_dict(state_dict)


class DistributedTrainer:
    """
    Distributed training with optimizations.
    
    Implements Requirements 7.18, 7.19:
    - Overlap communication and computation in DDP
    - Implement ZeRO optimizer (stage 1)
    """
    
    def __init__(
        self,
        model: nn.Module,
        rank: int,
        world_size: int,
        backend: str = 'nccl',
        use_zero: bool = False
    ):
        """
        Args:
            model: Model to train
            rank: Process rank
            world_size: Number of processes
            backend: Distributed backend ('nccl' for GPU, 'gloo' for CPU)
            use_zero: Whether to use ZeRO optimizer
        """
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.use_zero = use_zero
        
        # Setup distributed process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method='env://',
                world_size=world_size,
                rank=rank
            )
        
        # Move model to device
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        # Wrap model with DDP
        if not use_zero:
            self.model = DDP(
                self.model,
                device_ids=[rank] if torch.cuda.is_available() else None,
                find_unused_parameters=False,
                gradient_as_bucket_view=True  # Optimization: reduce memory copies
            )
        
        print(f"[Rank {rank}] Initialized distributed trainer (world_size={world_size})")
    
    def create_dataloader(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> DataLoader:
        """
        Create distributed dataloader.
        
        Args:
            dataset: Dataset
            batch_size: Batch size per GPU
            shuffle: Whether to shuffle
            num_workers: Number of data loading workers
        
        Returns:
            dataloader: Distributed dataloader
        """
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        return dataloader
    
    def create_optimizer(
        self,
        optimizer_class,
        **optimizer_kwargs
    ):
        """
        Create optimizer (ZeRO or standard).
        
        Args:
            optimizer_class: Optimizer class
            **optimizer_kwargs: Optimizer arguments
        
        Returns:
            optimizer: Optimizer instance
        """
        if self.use_zero:
            optimizer = ZeROOptimizer(
                optimizer_class,
                self.model.parameters(),
                world_size=self.world_size,
                rank=self.rank,
                **optimizer_kwargs
            )
        else:
            optimizer = optimizer_class(self.model.parameters(), **optimizer_kwargs)
        
        return optimizer
    
    def train_step(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        optimizer,
        criterion
    ) -> float:
        """
        Single training step with gradient synchronization.
        
        Args:
            x_batch: Input batch
            y_batch: Target batch
            optimizer: Optimizer
            criterion: Loss criterion
        
        Returns:
            loss: Scalar loss value
        """
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if hasattr(self.model, 'module'):
            # DDP wrapped model
            model = self.model.module
        else:
            model = self.model
        
        if hasattr(model, 'forward') and 'ponder_cost' in str(model.forward.__code__.co_varnames):
            logits, ponder_cost = self.model(x_batch)
            loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
            loss = loss + 0.01 * ponder_cost
        else:
            logits = self.model(x_batch)
            loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
        
        # Backward pass (gradients automatically synchronized in DDP)
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        return loss.item()
    
    def all_reduce_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Average metrics across all processes.
        
        Args:
            metrics: Dictionary of metrics
        
        Returns:
            averaged_metrics: Averaged metrics
        """
        averaged_metrics = {}
        
        for key, value in metrics.items():
            tensor = torch.tensor(value, device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            averaged_metrics[key] = (tensor / self.world_size).item()
        
        return averaged_metrics
    
    def cleanup(self):
        """Cleanup distributed process group."""
        if dist.is_initialized():
            dist.destroy_process_group()


def setup_distributed_training(
    rank: int,
    world_size: int,
    backend: str = 'nccl'
):
    """
    Setup distributed training environment.
    
    Args:
        rank: Process rank
        world_size: Number of processes
        backend: Distributed backend
    """
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def train_distributed(
    rank: int,
    world_size: int,
    model: nn.Module,
    train_dataset,
    optimizer_class,
    criterion,
    num_epochs: int = 5,
    batch_size: int = 32,
    use_zero: bool = False,
    backend: str = 'nccl',
    **optimizer_kwargs
) -> Dict:
    """
    Distributed training function.
    
    Args:
        rank: Process rank
        world_size: Number of processes
        model: Model to train
        train_dataset: Training dataset
        optimizer_class: Optimizer class
        criterion: Loss criterion
        num_epochs: Number of epochs
        batch_size: Batch size per GPU
        use_zero: Whether to use ZeRO optimizer
        backend: Distributed backend
        **optimizer_kwargs: Optimizer arguments
    
    Returns:
        metrics: Training metrics
    """
    # Setup
    setup_distributed_training(rank, world_size, backend)
    
    # Create trainer
    trainer = DistributedTrainer(
        model,
        rank=rank,
        world_size=world_size,
        backend=backend,
        use_zero=use_zero
    )
    
    # Create dataloader
    dataloader = trainer.create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Create optimizer
    optimizer = trainer.create_optimizer(optimizer_class, **optimizer_kwargs)
    
    if rank == 0:
        print(f"\nDistributed training:")
        print(f"  World size: {world_size}")
        print(f"  Batch size per GPU: {batch_size}")
        print(f"  Effective batch size: {batch_size * world_size}")
        print(f"  ZeRO optimizer: {use_zero}")
        print()
    
    # Training loop
    for epoch in range(num_epochs):
        trainer.model.train()
        epoch_loss = 0
        epoch_batches = 0
        
        # Set epoch for sampler (ensures different shuffle each epoch)
        dataloader.sampler.set_epoch(epoch)
        
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            loss = trainer.train_step(x_batch, y_batch, optimizer, criterion)
            
            epoch_loss += loss
            epoch_batches += 1
            
            if rank == 0 and (batch_idx + 1) % 50 == 0:
                avg_loss = epoch_loss / epoch_batches
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {avg_loss:.4f}")
        
        # Average metrics across all processes
        epoch_metrics = {'loss': epoch_loss / epoch_batches}
        averaged_metrics = trainer.all_reduce_metrics(epoch_metrics)
        
        if rank == 0:
            print(f"\nEpoch {epoch+1} completed: Avg Loss = {averaged_metrics['loss']:.4f}")
            print("-" * 60)
    
    # Cleanup
    trainer.cleanup()
    
    return {'final_loss': averaged_metrics['loss']}


def launch_distributed_training(
    model_fn,
    train_dataset,
    world_size: int,
    num_epochs: int = 5,
    batch_size: int = 32,
    use_zero: bool = False,
    **optimizer_kwargs
):
    """
    Launch distributed training across multiple processes.
    
    Args:
        model_fn: Function that returns model instance
        train_dataset: Training dataset
        world_size: Number of processes
        num_epochs: Number of epochs
        batch_size: Batch size per GPU
        use_zero: Whether to use ZeRO optimizer
        **optimizer_kwargs: Optimizer arguments
    """
    import torch.multiprocessing as mp
    
    if world_size > torch.cuda.device_count():
        print(f"Warning: world_size ({world_size}) > available GPUs ({torch.cuda.device_count()})")
        world_size = torch.cuda.device_count()
    
    print(f"Launching distributed training with {world_size} processes...")
    
    # Spawn processes
    mp.spawn(
        train_distributed,
        args=(
            world_size,
            model_fn(),
            train_dataset,
            torch.optim.AdamW,
            nn.CrossEntropyLoss(),
            num_epochs,
            batch_size,
            use_zero,
            'nccl' if torch.cuda.is_available() else 'gloo'
        ),
        nprocs=world_size,
        join=True
    )
    
    print("Distributed training completed!")


class GradientAccumulator:
    """
    Gradient accumulation for simulating larger batch sizes.
    
    Implements Requirement 7.18 (related):
    - Accumulate gradients over multiple steps
    - Simulate larger batch sizes without OOM
    """
    
    def __init__(
        self,
        model: nn.Module,
        accumulation_steps: int = 4
    ):
        """
        Args:
            model: Model to train
            accumulation_steps: Number of steps to accumulate gradients
        """
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
    
    def train_step(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        optimizer,
        criterion,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> tuple:
        """
        Training step with gradient accumulation.
        
        Args:
            x_batch: Input batch
            y_batch: Target batch
            optimizer: Optimizer
            criterion: Loss criterion
            device: Device for computation
        
        Returns:
            loss: Scalar loss value
            should_step: Whether optimizer should step
        """
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        if hasattr(self.model, 'forward') and 'ponder_cost' in str(self.model.forward.__code__.co_varnames):
            logits, ponder_cost = self.model(x_batch)
            loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
            loss = loss + 0.01 * ponder_cost
        else:
            logits = self.model(x_batch)
            loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
        
        # Scale loss by accumulation steps
        loss = loss / self.accumulation_steps
        
        # Backward pass
        loss.backward()
        
        self.current_step += 1
        
        # Check if we should step optimizer
        should_step = (self.current_step % self.accumulation_steps == 0)
        
        if should_step:
            optimizer.step()
            optimizer.zero_grad()
        
        return loss.item() * self.accumulation_steps, should_step
    
    def reset(self):
        """Reset accumulation counter."""
        self.current_step = 0


def train_with_gradient_accumulation(
    model: nn.Module,
    train_dataset,
    optimizer,
    criterion,
    num_epochs: int = 5,
    batch_size: int = 8,
    accumulation_steps: int = 4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """
    Train with gradient accumulation.
    
    Effective batch size = batch_size * accumulation_steps
    
    Args:
        model: Model to train
        train_dataset: Training dataset
        optimizer: Optimizer
        criterion: Loss criterion
        num_epochs: Number of epochs
        batch_size: Physical batch size
        accumulation_steps: Gradient accumulation steps
        device: Device for computation
    
    Returns:
        metrics: Training metrics
    """
    model = model.to(device)
    
    accumulator = GradientAccumulator(model, accumulation_steps)
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )
    
    print(f"Training with gradient accumulation:")
    print(f"  Physical batch size: {batch_size}")
    print(f"  Accumulation steps: {accumulation_steps}")
    print(f"  Effective batch size: {batch_size * accumulation_steps}")
    print()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_batches = 0
        
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            loss, stepped = accumulator.train_step(x_batch, y_batch, optimizer, criterion, device)
            
            epoch_loss += loss
            epoch_batches += 1
            
            if (batch_idx + 1) % 50 == 0:
                avg_loss = epoch_loss / epoch_batches
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {avg_loss:.4f}")
        
        avg_epoch_loss = epoch_loss / epoch_batches
        print(f"\nEpoch {epoch+1} completed: Avg Loss = {avg_epoch_loss:.4f}")
        print("-" * 60)
    
    return {'final_loss': avg_epoch_loss}
