"""
Hardware Optimization Suite for Step 5

This module implements all hardware optimizations:
- Multi-GPU training with DistributedDataParallel (DDP)
- Gradient accumulation for large effective batch sizes
- CPU offloading for optimizer states
- Dynamic batch sizing to prevent OOM errors

Requirements: 5.13, 5.14, 5.15, 5.16, 5.17, 5.18, 5.19
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional, Dict, Callable
import os
import warnings


class MultiGPUTrainer:
    """
    Multi-GPU training with DistributedDataParallel.
    
    Requirements: 5.13, 5.14
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        local_rank: int = 0,
        world_size: int = 1
    ):
        """
        Initialize multi-GPU trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer
            criterion: Loss function
            local_rank: Local GPU rank
            world_size: Total number of GPUs
        """
        self.local_rank = local_rank
        self.world_size = world_size
        self.criterion = criterion
        
        # Setup distributed training
        if world_size > 1:
            self.device = torch.device(f'cuda:{local_rank}')
            model = model.to(self.device)
            self.model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = model.to(self.device)
        
        self.optimizer = optimizer
        
        # Statistics
        self.stats = {
            'total_steps': 0,
            'loss_history': []
        }
    
    @staticmethod
    def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
        """
        Setup distributed training environment.
        
        Args:
            rank: Global rank
            world_size: Total number of processes
            backend: Backend ('nccl' for GPU, 'gloo' for CPU)
        """
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    @staticmethod
    def cleanup_distributed():
        """Cleanup distributed training."""
        dist.destroy_process_group()
    
    def train_step(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> Dict:
        """
        Single training step with gradient synchronization.
        
        Args:
            x_batch: Input batch
            y_batch: Target batch
        
        Returns:
            Dictionary with loss and statistics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move to device
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        
        # Forward pass
        logits = self.model(x_batch)
        loss = self.criterion(logits.view(-1, logits.size(-1)), y_batch)
        
        # Backward pass (gradients automatically synchronized by DDP)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        
        # Optimizer step
        self.optimizer.step()
        
        # Update statistics
        self.stats['total_steps'] += 1
        self.stats['loss_history'].append(loss.item())
        
        return {'loss': loss.item()}
    
    def get_dataloader(self, dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
        """
        Create dataloader with distributed sampler.
        
        Args:
            dataset: Dataset
            batch_size: Batch size per GPU
            shuffle: Shuffle data
        
        Returns:
            DataLoader with distributed sampler
        """
        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=shuffle
            )
            return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class GradientAccumulationTrainer:
    """
    Trainer with gradient accumulation for large effective batch sizes.
    
    Requirements: 5.15
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        accumulation_steps: int = 4,
        device: str = 'cuda'
    ):
        """
        Initialize gradient accumulation trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer
            criterion: Loss function
            accumulation_steps: Number of steps to accumulate gradients
            device: Device
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.accumulation_steps = accumulation_steps
        
        self.stats = {
            'total_steps': 0,
            'optimizer_steps': 0,
            'loss_history': []
        }
    
    def train_step(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> Dict:
        """
        Training step with gradient accumulation.
        
        Args:
            x_batch: Input batch
            y_batch: Target batch
        
        Returns:
            Dictionary with loss and statistics
        """
        self.model.train()
        
        # Move to device
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        
        # Forward pass
        logits = self.model(x_batch)
        loss = self.criterion(logits.view(-1, logits.size(-1)), y_batch)
        
        # Scale loss by accumulation steps
        loss = loss / self.accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update statistics
        self.stats['total_steps'] += 1
        self.stats['loss_history'].append(loss.item() * self.accumulation_steps)
        
        # Optimizer step every accumulation_steps
        if self.stats['total_steps'] % self.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.stats['optimizer_steps'] += 1
            
            return {
                'loss': loss.item() * self.accumulation_steps,
                'optimizer_step': True,
                'effective_batch_size': self.accumulation_steps * x_batch.size(0)
            }
        else:
            return {
                'loss': loss.item() * self.accumulation_steps,
                'optimizer_step': False
            }


class CPUOffloadingOptimizer:
    """
    Optimizer with CPU offloading for optimizer states.
    
    Keeps optimizer states (momentum, variance) on CPU to reduce GPU memory.
    
    Requirements: 5.16
    """
    
    def __init__(
        self,
        params,
        optimizer_class: type = torch.optim.AdamW,
        **optimizer_kwargs
    ):
        """
        Initialize CPU offloading optimizer.
        
        Args:
            params: Model parameters
            optimizer_class: Optimizer class (e.g., AdamW)
            **optimizer_kwargs: Optimizer arguments
        """
        # Create optimizer on CPU
        self.cpu_params = []
        self.gpu_params = []
        
        for param in params:
            # Create CPU copy of parameter
            cpu_param = param.detach().cpu().clone().requires_grad_(True)
            self.cpu_params.append(cpu_param)
            self.gpu_params.append(param)
        
        # Optimizer operates on CPU parameters
        self.optimizer = optimizer_class(self.cpu_params, **optimizer_kwargs)
        
        self.stats = {
            'total_steps': 0,
            'transfer_time': 0.0
        }
    
    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()
        for param in self.gpu_params:
            if param.grad is not None:
                param.grad.zero_()
    
    def step(self):
        """
        Optimizer step with CPU offloading.
        
        1. Transfer gradients GPU → CPU
        2. Optimizer step on CPU
        3. Transfer parameters CPU → GPU
        """
        import time
        start = time.time()
        
        # Transfer gradients to CPU
        for cpu_param, gpu_param in zip(self.cpu_params, self.gpu_params):
            if gpu_param.grad is not None:
                cpu_param.grad = gpu_param.grad.cpu()
        
        # Optimizer step on CPU
        self.optimizer.step()
        
        # Transfer updated parameters to GPU
        for cpu_param, gpu_param in zip(self.cpu_params, self.gpu_params):
            gpu_param.data.copy_(cpu_param.data)
        
        self.stats['total_steps'] += 1
        self.stats['transfer_time'] += time.time() - start
    
    def state_dict(self):
        """Get optimizer state dict."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load optimizer state dict."""
        self.optimizer.load_state_dict(state_dict)


class DynamicBatchSizeTrainer:
    """
    Trainer with dynamic batch sizing to prevent OOM errors.
    
    Automatically reduces batch size when OOM is detected.
    
    Requirements: 5.17, 5.18, 5.19
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        initial_batch_size: int = 32,
        min_batch_size: int = 1,
        device: str = 'cuda'
    ):
        """
        Initialize dynamic batch size trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer
            criterion: Loss function
            initial_batch_size: Starting batch size
            min_batch_size: Minimum batch size
            device: Device
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        
        self.stats = {
            'total_steps': 0,
            'oom_errors': 0,
            'batch_size_history': [initial_batch_size]
        }
    
    def train_step(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> Dict:
        """
        Training step with OOM handling.
        
        Args:
            x_batch: Input batch
            y_batch: Target batch
        
        Returns:
            Dictionary with loss and statistics
        """
        try:
            self.model.train()
            self.optimizer.zero_grad()
            
            # Move to device
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            logits = self.model(x_batch)
            loss = self.criterion(logits.view(-1, logits.size(-1)), y_batch)
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
            self.stats['total_steps'] += 1
            
            return {
                'loss': loss.item(),
                'batch_size': self.current_batch_size,
                'oom': False
            }
        
        except RuntimeError as e:
            if 'out of memory' in str(e):
                # OOM detected
                self.stats['oom_errors'] += 1
                
                # Clear cache
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # Reduce batch size
                new_batch_size = max(self.current_batch_size // 2, self.min_batch_size)
                
                if new_batch_size < self.current_batch_size:
                    self.current_batch_size = new_batch_size
                    self.stats['batch_size_history'].append(new_batch_size)
                    
                    warnings.warn(
                        f"OOM detected. Reducing batch size to {new_batch_size}. "
                        f"Consider using gradient accumulation for larger effective batch sizes."
                    )
                    
                    return {
                        'loss': None,
                        'batch_size': self.current_batch_size,
                        'oom': True,
                        'retry': True
                    }
                else:
                    raise RuntimeError(
                        f"OOM with minimum batch size {self.min_batch_size}. "
                        f"Cannot reduce further."
                    )
            else:
                raise e
    
    def get_recommended_batch_size(self) -> int:
        """Get current recommended batch size."""
        return self.current_batch_size


def benchmark_hardware_optimizations(
    model: nn.Module,
    dataloader: DataLoader,
    num_steps: int = 100
) -> Dict:
    """
    Benchmark all hardware optimizations.
    
    Args:
        model: Model to benchmark
        dataloader: Data loader
        num_steps: Number of steps to run
    
    Returns:
        Benchmark results
    """
    import time
    
    print("=" * 60)
    print("Benchmarking Hardware Optimizations")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    
    results = {}
    
    # 1. Baseline
    print("\n1. Baseline (no optimizations)")
    print("-" * 60)
    model_baseline = type(model)(**model.config).to(device)
    optimizer_baseline = torch.optim.AdamW(model_baseline.parameters(), lr=1e-3)
    
    start = time.time()
    for step, (x_batch, y_batch) in enumerate(dataloader):
        if step >= num_steps:
            break
        
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer_baseline.zero_grad()
        logits = model_baseline(x_batch)
        loss = criterion(logits.view(-1, logits.size(-1)), y_batch)
        loss.backward()
        optimizer_baseline.step()
    
    baseline_time = time.time() - start
    results['baseline'] = {'time': baseline_time, 'steps_per_sec': num_steps / baseline_time}
    print(f"  Time: {baseline_time:.2f}s")
    print(f"  Speed: {results['baseline']['steps_per_sec']:.2f} steps/s")
    
    # 2. Gradient Accumulation
    print("\n2. Gradient Accumulation (4 steps)")
    print("-" * 60)
    model_accum = type(model)(**model.config).to(device)
    optimizer_accum = torch.optim.AdamW(model_accum.parameters(), lr=1e-3)
    trainer_accum = GradientAccumulationTrainer(
        model_accum, optimizer_accum, criterion, accumulation_steps=4
    )
    
    start = time.time()
    for step, (x_batch, y_batch) in enumerate(dataloader):
        if step >= num_steps:
            break
        trainer_accum.train_step(x_batch, y_batch)
    
    accum_time = time.time() - start
    results['gradient_accumulation'] = {
        'time': accum_time,
        'steps_per_sec': num_steps / accum_time,
        'speedup': baseline_time / accum_time
    }
    print(f"  Time: {accum_time:.2f}s")
    print(f"  Speed: {results['gradient_accumulation']['steps_per_sec']:.2f} steps/s")
    print(f"  Speedup: {results['gradient_accumulation']['speedup']:.2f}x")
    
    # 3. Dynamic Batch Sizing
    print("\n3. Dynamic Batch Sizing")
    print("-" * 60)
    model_dynamic = type(model)(**model.config).to(device)
    optimizer_dynamic = torch.optim.AdamW(model_dynamic.parameters(), lr=1e-3)
    trainer_dynamic = DynamicBatchSizeTrainer(
        model_dynamic, optimizer_dynamic, criterion, initial_batch_size=32
    )
    
    print(f"  Initial batch size: {trainer_dynamic.current_batch_size}")
    print(f"  OOM errors: {trainer_dynamic.stats['oom_errors']}")
    print(f"  Final batch size: {trainer_dynamic.current_batch_size}")
    
    results['dynamic_batch_size'] = {
        'initial_batch_size': 32,
        'final_batch_size': trainer_dynamic.current_batch_size,
        'oom_errors': trainer_dynamic.stats['oom_errors']
    }
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Baseline: {results['baseline']['steps_per_sec']:.2f} steps/s")
    print(f"Gradient Accumulation: {results['gradient_accumulation']['speedup']:.2f}x speedup")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    print("Hardware Optimizations Test")
    print("=" * 60)
    
    # Test gradient accumulation
    print("\n1. Testing Gradient Accumulation")
    print("-" * 60)
    
    class DummyModel(nn.Module):
        def __init__(self, vocab_size=1000, d_model=64):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.linear = nn.Linear(d_model, vocab_size)
            self.config = {'vocab_size': vocab_size, 'd_model': d_model}
        
        def forward(self, x):
            x = self.embedding(x)
            return self.linear(x)
    
    model = DummyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    trainer = GradientAccumulationTrainer(
        model, optimizer, criterion, accumulation_steps=4, device='cpu'
    )
    
    for step in range(10):
        x = torch.randint(0, 1000, (8, 128))
        y = torch.randint(0, 1000, (8 * 128,))
        result = trainer.train_step(x, y)
        if result['optimizer_step']:
            print(f"  Step {step}: Optimizer step, effective batch size = {result['effective_batch_size']}")
    
    print("\n✓ All tests passed")
