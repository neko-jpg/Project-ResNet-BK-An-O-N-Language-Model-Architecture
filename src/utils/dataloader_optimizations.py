"""
Data Loading Optimizations for Maximum Throughput

Optimized DataLoader configuration for Phase 8 training.
"""

import torch
from torch.utils.data import DataLoader
from typing import Optional


def get_optimized_dataloader(
    dataset,
    batch_size: int = 1,
    num_workers: int = 8,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    drop_last: bool = False,
    shuffle: bool = True,
):
    """
    Create optimized DataLoader for maximum throughput.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        num_workers: Number of worker processes (8 recommended for high-core CPUs)
        pin_memory: Pin memory for faster H2D transfer (True recommended for CUDA)
        prefetch_factor: Number of batches to prefetch (4 recommended)
        persistent_workers: Keep workers alive between epochs (True recommended)
        drop_last: Drop last incomplete batch
        shuffle: Shuffle data
        
    Returns:
        Optimized DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=drop_last,
        multiprocessing_context='spawn' if num_workers > 0 else None,
    )


def apply_dataloader_optimizations(dataloader_kwargs: dict) -> dict:
    """
    Apply recommended optimizations to DataLoader kwargs.
    
    Args:
        dataloader_kwargs: Existing DataLoader arguments
        
    Returns:
        Optimized kwargs
    """
    # Set defaults if not specified
    optimizations = {
        'num_workers': 8,
        'pin_memory': torch.cuda.is_available(),
        'prefetch_factor': 4,
        'persistent_workers': True,
        'multiprocessing_context': 'spawn',
    }
    
    # Merge with existing kwargs (existing takes precedence)
    return {**optimizations, **dataloader_kwargs}
