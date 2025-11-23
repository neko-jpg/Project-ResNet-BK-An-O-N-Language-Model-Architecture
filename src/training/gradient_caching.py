"""
Gradient Caching Trainer

Implements gradient caching to reuse gradients from similar examples,
reducing backward pass frequency.

Based on Step 7 design for achieving 10× cost reduction through data efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import List, Optional, Tuple, Dict
import numpy as np


class GradientCachingTrainer:
    """
    Reuse gradients from similar examples to reduce backward pass frequency.
    
    Implements Requirements 7.11, 7.12:
    - Compute example embeddings
    - Cache gradients for similar examples
    - Reuse cached gradients when similarity > threshold
    """
    
    def __init__(
        self,
        model: nn.Module,
        cache_size: int = 100,
        similarity_threshold: float = 0.9,
        embedding_method: str = 'mean_token',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model: Model for training
            cache_size: Maximum number of cached gradient sets
            similarity_threshold: Minimum similarity to reuse cached gradients
            embedding_method: Method for computing example embeddings
                - 'mean_token': Mean of token embeddings
                - 'cls_token': First token embedding
                - 'last_hidden': Last layer hidden state
            device: Device for computation
        """
        self.model = model.to(device)
        self.device = device
        self.cache_size = cache_size
        self.similarity_threshold = similarity_threshold
        self.embedding_method = embedding_method
        
        # Cache: deque of (example_embedding, gradients)
        self.gradient_cache = deque(maxlen=cache_size)
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_queries = 0
    
    def compute_example_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute embedding for example x.
        
        Args:
            x: (batch_size, seq_len) or (seq_len,) tensor
        
        Returns:
            embedding: (embedding_dim,) tensor
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            if self.embedding_method == 'mean_token':
                # Mean of token embeddings
                if hasattr(self.model, 'token_embedding'):
                    emb = self.model.token_embedding(x)  # (batch_size, seq_len, d_model)
                    embedding = emb.mean(dim=(0, 1))  # (d_model,)
                else:
                    # Fallback: use input token IDs directly
                    embedding = x.float().mean(dim=(0, 1))
            
            elif self.embedding_method == 'cls_token':
                # First token embedding
                if hasattr(self.model, 'token_embedding'):
                    emb = self.model.token_embedding(x)
                    embedding = emb[:, 0, :].mean(dim=0)  # (d_model,)
                else:
                    embedding = x[:, 0].float()
            
            elif self.embedding_method == 'last_hidden':
                # Last layer hidden state
                # This requires a forward pass, so it's more expensive
                if hasattr(self.model, 'token_embedding'):
                    emb = self.model.token_embedding(x)
                    embedding = emb.mean(dim=(0, 1))
                else:
                    embedding = x.float().mean(dim=(0, 1))
            
            else:
                raise ValueError(f"Unknown embedding method: {self.embedding_method}")
        
        return embedding
    
    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """
        Compute cosine similarity between embeddings.
        
        Args:
            emb1: (embedding_dim,) tensor
            emb2: (embedding_dim,) tensor
        
        Returns:
            similarity: Scalar similarity score [0, 1]
        """
        # Flatten to 2D to support scalar (0-dim) embeddings from fallback paths
        emb1_flat = emb1.reshape(1, -1).float()
        emb2_flat = emb2.reshape(1, -1).float()
        similarity = F.cosine_similarity(emb1_flat, emb2_flat, dim=1)
        return similarity.item()
    
    def find_similar_cached(self, example_embedding: torch.Tensor) -> Optional[Tuple[List[torch.Tensor], float]]:
        """
        Find cached gradients for similar example.
        
        Args:
            example_embedding: (embedding_dim,) tensor
        
        Returns:
            Tuple of (cached_grads, cached_loss) or None if no match
        """
        self.total_queries += 1
        
        for cached_emb, cached_grads, cached_loss in self.gradient_cache:
            similarity = self.compute_similarity(example_embedding, cached_emb)
            
            if similarity >= self.similarity_threshold:
                self.cache_hits += 1
                return cached_grads, cached_loss
        
        self.cache_misses += 1
        return None
    
    def cache_gradients(self, example_embedding: torch.Tensor, loss_val: float):
        """
        Cache current gradients with example embedding and loss.
        
        Args:
            example_embedding: (embedding_dim,) tensor
            loss_val: scalar loss value
        """
        # Extract gradients from model
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.clone().detach())
            else:
                grads.append(None)
        
        # Add to cache
        self.gradient_cache.append((example_embedding.detach(), grads, loss_val))
    
    def apply_cached_gradients(self, cached_grads: List[torch.Tensor], scale: float = 1.0):
        """
        Apply cached gradients to model parameters.
        
        Args:
            cached_grads: List of gradient tensors
            scale: Scaling factor for gradients
        """
        for param, cached_grad in zip(self.model.parameters(), cached_grads):
            if cached_grad is not None:
                param.grad = cached_grad.clone() * scale
            else:
                param.grad = None
    
    def train_step(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        optimizer,
        criterion,
        force_compute: bool = False
    ) -> Tuple[float, bool]:
        """
        Training step with gradient caching.
        
        Args:
            x_batch: (batch_size, seq_len) input tensor
            y_batch: (batch_size, seq_len) target tensor
            optimizer: Optimizer
            criterion: Loss criterion
            force_compute: If True, always compute gradients (don't use cache)
        
        Returns:
            loss: Scalar loss value
            used_cache: Whether cached gradients were used
        """
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        
        # Compute example embedding
        example_emb = self.compute_example_embedding(x_batch)
        
        # Check cache (unless forced to compute)
        cache_result = None if force_compute else self.find_similar_cached(example_emb)
        
        if cache_result is not None:
            cached_grads, cached_loss_val = cache_result
            
            # Verification Step: Compute current loss to ensure safety
            # We must compute forward pass anyway to check loss deviation
            with torch.no_grad():
                if hasattr(self.model, 'forward') and 'ponder_cost' in str(self.model.forward.__code__.co_varnames):
                    logits, _ = self.model(x_batch)
                else:
                    logits = self.model(x_batch)
                
                current_loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))

            # Safety check: if loss deviation is too high, discard cache
            # This prevents reusing gradients when the function landscape has changed significantly
            loss_diff = abs(current_loss.item() - cached_loss_val)
            rel_diff = loss_diff / (abs(cached_loss_val) + 1e-9)
            
            if rel_diff > 0.1: # 10% tolerance threshold
                 # Discard cache, fallback to full backward
                 optimizer.zero_grad()
                 # Need to re-run forward with grad enabled
                 if hasattr(self.model, 'forward') and 'ponder_cost' in str(self.model.forward.__code__.co_varnames):
                    logits, ponder_cost = self.model(x_batch)
                    loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
                    loss = loss + 0.01 * ponder_cost
                 else:
                    logits = self.model(x_batch)
                    loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))

                 loss.backward()
                 self.cache_gradients(example_emb, loss.item())
                 optimizer.step()
                 return loss.item(), False # False means we computed fresh gradients

            else:
                # Cache is valid, apply gradients with scaling
                # Normalize gradient scale based on loss ratio (Requirement: Normalize gradient scale differences)
                scale_factor = current_loss.item() / (cached_loss_val + 1e-9)
                # Clamp scale to prevent instability
                scale_factor = max(0.5, min(2.0, scale_factor))

                optimizer.zero_grad()
                self.apply_cached_gradients(cached_grads, scale=scale_factor)
                optimizer.step()
                return current_loss.item(), True  # Used cache
        
        else:
            # Standard training step (compute gradients)
            optimizer.zero_grad()
            
            # Forward pass
            if hasattr(self.model, 'forward') and 'ponder_cost' in str(self.model.forward.__code__.co_varnames):
                logits, ponder_cost = self.model(x_batch)
                loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
                loss = loss + 0.01 * ponder_cost
            else:
                logits = self.model(x_batch)
                loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Cache gradients
            self.cache_gradients(example_emb, loss.item())
            
            # Optimizer step
            optimizer.step()
            
            return loss.item(), False  # Computed gradients
    
    def get_cache_statistics(self) -> Dict:
        """
        Get cache hit/miss statistics.
        
        Returns:
            stats: Dictionary with cache statistics
        """
        hit_rate = self.cache_hits / self.total_queries if self.total_queries > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_queries': self.total_queries,
            'hit_rate': hit_rate,
            'cache_size': len(self.gradient_cache),
            'max_cache_size': self.cache_size
        }
    
    def reset_statistics(self):
        """Reset cache statistics."""
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_queries = 0
    
    def clear_cache(self):
        """Clear gradient cache."""
        self.gradient_cache.clear()
        self.reset_statistics()


class AdaptiveGradientCachingTrainer(GradientCachingTrainer):
    """
    Adaptive gradient caching with dynamic similarity threshold.
    
    Adjusts similarity threshold based on cache performance.
    """
    
    def __init__(
        self,
        model: nn.Module,
        cache_size: int = 100,
        initial_similarity_threshold: float = 0.9,
        min_threshold: float = 0.7,
        max_threshold: float = 0.95,
        adaptation_rate: float = 0.01,
        target_hit_rate: float = 0.3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model: Model for training
            cache_size: Maximum cache size
            initial_similarity_threshold: Initial similarity threshold
            min_threshold: Minimum allowed threshold
            max_threshold: Maximum allowed threshold
            adaptation_rate: Rate of threshold adjustment
            target_hit_rate: Target cache hit rate
            device: Device for computation
        """
        super().__init__(
            model,
            cache_size=cache_size,
            similarity_threshold=initial_similarity_threshold,
            device=device
        )
        
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.adaptation_rate = adaptation_rate
        self.target_hit_rate = target_hit_rate
    
    def adapt_threshold(self):
        """
        Adapt similarity threshold based on cache performance.
        
        If hit rate too low: decrease threshold (more permissive)
        If hit rate too high: increase threshold (more selective)
        """
        if self.total_queries < 10:
            return  # Not enough data
        
        current_hit_rate = self.cache_hits / self.total_queries
        
        # Adjust threshold
        if current_hit_rate < self.target_hit_rate:
            # Hit rate too low, decrease threshold
            self.similarity_threshold -= self.adaptation_rate
        else:
            # Hit rate acceptable or high, increase threshold
            self.similarity_threshold += self.adaptation_rate
        
        # Clamp to valid range
        self.similarity_threshold = max(self.min_threshold, min(self.max_threshold, self.similarity_threshold))
    
    def train_step(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        optimizer,
        criterion,
        force_compute: bool = False
    ) -> Tuple[float, bool]:
        """Training step with adaptive threshold."""
        loss, used_cache = super().train_step(x_batch, y_batch, optimizer, criterion, force_compute)
        
        # Adapt threshold periodically
        if self.total_queries % 100 == 0:
            self.adapt_threshold()
        
        return loss, used_cache


def create_gradient_caching_trainer(
    model: nn.Module,
    cache_size: int = 100,
    similarity_threshold: float = 0.9,
    adaptive: bool = True,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> GradientCachingTrainer:
    """
    Create gradient caching trainer.
    
    Args:
        model: Model for training
        cache_size: Maximum cache size
        similarity_threshold: Similarity threshold for cache hits
        adaptive: Whether to use adaptive threshold
        device: Device for computation
    
    Returns:
        trainer: GradientCachingTrainer instance
    """
    if adaptive:
        trainer = AdaptiveGradientCachingTrainer(
            model,
            cache_size=cache_size,
            initial_similarity_threshold=similarity_threshold,
            device=device
        )
        print(f"Created adaptive gradient caching trainer:")
        print(f"  Cache size: {cache_size}")
        print(f"  Initial similarity threshold: {similarity_threshold}")
        print(f"  Adaptive threshold: enabled")
    else:
        trainer = GradientCachingTrainer(
            model,
            cache_size=cache_size,
            similarity_threshold=similarity_threshold,
            device=device
        )
        print(f"Created gradient caching trainer:")
        print(f"  Cache size: {cache_size}")
        print(f"  Similarity threshold: {similarity_threshold}")
        print(f"  Adaptive threshold: disabled")
    
    return trainer


def train_with_gradient_caching(
    model: nn.Module,
    train_dataset,
    optimizer,
    criterion,
    num_epochs: int = 5,
    batch_size: int = 32,
    cache_size: int = 100,
    similarity_threshold: float = 0.9,
    adaptive: bool = True,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """
    Train model with gradient caching.
    
    Args:
        model: Model to train
        train_dataset: Training dataset
        optimizer: Optimizer
        criterion: Loss criterion
        num_epochs: Number of epochs
        batch_size: Batch size
        cache_size: Gradient cache size
        similarity_threshold: Similarity threshold
        adaptive: Use adaptive threshold
        device: Device for computation
    
    Returns:
        metrics: Training metrics including cache statistics
    """
    from torch.utils.data import DataLoader
    
    # Create trainer
    trainer = create_gradient_caching_trainer(
        model,
        cache_size=cache_size,
        similarity_threshold=similarity_threshold,
        adaptive=adaptive,
        device=device
    )
    
    # Create dataloader
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )
    
    print(f"\nTraining with gradient caching:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Dataset size: {len(train_dataset)}")
    print()
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_batches = 0
        epoch_cache_hits = 0
        
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            loss, used_cache = trainer.train_step(x_batch, y_batch, optimizer, criterion)
            
            epoch_loss += loss
            epoch_batches += 1
            if used_cache:
                epoch_cache_hits += 1
            
            if (batch_idx + 1) % 50 == 0:
                avg_loss = epoch_loss / epoch_batches
                cache_hit_rate = epoch_cache_hits / epoch_batches
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                      f"Loss: {avg_loss:.4f}, Cache Hit Rate: {cache_hit_rate:.2%}")
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / epoch_batches
        epoch_hit_rate = epoch_cache_hits / epoch_batches
        
        print(f"\nEpoch {epoch+1} completed:")
        print(f"  Avg Loss: {avg_epoch_loss:.4f}")
        print(f"  Cache Hit Rate: {epoch_hit_rate:.2%}")
        
        if adaptive and isinstance(trainer, AdaptiveGradientCachingTrainer):
            print(f"  Current Similarity Threshold: {trainer.similarity_threshold:.3f}")
        
        print("-" * 60)
    
    # Final statistics
    final_stats = trainer.get_cache_statistics()
    
    print("\n" + "=" * 60)
    print("GRADIENT CACHING STATISTICS")
    print("=" * 60)
    print(f"Total queries: {final_stats['total_queries']}")
    print(f"Cache hits: {final_stats['cache_hits']}")
    print(f"Cache misses: {final_stats['cache_misses']}")
    print(f"Hit rate: {final_stats['hit_rate']:.2%}")
    print(f"Cache size: {final_stats['cache_size']}/{final_stats['max_cache_size']}")
    
    return final_stats
