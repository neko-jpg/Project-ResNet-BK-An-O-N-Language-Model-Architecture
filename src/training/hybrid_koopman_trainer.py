"""
Hybrid Koopman-Gradient Trainer
Implements phased training combining gradient-based and Koopman-based learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .koopman_scheduler import KoopmanLossScheduler


class HybridKoopmanTrainer:
    """
    Training loop with hybrid Koopman-gradient learning.
    
    Training phases:
    1. Gradient warmup (epochs 0-2): Standard gradient-based training
    2. Hybrid phase (epochs 3-5): Gradient + Koopman auxiliary loss
    3. Koopman-dominant (epochs 6+): Gradually increase Koopman weight
    
    Features:
    - Automatic fallback to gradients if Koopman fails
    - Koopman operator updates using streaming DMD
    - Loss weight scheduling
    - Comprehensive metrics tracking
    """
    
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        koopman_weight_min=0.0,
        koopman_weight_max=0.5,
        koopman_start_epoch=2,
        total_epochs=10,
        schedule_type='linear',
        enable_koopman_updates=True,
        fallback_threshold=10.0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize hybrid Koopman trainer.
        
        Args:
            model: KoopmanLanguageModel instance
            optimizer: PyTorch optimizer
            criterion: loss function (e.g., CrossEntropyLoss)
            koopman_weight_min: minimum Koopman loss weight
            koopman_weight_max: maximum Koopman loss weight
            koopman_start_epoch: epoch to start Koopman learning
            total_epochs: total number of training epochs
            schedule_type: 'linear', 'exponential', or 'step'
            enable_koopman_updates: whether to update Koopman operators
            fallback_threshold: if Koopman loss exceeds this, fallback to gradients
            device: device to use for training
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        self.koopman_start_epoch = koopman_start_epoch
        self.enable_koopman_updates = enable_koopman_updates
        self.fallback_threshold = fallback_threshold
        
        # Koopman loss scheduler
        self.koopman_scheduler = KoopmanLossScheduler(
            min_weight=koopman_weight_min,
            max_weight=koopman_weight_max,
            warmup_epochs=koopman_start_epoch,
            total_epochs=total_epochs,
            schedule_type=schedule_type
        )
        
        self.current_epoch = 0
        self.koopman_enabled = False
        self.koopman_failed = False
        
        # Move model to device
        self.model.to(device)
    
    def train_step(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor
    ) -> Dict[str, float]:
        """
        Single training step with hybrid Koopman-gradient learning.
        
        Args:
            x_batch: (B, N) input token indices
            y_batch: (B*N,) target token indices (flattened)
        
        Returns:
            metrics: dictionary of training metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        
        # Phase 1: Standard forward pass with gradient-based learning
        # Get hidden states at each layer for Koopman operator updates
        hidden_states = []
        
        # Manual forward pass to capture intermediate states
        batch_size, n_seq = x_batch.shape
        tok_emb = self.model.token_embedding(x_batch)
        pos = torch.arange(0, n_seq, dtype=torch.long, device=self.device).unsqueeze(0)
        pos_emb = self.model.position_embedding(pos)
        h = tok_emb + pos_emb
        
        for block in self.model.blocks:
            h_prev = h
            h = block(h, use_koopman=False)  # Standard forward
            hidden_states.append((h_prev, h))
        
        # Output
        h = self.model.layer_norm_final(h)
        logits = self.model.lm_head(h)
        
        # Language modeling loss
        loss_lm = self.criterion(logits.view(-1, logits.size(-1)), y_batch)
        
        # Phase 2: Koopman auxiliary loss (if enabled)
        loss_koopman = torch.tensor(0.0, device=self.device)
        koopman_weight = 0.0
        
        if self.koopman_enabled and not self.koopman_failed:
            try:
                # Compute Koopman loss for each layer
                koopman_losses = []
                for layer, (h_prev, h_next) in zip(self.model.blocks, hidden_states):
                    layer_loss = layer.bk_layer.koopman_loss(h_prev, h_next)
                    koopman_losses.append(layer_loss)
                
                loss_koopman = torch.stack(koopman_losses).mean()
                
                # Check for numerical issues
                if torch.isnan(loss_koopman) or torch.isinf(loss_koopman):
                    print("Warning: Koopman loss is NaN/Inf, falling back to gradients only")
                    self.koopman_failed = True
                    loss_koopman = torch.tensor(0.0, device=self.device)
                    koopman_weight = 0.0
                elif loss_koopman.item() > self.fallback_threshold:
                    print(f"Warning: Koopman loss too high ({loss_koopman.item():.2f}), reducing weight")
                    koopman_weight = self.koopman_scheduler.get_weight() * 0.1
                else:
                    koopman_weight = self.koopman_scheduler.get_weight()
            
            except Exception as e:
                print(f"Warning: Koopman loss computation failed: {e}")
                self.koopman_failed = True
                loss_koopman = torch.tensor(0.0, device=self.device)
                koopman_weight = 0.0
        
        # Total loss
        total_loss = loss_lm + koopman_weight * loss_koopman
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        
        # Optimizer step
        self.optimizer.step()
        
        # Phase 3: Update Koopman operators (no gradients)
        if self.enable_koopman_updates and self.koopman_enabled and not self.koopman_failed:
            with torch.no_grad():
                for layer, (h_prev, h_next) in zip(self.model.blocks, hidden_states):
                    try:
                        layer.bk_layer.update_koopman_operator(
                            h_prev.detach(),
                            h_next.detach()
                        )
                    except Exception as e:
                        print(f"Warning: Koopman operator update failed: {e}")
                        # Continue training even if update fails
                        pass
        
        # Return metrics
        metrics = {
            'loss_lm': loss_lm.item(),
            'loss_koopman': loss_koopman.item() if isinstance(loss_koopman, torch.Tensor) else 0.0,
            'koopman_weight': koopman_weight,
            'total_loss': total_loss.item(),
            'koopman_enabled': self.koopman_enabled,
            'koopman_failed': self.koopman_failed,
        }
        
        return metrics
    
    def train_epoch(
        self,
        train_loader,
        epoch: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            epoch: current epoch number (if None, use internal counter)
        
        Returns:
            epoch_metrics: averaged metrics for the epoch
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        # Update Koopman scheduler
        self.koopman_scheduler.step(self.current_epoch)
        
        # Enable Koopman learning after warmup
        if self.current_epoch >= self.koopman_start_epoch:
            self.koopman_enabled = True
        
        # Accumulate metrics
        total_loss_lm = 0.0
        total_loss_koopman = 0.0
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            metrics = self.train_step(x_batch, y_batch)
            
            total_loss_lm += metrics['loss_lm']
            total_loss_koopman += metrics['loss_koopman']
            total_loss += metrics['total_loss']
            num_batches += 1
        
        # Average metrics
        epoch_metrics = {
            'epoch': self.current_epoch,
            'loss_lm': total_loss_lm / num_batches,
            'loss_koopman': total_loss_koopman / num_batches,
            'koopman_weight': self.koopman_scheduler.get_weight(),
            'total_loss': total_loss / num_batches,
            'koopman_enabled': self.koopman_enabled,
            'koopman_failed': self.koopman_failed,
        }
        
        return epoch_metrics
    
    def evaluate(
        self,
        val_loader,
        use_koopman=False
    ) -> Tuple[float, float]:
        """
        Evaluate model on validation set.
        
        Args:
            val_loader: DataLoader for validation data
            use_koopman: if True, use Koopman prediction; else use standard forward
        
        Returns:
            avg_loss: average validation loss
            perplexity: validation perplexity
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                logits = self.model(x_batch, use_koopman=use_koopman)
                loss = self.criterion(logits.view(-1, logits.size(-1)), y_batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return avg_loss, perplexity
    
    def state_dict(self):
        """Get trainer state for checkpointing."""
        return {
            'current_epoch': self.current_epoch,
            'koopman_enabled': self.koopman_enabled,
            'koopman_failed': self.koopman_failed,
            'koopman_scheduler': self.koopman_scheduler.state_dict(),
        }
    
    def load_state_dict(self, state_dict):
        """Load trainer state from checkpoint."""
        self.current_epoch = state_dict['current_epoch']
        self.koopman_enabled = state_dict['koopman_enabled']
        self.koopman_failed = state_dict['koopman_failed']
        self.koopman_scheduler.load_state_dict(state_dict['koopman_scheduler'])
