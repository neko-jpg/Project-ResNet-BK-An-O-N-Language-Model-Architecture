"""
Physics-Informed Trainer
Training loop with energy conservation constraints and monitoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import numpy as np


class PhysicsInformedTrainer:
    """
    Trainer for physics-informed learning with energy conservation.
    
    Features:
    - Energy conservation constraint: L_energy = ||E(x_t) - E(x_{t-1})||^2
    - Automatic Lagrange multiplier adjustment
    - Energy drift monitoring
    - Hamiltonian structure preservation
    
    Args:
        model: ResNet-BK model with PhysicsInformedBKLayer
        optimizer: PyTorch optimizer
        criterion: loss function (e.g., CrossEntropyLoss)
        lambda_energy_init: initial Lagrange multiplier for energy conservation
        lambda_energy_lr: learning rate for Lagrange multiplier adaptation
        energy_target_drift: target energy drift (for automatic lambda adjustment)
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        lambda_energy_init: float = 0.1,
        lambda_energy_lr: float = 0.01,
        energy_target_drift: float = 0.1
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lambda_energy_lr = lambda_energy_lr
        self.energy_target_drift = energy_target_drift
        
        # Initialize Lagrange multipliers for all physics-informed layers
        self._initialize_lagrange_multipliers(lambda_energy_init)
        
        # Energy drift tracking
        self.energy_history = []
        self.energy_drift_history = []
        
        # Loss tracking
        self.loss_history = {
            'total': [],
            'lm': [],
            'energy_conservation': [],
            'energy_drift': []
        }
    
    def _initialize_lagrange_multipliers(self, lambda_init: float):
        """Initialize Lagrange multipliers for all physics-informed layers."""
        for module in self.model.modules():
            if hasattr(module, 'lambda_energy'):
                module.lambda_energy.data.fill_(lambda_init)
    
    def train_step(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        x_prev_batch: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Single training step with energy conservation.
        
        Args:
            x_batch: (B, N) input token indices
            y_batch: (B*N,) target token indices (flattened)
            x_prev_batch: (B, N) previous batch tokens (for energy conservation)
        
        Returns:
            metrics: dict with loss components and energy metrics
        """
        self.optimizer.zero_grad()
        
        # Forward pass
        logits = self.model(x_batch)  # (B, N, vocab_size)
        
        # Language modeling loss
        loss_lm = self.criterion(logits.view(-1, logits.size(-1)), y_batch)
        
        # Energy conservation loss (if previous batch available)
        loss_energy = torch.tensor(0.0, device=x_batch.device)
        energy_metrics = {}
        
        if x_prev_batch is not None:
            # Get embeddings for current and previous batches
            x_embed = self.model.token_embedding(x_batch)  # (B, N, D)
            x_prev_embed = self.model.token_embedding(x_prev_batch)  # (B, N, D)
            
            # Compute energy conservation loss for each physics-informed layer
            total_energy_loss = 0.0
            layer_count = 0
            
            for block in self.model.blocks:
                if hasattr(block, 'bk_layer') and hasattr(block.bk_layer, 'compute_energy'):
                    # Compute energies
                    E_current, T_current, V_current = block.bk_layer.compute_energy(
                        x_embed, x_prev_embed
                    )
                    E_prev, _, _ = block.bk_layer.compute_energy(x_prev_embed, None)
                    
                    # Energy conservation loss
                    layer_energy_loss = block.bk_layer.energy_conservation_loss(
                        E_current, E_prev
                    )
                    
                    # Weighted by Lagrange multiplier
                    total_energy_loss += block.bk_layer.lambda_energy * layer_energy_loss
                    layer_count += 1
                    
                    # Track metrics
                    energy_metrics[f'E_current_layer{layer_count}'] = E_current.mean().item()
                    energy_metrics[f'E_prev_layer{layer_count}'] = E_prev.mean().item()
                    energy_metrics[f'T_current_layer{layer_count}'] = T_current.mean().item()
                    energy_metrics[f'V_current_layer{layer_count}'] = V_current.mean().item()
                    energy_metrics[f'energy_drift_layer{layer_count}'] = (
                        (E_current - E_prev).abs().mean().item()
                    )
            
            if layer_count > 0:
                loss_energy = total_energy_loss / layer_count
        
        # Total loss
        loss_total = loss_lm + loss_energy
        
        # Backward pass
        loss_total.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        
        # Optimizer step
        self.optimizer.step()
        
        # Update Lagrange multipliers based on energy drift
        if x_prev_batch is not None:
            self._update_lagrange_multipliers(energy_metrics)
        
        # Compile metrics
        metrics = {
            'loss_total': loss_total.item(),
            'loss_lm': loss_lm.item(),
            'loss_energy': loss_energy.item() if isinstance(loss_energy, torch.Tensor) else 0.0,
            **energy_metrics
        }
        
        # Track history
        self.loss_history['total'].append(metrics['loss_total'])
        self.loss_history['lm'].append(metrics['loss_lm'])
        self.loss_history['energy_conservation'].append(metrics['loss_energy'])
        
        return metrics
    
    def _update_lagrange_multipliers(self, energy_metrics: Dict[str, float]):
        """
        Automatically adjust Lagrange multipliers based on energy drift.
        
        If energy drift > target: increase lambda (enforce conservation more)
        If energy drift < target: decrease lambda (relax constraint)
        """
        # Compute average energy drift across layers
        drift_keys = [k for k in energy_metrics.keys() if 'energy_drift' in k]
        if not drift_keys:
            return
        
        avg_drift = np.mean([energy_metrics[k] for k in drift_keys])
        self.energy_drift_history.append(avg_drift)
        
        # Adjust Lagrange multipliers
        drift_error = avg_drift - self.energy_target_drift
        
        for block in self.model.blocks:
            if hasattr(block, 'bk_layer') and hasattr(block.bk_layer, 'lambda_energy'):
                # Gradient ascent on lambda (increase if drift too high)
                with torch.no_grad():
                    lambda_update = self.lambda_energy_lr * drift_error
                    block.bk_layer.lambda_energy.data += lambda_update
                    
                    # Clamp to reasonable range
                    block.bk_layer.lambda_energy.data.clamp_(0.01, 10.0)
    
    def monitor_energy_drift(self) -> Dict[str, float]:
        """
        Monitor energy drift statistics.
        
        Returns:
            stats: dict with energy drift statistics
        """
        if not self.energy_drift_history:
            return {}
        
        recent_drift = self.energy_drift_history[-100:]  # Last 100 steps
        
        stats = {
            'energy_drift_mean': np.mean(recent_drift),
            'energy_drift_std': np.std(recent_drift),
            'energy_drift_max': np.max(recent_drift),
            'energy_drift_min': np.min(recent_drift),
        }
        
        return stats
    
    def get_lagrange_multipliers(self) -> List[float]:
        """Get current Lagrange multipliers for all layers."""
        lambdas = []
        for block in self.model.blocks:
            if hasattr(block, 'bk_layer') and hasattr(block.bk_layer, 'lambda_energy'):
                lambdas.append(block.bk_layer.lambda_energy.item())
        return lambdas
    
    def train_epoch(
        self,
        train_loader,
        epoch: int,
        log_interval: int = 100
    ) -> Dict[str, float]:
        """
        Train for one epoch with energy conservation monitoring.
        
        Args:
            train_loader: DataLoader for training data
            epoch: current epoch number
            log_interval: steps between logging
        
        Returns:
            epoch_metrics: dict with average metrics for the epoch
        """
        self.model.train()
        
        epoch_metrics = {
            'loss_total': 0.0,
            'loss_lm': 0.0,
            'loss_energy': 0.0,
            'energy_drift': 0.0
        }
        
        x_prev_batch = None
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            # Move to device
            device = next(self.model.parameters()).device
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Training step
            metrics = self.train_step(x_batch, y_batch, x_prev_batch)
            
            # Accumulate metrics
            for key in epoch_metrics.keys():
                if key in metrics:
                    epoch_metrics[key] += metrics[key]
                elif 'energy_drift' in key and 'energy_drift_layer1' in metrics:
                    # Average drift across layers
                    drift_keys = [k for k in metrics.keys() if 'energy_drift' in k]
                    epoch_metrics['energy_drift'] += np.mean([metrics[k] for k in drift_keys])
            
            # Save current batch as previous for next iteration
            x_prev_batch = x_batch.detach()
            
            # Logging
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = epoch_metrics['loss_total'] / (batch_idx + 1)
                avg_energy_loss = epoch_metrics['loss_energy'] / (batch_idx + 1)
                drift_stats = self.monitor_energy_drift()
                lambdas = self.get_lagrange_multipliers()
                
                print(f"Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.4f} | "
                      f"Energy Loss: {avg_energy_loss:.4f} | "
                      f"Drift: {drift_stats.get('energy_drift_mean', 0):.4f} | "
                      f"Lambda: {lambdas[0]:.4f}")
        
        # Average metrics over epoch
        num_batches = len(train_loader)
        for key in epoch_metrics.keys():
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def evaluate(
        self,
        val_loader,
        compute_energy_metrics: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model on validation set.
        
        Args:
            val_loader: DataLoader for validation data
            compute_energy_metrics: if True, compute energy conservation metrics
        
        Returns:
            val_metrics: dict with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_energy_drift = 0.0
        num_batches = 0
        
        x_prev_batch = None
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                # Move to device
                device = next(self.model.parameters()).device
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                # Forward pass
                logits = self.model(x_batch)
                loss = self.criterion(logits.view(-1, logits.size(-1)), y_batch)
                total_loss += loss.item()
                
                # Energy metrics
                if compute_energy_metrics and x_prev_batch is not None:
                    x_embed = self.model.token_embedding(x_batch)
                    x_prev_embed = self.model.token_embedding(x_prev_batch)
                    
                    for block in self.model.blocks:
                        if hasattr(block, 'bk_layer') and hasattr(block.bk_layer, 'compute_energy'):
                            E_current, _, _ = block.bk_layer.compute_energy(x_embed, x_prev_embed)
                            E_prev, _, _ = block.bk_layer.compute_energy(x_prev_embed, None)
                            drift = (E_current - E_prev).abs().mean().item()
                            total_energy_drift += drift
                            break  # Only first layer for efficiency
                
                x_prev_batch = x_batch
                num_batches += 1
        
        val_metrics = {
            'val_loss': total_loss / num_batches,
            'val_perplexity': torch.exp(torch.tensor(total_loss / num_batches)).item(),
        }
        
        if compute_energy_metrics:
            val_metrics['val_energy_drift'] = total_energy_drift / num_batches
        
        return val_metrics
