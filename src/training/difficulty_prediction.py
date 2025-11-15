"""
Example Difficulty Prediction

Implements lightweight model to predict training loss before forward pass,
enabling skipping of easy examples during training.

Based on Step 7 design for achieving 10× cost reduction through data efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Tuple
import numpy as np


class DifficultyPredictor(nn.Module):
    """
    Lightweight model to predict training loss for an example.
    
    Implements Requirements 7.13, 7.14:
    - Train lightweight model to predict training loss
    - Skip easy examples during training
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2
    ):
        """
        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        """
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.predictor = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict difficulty (training loss) for examples.
        
        Args:
            x: (batch_size, input_dim) example embeddings
        
        Returns:
            predicted_loss: (batch_size,) predicted training losses
        """
        return self.predictor(x).squeeze(-1)


class DifficultyPredictionTrainer:
    """
    Train difficulty predictor and use it to skip easy examples.
    """
    
    def __init__(
        self,
        main_model: nn.Module,
        difficulty_predictor: DifficultyPredictor,
        skip_threshold: float = 0.5,
        embedding_method: str = 'mean_token',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            main_model: Main model being trained
            difficulty_predictor: Difficulty prediction model
            skip_threshold: Skip examples with predicted loss < threshold
            embedding_method: Method for computing example embeddings
            device: Device for computation
        """
        self.main_model = main_model.to(device)
        self.difficulty_predictor = difficulty_predictor.to(device)
        self.skip_threshold = skip_threshold
        self.embedding_method = embedding_method
        self.device = device
        
        # Statistics
        self.total_examples = 0
        self.skipped_examples = 0
    
    def compute_example_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute embedding for example x.
        
        Args:
            x: (batch_size, seq_len) tensor
        
        Returns:
            embedding: (batch_size, embedding_dim) tensor
        """
        with torch.no_grad():
            if self.embedding_method == 'mean_token':
                # Mean of token embeddings
                if hasattr(self.main_model, 'token_embedding'):
                    emb = self.main_model.token_embedding(x)  # (batch_size, seq_len, d_model)
                    embedding = emb.mean(dim=1)  # (batch_size, d_model)
                else:
                    # Fallback: use input token IDs directly
                    embedding = x.float().mean(dim=1)
            
            elif self.embedding_method == 'cls_token':
                # First token embedding
                if hasattr(self.main_model, 'token_embedding'):
                    emb = self.main_model.token_embedding(x)
                    embedding = emb[:, 0, :]  # (batch_size, d_model)
                else:
                    embedding = x[:, 0].float().unsqueeze(-1)
            
            else:
                raise ValueError(f"Unknown embedding method: {self.embedding_method}")
        
        return embedding
    
    def train_difficulty_predictor(
        self,
        train_dataset,
        optimizer,
        criterion,
        num_epochs: int = 3,
        batch_size: int = 32
    ) -> Dict:
        """
        Train difficulty predictor on training data.
        
        Collect (example_embedding, actual_loss) pairs and train predictor.
        
        Args:
            train_dataset: Training dataset
            optimizer: Optimizer for difficulty predictor
            criterion: Loss criterion for main model
            num_epochs: Number of training epochs
            batch_size: Batch size
        
        Returns:
            metrics: Training metrics
        """
        print("=" * 60)
        print("TRAINING DIFFICULTY PREDICTOR")
        print("=" * 60)
        
        # Step 1: Collect training data (embeddings, actual losses)
        print("\n[1/2] Collecting training data...")
        
        dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )
        
        embeddings_list = []
        losses_list = []
        
        self.main_model.eval()
        
        with torch.no_grad():
            for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Compute embedding
                emb = self.compute_example_embedding(x_batch)
                
                # Compute actual loss
                if hasattr(self.main_model, 'forward') and 'ponder_cost' in str(self.main_model.forward.__code__.co_varnames):
                    logits, _ = self.main_model(x_batch)
                else:
                    logits = self.main_model(x_batch)
                
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y_batch.view(-1),
                    reduction='none'
                )
                loss_per_example = loss.view(x_batch.size(0), -1).mean(dim=1)
                
                embeddings_list.append(emb.cpu())
                losses_list.append(loss_per_example.cpu())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {(batch_idx + 1) * batch_size}/{len(train_dataset)} examples")
        
        embeddings = torch.cat(embeddings_list, dim=0)
        losses = torch.cat(losses_list, dim=0)
        
        print(f"Collected {len(embeddings)} training examples")
        print(f"  Loss range: {losses.min().item():.4f} - {losses.max().item():.4f}")
        
        # Step 2: Train difficulty predictor
        print("\n[2/2] Training predictor...")
        
        # Create dataset
        predictor_dataset = torch.utils.data.TensorDataset(embeddings, losses)
        predictor_loader = DataLoader(
            predictor_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )
        
        self.difficulty_predictor.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_batches = 0
            
            for emb_batch, loss_batch in predictor_loader:
                emb_batch = emb_batch.to(self.device)
                loss_batch = loss_batch.to(self.device)
                
                optimizer.zero_grad()
                
                # Predict loss
                pred_loss = self.difficulty_predictor(emb_batch)
                
                # MSE loss
                loss = F.mse_loss(pred_loss, loss_batch)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_batches += 1
            
            avg_loss = epoch_loss / epoch_batches
            print(f"Epoch {epoch+1}/{num_epochs}: MSE Loss = {avg_loss:.6f}")
        
        # Evaluate predictor
        self.difficulty_predictor.eval()
        
        with torch.no_grad():
            pred_losses = self.difficulty_predictor(embeddings.to(self.device))
            mse = F.mse_loss(pred_losses, losses.to(self.device))
            mae = F.l1_loss(pred_losses, losses.to(self.device))
            
            # Correlation
            pred_losses_np = pred_losses.cpu().numpy()
            losses_np = losses.numpy()
            correlation = np.corrcoef(pred_losses_np, losses_np)[0, 1]
        
        print(f"\nPredictor evaluation:")
        print(f"  MSE: {mse.item():.6f}")
        print(f"  MAE: {mae.item():.6f}")
        print(f"  Correlation: {correlation:.4f}")
        
        return {
            'mse': mse.item(),
            'mae': mae.item(),
            'correlation': correlation
        }
    
    def train_step_with_skipping(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        optimizer,
        criterion
    ) -> Tuple[float, int, int]:
        """
        Training step with example skipping based on predicted difficulty.
        
        Args:
            x_batch: (batch_size, seq_len) input tensor
            y_batch: (batch_size, seq_len) target tensor
            optimizer: Optimizer for main model
            criterion: Loss criterion
        
        Returns:
            loss: Average loss for processed examples
            num_processed: Number of examples processed
            num_skipped: Number of examples skipped
        """
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        
        batch_size = x_batch.size(0)
        
        # Compute embeddings
        emb = self.compute_example_embedding(x_batch)
        
        # Predict difficulty
        self.difficulty_predictor.eval()
        with torch.no_grad():
            pred_loss = self.difficulty_predictor(emb)
        
        # Determine which examples to process
        process_mask = pred_loss >= self.skip_threshold
        num_processed = process_mask.sum().item()
        num_skipped = batch_size - num_processed
        
        self.total_examples += batch_size
        self.skipped_examples += num_skipped
        
        if num_processed == 0:
            # Skip entire batch
            return 0.0, 0, num_skipped
        
        # Filter examples
        x_filtered = x_batch[process_mask]
        y_filtered = y_batch[process_mask]
        
        # Standard training step on filtered examples
        optimizer.zero_grad()
        
        self.main_model.train()
        
        if hasattr(self.main_model, 'forward') and 'ponder_cost' in str(self.main_model.forward.__code__.co_varnames):
            logits, ponder_cost = self.main_model(x_filtered)
            loss = criterion(logits.view(-1, logits.size(-1)), y_filtered.view(-1))
            loss = loss + 0.01 * ponder_cost
        else:
            logits = self.main_model(x_filtered)
            loss = criterion(logits.view(-1, logits.size(-1)), y_filtered.view(-1))
        
        loss.backward()
        optimizer.step()
        
        return loss.item(), num_processed, num_skipped
    
    def get_skip_statistics(self) -> Dict:
        """
        Get example skipping statistics.
        
        Returns:
            stats: Dictionary with skipping statistics
        """
        skip_rate = self.skipped_examples / self.total_examples if self.total_examples > 0 else 0
        
        return {
            'total_examples': self.total_examples,
            'skipped_examples': self.skipped_examples,
            'processed_examples': self.total_examples - self.skipped_examples,
            'skip_rate': skip_rate
        }
    
    def reset_statistics(self):
        """Reset skipping statistics."""
        self.total_examples = 0
        self.skipped_examples = 0


def train_with_difficulty_prediction(
    model: nn.Module,
    train_dataset,
    optimizer,
    criterion,
    num_epochs: int = 5,
    batch_size: int = 32,
    skip_threshold: float = 0.5,
    predictor_hidden_dim: int = 64,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """
    Train model with difficulty-based example skipping.
    
    Args:
        model: Model to train
        train_dataset: Training dataset
        optimizer: Optimizer
        criterion: Loss criterion
        num_epochs: Number of epochs
        batch_size: Batch size
        skip_threshold: Skip examples with predicted loss < threshold
        predictor_hidden_dim: Hidden dimension for difficulty predictor
        device: Device for computation
    
    Returns:
        metrics: Training metrics including skip statistics
    """
    # Determine input dimension
    if hasattr(model, 'token_embedding'):
        input_dim = model.token_embedding.embedding_dim
    else:
        input_dim = 128  # Default
    
    # Create difficulty predictor
    difficulty_predictor = DifficultyPredictor(
        input_dim=input_dim,
        hidden_dim=predictor_hidden_dim,
        num_layers=2
    )
    
    # Create trainer
    trainer = DifficultyPredictionTrainer(
        main_model=model,
        difficulty_predictor=difficulty_predictor,
        skip_threshold=skip_threshold,
        device=device
    )
    
    # Train difficulty predictor
    predictor_optimizer = torch.optim.Adam(difficulty_predictor.parameters(), lr=1e-3)
    trainer.train_difficulty_predictor(
        train_dataset,
        predictor_optimizer,
        criterion,
        num_epochs=3,
        batch_size=batch_size
    )
    
    # Train main model with skipping
    print("\n" + "=" * 60)
    print("TRAINING WITH EXAMPLE SKIPPING")
    print("=" * 60)
    print(f"Skip threshold: {skip_threshold}")
    print()
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_processed = 0
        epoch_skipped = 0
        
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            loss, num_processed, num_skipped = trainer.train_step_with_skipping(
                x_batch, y_batch, optimizer, criterion
            )
            
            if num_processed > 0:
                epoch_loss += loss * num_processed
            epoch_processed += num_processed
            epoch_skipped += num_skipped
            
            if (batch_idx + 1) % 50 == 0:
                avg_loss = epoch_loss / epoch_processed if epoch_processed > 0 else 0
                skip_rate = epoch_skipped / (epoch_processed + epoch_skipped)
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                      f"Loss: {avg_loss:.4f}, Skip Rate: {skip_rate:.2%}")
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / epoch_processed if epoch_processed > 0 else 0
        epoch_skip_rate = epoch_skipped / (epoch_processed + epoch_skipped)
        
        print(f"\nEpoch {epoch+1} completed:")
        print(f"  Avg Loss: {avg_epoch_loss:.4f}")
        print(f"  Processed: {epoch_processed}")
        print(f"  Skipped: {epoch_skipped}")
        print(f"  Skip Rate: {epoch_skip_rate:.2%}")
        print("-" * 60)
    
    # Final statistics
    final_stats = trainer.get_skip_statistics()
    
    print("\n" + "=" * 60)
    print("EXAMPLE SKIPPING STATISTICS")
    print("=" * 60)
    print(f"Total examples: {final_stats['total_examples']}")
    print(f"Processed: {final_stats['processed_examples']}")
    print(f"Skipped: {final_stats['skipped_examples']}")
    print(f"Skip rate: {final_stats['skip_rate']:.2%}")
    print(f"Speedup: {1 / (1 - final_stats['skip_rate']):.2f}×")
    
    return final_stats
