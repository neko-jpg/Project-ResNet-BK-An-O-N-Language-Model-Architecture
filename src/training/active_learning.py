"""
Active Learning Selector

Implements active learning by selecting most informative examples for training
based on model uncertainty.

Based on Step 7 design for achieving 10× cost reduction through data efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import List, Literal, Optional, Tuple


class ActiveLearningSelector:
    """
    Select most informative examples for training.
    
    Implements Requirements 7.7, 7.8:
    - Compute uncertainty (entropy) for each example
    - Select top-k most uncertain examples
    """
    
    def __init__(
        self,
        model: nn.Module,
        selection_strategy: Literal['uncertainty', 'margin', 'entropy', 'variation_ratio'] = 'uncertainty',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model: Model for computing uncertainty
            selection_strategy: Strategy for selecting examples
                - 'uncertainty': Entropy of output distribution
                - 'margin': Difference between top-2 predictions
                - 'entropy': Same as uncertainty
                - 'variation_ratio': 1 - (most common prediction frequency)
            device: Device for computation
        """
        self.model = model.to(device)
        self.device = device
        self.selection_strategy = selection_strategy
    
    def compute_uncertainty(self, x: torch.Tensor) -> float:
        """
        Compute model uncertainty for example x.
        
        Uncertainty = entropy of output distribution
        
        Args:
            x: (seq_len,) or (1, seq_len) tensor
        
        Returns:
            uncertainty: Scalar uncertainty score
        """
        self.model.eval()
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = x.to(self.device)
        
        with torch.no_grad():
            # Forward pass
            if hasattr(self.model, 'forward') and 'ponder_cost' in str(self.model.forward.__code__.co_varnames):
                # ACT model returns (logits, ponder_cost)
                logits, _ = self.model(x)
            else:
                logits = self.model(x)
            
            # Compute uncertainty based on strategy
            if self.selection_strategy in ['uncertainty', 'entropy']:
                # Entropy: H(p) = -Σ p(x) log p(x)
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                uncertainty = entropy.mean().item()
            
            elif self.selection_strategy == 'margin':
                # Margin: difference between top-2 predictions
                probs = F.softmax(logits, dim=-1)
                top2_probs, _ = torch.topk(probs, k=2, dim=-1)
                margin = top2_probs[..., 0] - top2_probs[..., 1]
                # Lower margin = higher uncertainty
                uncertainty = (1.0 - margin).mean().item()
            
            elif self.selection_strategy == 'variation_ratio':
                # Variation ratio: 1 - (frequency of most common prediction)
                probs = F.softmax(logits, dim=-1)
                max_probs, _ = torch.max(probs, dim=-1)
                variation_ratio = 1.0 - max_probs
                uncertainty = variation_ratio.mean().item()
            
            else:
                raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")
        
        return uncertainty
    
    def compute_uncertainties_batch(
        self,
        dataset,
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Compute uncertainties for all examples in dataset.
        
        Args:
            dataset: Dataset with (input, target) pairs
            batch_size: Batch size for processing
        
        Returns:
            uncertainties: (num_examples,) tensor of uncertainty scores
        """
        self.model.eval()
        uncertainties = []
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )
        
        print(f"Computing uncertainties for {len(dataset)} examples...")
        
        with torch.no_grad():
            for batch_idx, (x_batch, _) in enumerate(dataloader):
                x_batch = x_batch.to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'forward') and 'ponder_cost' in str(self.model.forward.__code__.co_varnames):
                    logits, _ = self.model(x_batch)
                else:
                    logits = self.model(x_batch)
                
                # Compute uncertainty for each example in batch
                if self.selection_strategy in ['uncertainty', 'entropy']:
                    probs = F.softmax(logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                    batch_uncertainties = entropy.mean(dim=1)  # Average over sequence
                
                elif self.selection_strategy == 'margin':
                    probs = F.softmax(logits, dim=-1)
                    top2_probs, _ = torch.topk(probs, k=2, dim=-1)
                    margin = top2_probs[..., 0] - top2_probs[..., 1]
                    batch_uncertainties = (1.0 - margin).mean(dim=1)
                
                elif self.selection_strategy == 'variation_ratio':
                    probs = F.softmax(logits, dim=-1)
                    max_probs, _ = torch.max(probs, dim=-1)
                    variation_ratio = 1.0 - max_probs
                    batch_uncertainties = variation_ratio.mean(dim=1)
                
                uncertainties.append(batch_uncertainties.cpu())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {(batch_idx + 1) * batch_size}/{len(dataset)} examples")
        
        uncertainties = torch.cat(uncertainties)
        
        print(f"Uncertainties computed:")
        print(f"  Min: {uncertainties.min().item():.4f}")
        print(f"  Max: {uncertainties.max().item():.4f}")
        print(f"  Mean: {uncertainties.mean().item():.4f}")
        print(f"  Median: {uncertainties.median().item():.4f}")
        
        return uncertainties
    
    def select_examples(
        self,
        unlabeled_pool,
        num_select: int,
        batch_size: int = 32
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Select num_select most informative examples from pool.
        
        Args:
            unlabeled_pool: Dataset of unlabeled examples
            num_select: Number of examples to select
            batch_size: Batch size for processing
        
        Returns:
            selected_indices: List of selected example indices
            uncertainties: Uncertainty scores for all examples
        """
        # Compute uncertainties for all examples
        uncertainties = self.compute_uncertainties_batch(unlabeled_pool, batch_size)
        
        # Select top-k most uncertain
        num_select = min(num_select, len(uncertainties))
        _, indices = torch.topk(uncertainties, num_select)
        
        selected_indices = indices.tolist()
        
        print(f"Selected {len(selected_indices)} most uncertain examples")
        print(f"  Uncertainty range: {uncertainties[selected_indices[0]].item():.4f} - {uncertainties[selected_indices[-1]].item():.4f}")
        
        return selected_indices, uncertainties
    
    def create_active_dataloader(
        self,
        dataset,
        num_select: int,
        batch_size: int = 32,
        shuffle: bool = True
    ) -> DataLoader:
        """
        Create dataloader with actively selected examples.
        
        Args:
            dataset: Full dataset
            num_select: Number of examples to select
            batch_size: Batch size for dataloader
            shuffle: Whether to shuffle selected examples
        
        Returns:
            dataloader: DataLoader with selected examples
        """
        # Select examples
        selected_indices, _ = self.select_examples(dataset, num_select, batch_size)
        
        # Create subset
        subset = Subset(dataset, selected_indices)
        
        # Create dataloader
        dataloader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False
        )
        
        return dataloader


class ActiveLearningLoop:
    """
    Complete active learning training loop.
    
    Iteratively:
    1. Train on labeled data
    2. Select most uncertain examples from unlabeled pool
    3. Add to labeled set (simulated labeling)
    4. Repeat
    """
    
    def __init__(
        self,
        model: nn.Module,
        labeled_dataset,
        unlabeled_dataset,
        selection_strategy: str = 'uncertainty',
        num_select_per_round: int = 100,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model: Model to train
            labeled_dataset: Initial labeled dataset
            unlabeled_dataset: Pool of unlabeled examples
            selection_strategy: Active learning selection strategy
            num_select_per_round: Number of examples to select each round
            device: Device for computation
        """
        self.model = model
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.selection_strategy = selection_strategy
        self.num_select_per_round = num_select_per_round
        self.device = device
        
        self.selector = ActiveLearningSelector(
            model,
            selection_strategy=selection_strategy,
            device=device
        )
        
        # Track selected indices
        self.labeled_indices = list(range(len(labeled_dataset)))
        self.unlabeled_indices = list(range(len(unlabeled_dataset)))
    
    def run_round(
        self,
        optimizer,
        criterion,
        num_epochs: int = 1,
        batch_size: int = 32
    ) -> dict:
        """
        Run one round of active learning.
        
        1. Train on current labeled data
        2. Select new examples from unlabeled pool
        3. Add to labeled set
        
        Args:
            optimizer: Optimizer for training
            criterion: Loss criterion
            num_epochs: Number of training epochs per round
            batch_size: Batch size
        
        Returns:
            metrics: Dictionary with round metrics
        """
        print(f"\n=== Active Learning Round ===")
        print(f"Labeled examples: {len(self.labeled_indices)}")
        print(f"Unlabeled examples: {len(self.unlabeled_indices)}")
        
        # Step 1: Train on labeled data
        print("\n[1/3] Training on labeled data...")
        train_loader = DataLoader(
            self.labeled_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for epoch in range(num_epochs):
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                if hasattr(self.model, 'forward') and 'ponder_cost' in str(self.model.forward.__code__.co_varnames):
                    logits, ponder_cost = self.model(x_batch)
                    loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
                    loss = loss + 0.01 * ponder_cost  # Add ponder cost if ACT
                else:
                    logits = self.model(x_batch)
                    loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"  Average training loss: {avg_loss:.4f}")
        
        # Step 2: Select new examples
        if len(self.unlabeled_indices) == 0:
            print("\n[2/3] No unlabeled examples remaining")
            return {'avg_loss': avg_loss, 'num_selected': 0}
        
        print(f"\n[2/3] Selecting {self.num_select_per_round} new examples...")
        
        # Create subset of unlabeled data
        unlabeled_subset = Subset(self.unlabeled_dataset, self.unlabeled_indices)
        
        # Select examples
        selected_relative_indices, uncertainties = self.selector.select_examples(
            unlabeled_subset,
            self.num_select_per_round,
            batch_size=batch_size
        )
        
        # Convert relative indices to absolute indices
        selected_absolute_indices = [self.unlabeled_indices[i] for i in selected_relative_indices]
        
        # Step 3: Move selected examples to labeled set
        print(f"\n[3/3] Adding {len(selected_absolute_indices)} examples to labeled set...")
        
        # Add to labeled indices
        self.labeled_indices.extend(selected_absolute_indices)
        
        # Remove from unlabeled indices
        for idx in selected_absolute_indices:
            self.unlabeled_indices.remove(idx)
        
        # Update labeled dataset
        self.labeled_dataset = Subset(
            self.unlabeled_dataset.dataset if hasattr(self.unlabeled_dataset, 'dataset') else self.unlabeled_dataset,
            self.labeled_indices
        )
        
        print(f"Updated dataset sizes:")
        print(f"  Labeled: {len(self.labeled_indices)}")
        print(f"  Unlabeled: {len(self.unlabeled_indices)}")
        
        return {
            'avg_loss': avg_loss,
            'num_selected': len(selected_absolute_indices),
            'num_labeled': len(self.labeled_indices),
            'num_unlabeled': len(self.unlabeled_indices),
            'mean_uncertainty': uncertainties.mean().item(),
            'selected_uncertainty_mean': uncertainties[selected_relative_indices].mean().item()
        }


def create_active_learning_trainer(
    model: nn.Module,
    full_dataset,
    initial_labeled_ratio: float = 0.1,
    selection_strategy: str = 'uncertainty',
    num_select_per_round: int = 100,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> ActiveLearningLoop:
    """
    Create active learning trainer.
    
    Args:
        model: Model to train
        full_dataset: Full dataset (will be split into labeled/unlabeled)
        initial_labeled_ratio: Ratio of initial labeled data
        selection_strategy: Active learning strategy
        num_select_per_round: Examples to select per round
        device: Device for computation
    
    Returns:
        al_loop: ActiveLearningLoop instance
    """
    # Split dataset into initial labeled and unlabeled
    num_total = len(full_dataset)
    num_labeled = int(num_total * initial_labeled_ratio)
    
    # Random split
    indices = torch.randperm(num_total).tolist()
    labeled_indices = indices[:num_labeled]
    unlabeled_indices = indices[num_labeled:]
    
    labeled_dataset = Subset(full_dataset, labeled_indices)
    unlabeled_dataset = Subset(full_dataset, unlabeled_indices)
    
    print(f"Active learning setup:")
    print(f"  Total examples: {num_total}")
    print(f"  Initial labeled: {num_labeled} ({initial_labeled_ratio*100:.1f}%)")
    print(f"  Initial unlabeled: {len(unlabeled_indices)}")
    
    # Create active learning loop
    al_loop = ActiveLearningLoop(
        model=model,
        labeled_dataset=labeled_dataset,
        unlabeled_dataset=full_dataset,  # Keep reference to full dataset
        selection_strategy=selection_strategy,
        num_select_per_round=num_select_per_round,
        device=device
    )
    
    # Override indices
    al_loop.labeled_indices = labeled_indices
    al_loop.unlabeled_indices = unlabeled_indices
    
    return al_loop
