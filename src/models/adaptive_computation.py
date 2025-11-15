"""
Adaptive Computation Time (ACT) for ResNet-BK
Implements dynamic layer execution based on learned halting probabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_bk import MoEResNetBKLayer


class AdaptiveResNetBKBlock(nn.Module):
    """
    ResNet-BK block with adaptive computation time.
    Each token decides whether to continue processing based on learned halting probability.
    
    Architecture:
        Input -> LayerNorm -> MoEResNetBKLayer -> Halting Unit -> Weighted Output
    
    The halting unit predicts a probability p_halt for each token.
    Tokens accumulate halting probability until reaching threshold (default 0.99).
    """
    
    def __init__(self, d_model, n_seq, num_experts=4, top_k=1, dropout_p=0.1, threshold=0.99):
        super().__init__()
        self.d_model = d_model
        self.n_seq = n_seq
        self.threshold = threshold
        
        # Standard ResNet-BK components
        self.layer_norm = nn.LayerNorm(d_model)
        self.bk_layer = MoEResNetBKLayer(d_model, n_seq, num_experts, top_k, dropout_p)
        
        # Halting unit: predicts whether to stop processing this token
        # Uses smaller hidden dimension for efficiency
        self.halting_unit = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # Track ponder cost for this block (reset each forward pass)
        self.register_buffer('ponder_cost', torch.tensor(0.0))
    
    def forward(self, x, halting_prob_cumsum=None, still_running=None):
        """
        Forward pass with adaptive computation.
        
        Args:
            x: (B, N, D) - input tensor
            halting_prob_cumsum: (B, N) - cumulative halting probability from previous layers
            still_running: (B, N) - boolean mask of tokens still processing
        
        Returns:
            output: (B, N, D) - processed output
            halting_prob_cumsum: (B, N) - updated cumulative halting probability
            still_running: (B, N) - updated running mask
            weight: (B, N) - weight for this layer's contribution
        """
        B, N, D = x.shape
        
        # Initialize on first layer
        if halting_prob_cumsum is None:
            halting_prob_cumsum = torch.zeros(B, N, device=x.device)
            still_running = torch.ones(B, N, dtype=torch.bool, device=x.device)
        
        # Process through ResNet-BK layer
        x_normalized = self.layer_norm(x)
        x_processed = x + self.bk_layer(x_normalized)  # Residual connection
        
        # Compute halting probability for this step
        p_halt = self.halting_unit(x_processed).squeeze(-1)  # (B, N)
        
        # Only update for tokens still running
        p_halt_masked = p_halt * still_running.float()
        
        # Update cumulative halting probability
        halting_prob_cumsum_new = halting_prob_cumsum + p_halt_masked
        
        # Determine which tokens should halt (reached threshold)
        should_halt = halting_prob_cumsum_new >= self.threshold
        
        # Tokens that just halted this step
        just_halted = should_halt & still_running
        
        # Compute weight for this layer's contribution
        # For tokens that just halted: use remainder probability (1 - cumsum_before)
        # For tokens still running: use p_halt
        weight = torch.where(
            just_halted,
            1.0 - halting_prob_cumsum,  # Remainder to reach 1.0
            p_halt_masked
        )
        
        # Update running mask (tokens that halted are no longer running)
        still_running_new = still_running & (~should_halt)
        
        # Accumulate ponder cost (total computation used)
        # Ponder cost = sum of weights across all tokens
        self.ponder_cost = weight.sum()
        
        return x_processed, halting_prob_cumsum_new, still_running_new, weight
    
    def reset_ponder_cost(self):
        """Reset ponder cost counter (call at start of forward pass)."""
        self.ponder_cost = torch.tensor(0.0, device=self.ponder_cost.device)


class ACTLanguageModel(nn.Module):
    """
    Language model with Adaptive Computation Time.
    
    Dynamically determines how many layers to execute for each token.
    Adds ponder cost to loss to encourage early halting.
    """
    
    def __init__(
        self,
        vocab_size,
        d_model=64,
        n_layers=4,
        n_seq=128,
        num_experts=4,
        top_k=1,
        dropout_p=0.1,
        act_threshold=0.99,
        act_lambda=0.01
    ):
        super().__init__()
        self.d_model = d_model
        self.n_seq = n_seq
        self.n_layers = n_layers
        self.act_lambda = act_lambda  # Weight for ponder cost in loss
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(n_seq, d_model)
        
        # Adaptive ResNet-BK blocks
        self.blocks = nn.ModuleList([
            AdaptiveResNetBKBlock(
                d_model=d_model,
                n_seq=n_seq,
                num_experts=num_experts,
                top_k=top_k,
                dropout_p=dropout_p,
                threshold=act_threshold
            )
            for _ in range(n_layers)
        ])
        
        # Output layers
        self.layer_norm_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Track statistics
        self.register_buffer('avg_layers_executed', torch.tensor(0.0))
    
    def forward(self, x, return_ponder_cost=True):
        """
        Forward pass with adaptive computation.
        
        Args:
            x: (B, N) - token indices
            return_ponder_cost: if True, return ponder cost for loss computation
        
        Returns:
            logits: (B, N, vocab_size)
            ponder_cost: scalar (if return_ponder_cost=True)
        """
        B, N = x.shape
        assert N == self.n_seq, f"Sequence length mismatch: expected {self.n_seq}, got {N}"
        
        # Reset ponder costs
        for block in self.blocks:
            block.reset_ponder_cost()
        
        # Embeddings
        tok_emb = self.token_embedding(x)  # (B, N, D)
        pos = torch.arange(0, N, dtype=torch.long, device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)  # (1, N, D)
        h = tok_emb + pos_emb
        
        # Adaptive processing through layers
        halting_prob_cumsum = None
        still_running = None
        output_accumulator = torch.zeros_like(h)
        layers_executed = torch.zeros(B, N, device=x.device)
        
        for layer_idx, block in enumerate(self.blocks):
            # Process layer
            h, halting_prob_cumsum, still_running, weight = block(
                h, halting_prob_cumsum, still_running
            )
            
            # Accumulate weighted outputs
            output_accumulator += h * weight.unsqueeze(-1)
            
            # Track layers executed per token
            layers_executed += weight
            
            # Early exit if all tokens halted
            if not still_running.any():
                break
        
        # Final output
        h_final = self.layer_norm_final(output_accumulator)
        logits = self.lm_head(h_final)
        
        # Compute total ponder cost
        if return_ponder_cost:
            ponder_cost = sum(block.ponder_cost for block in self.blocks) / (B * N)
            
            # Update average layers executed (for monitoring)
            self.avg_layers_executed = layers_executed.mean()
            
            return logits, ponder_cost
        else:
            return logits
    
    def compute_loss(self, logits, targets, ponder_cost):
        """
        Compute total loss: cross-entropy + ponder cost penalty.
        
        Loss = CE_loss + λ * ponder_cost
        
        Args:
            logits: (B, N, vocab_size)
            targets: (B * N,) - flattened target indices
            ponder_cost: scalar - average ponder cost per token
        
        Returns:
            total_loss: scalar
            ce_loss: scalar (for monitoring)
            ponder_cost: scalar (for monitoring)
        """
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets)
        total_loss = ce_loss + self.act_lambda * ponder_cost
        
        return total_loss, ce_loss, ponder_cost
    
    def get_avg_layers_executed(self):
        """Get average number of layers executed per token."""
        return self.avg_layers_executed.item()


class ACTTrainer:
    """
    Trainer for ACT models with ponder cost monitoring.
    """
    
    def __init__(self, model, optimizer, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
        # Statistics
        self.total_ponder_cost = 0.0
        self.total_ce_loss = 0.0
        self.num_batches = 0
    
    def train_step(self, x_batch, y_batch):
        """
        Single training step.
        
        Args:
            x_batch: (B, N) - input token indices
            y_batch: (B * N,) - target token indices (flattened)
        
        Returns:
            metrics: dict with loss components and avg layers executed
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        logits, ponder_cost = self.model(x_batch, return_ponder_cost=True)
        
        # Compute loss
        total_loss, ce_loss, ponder_cost_val = self.model.compute_loss(
            logits, y_batch, ponder_cost
        )
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        
        # Update statistics
        self.total_ponder_cost += ponder_cost_val.item()
        self.total_ce_loss += ce_loss.item()
        self.num_batches += 1
        
        return {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'ponder_cost': ponder_cost_val.item(),
            'avg_layers_executed': self.model.get_avg_layers_executed()
        }
    
    def get_average_metrics(self):
        """Get average metrics across all training steps."""
        if self.num_batches == 0:
            return {'avg_ponder_cost': 0.0, 'avg_ce_loss': 0.0}
        
        return {
            'avg_ponder_cost': self.total_ponder_cost / self.num_batches,
            'avg_ce_loss': self.total_ce_loss / self.num_batches
        }
    
    def reset_statistics(self):
        """Reset training statistics."""
        self.total_ponder_cost = 0.0
        self.total_ce_loss = 0.0
        self.num_batches = 0
