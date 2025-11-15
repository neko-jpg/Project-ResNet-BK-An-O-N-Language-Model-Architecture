"""
Learned Sequence Length for ResNet-BK
Dynamically determines optimal sequence length N for each input.
Pads or truncates sequences based on predicted optimal length.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_bk import MoEResNetBKLayer


class SequenceLengthPredictor(nn.Module):
    """
    Predicts optimal sequence length for each input.
    
    Uses a lightweight encoder to analyze input and predict the minimum
    sequence length needed for accurate processing.
    
    Architecture:
        Input Embedding -> Pooling -> MLP -> Sequence Length Logits
    """
    
    def __init__(self, d_model, max_seq_len=128, num_length_bins=8):
        """
        Args:
            d_model: model dimension
            max_seq_len: maximum sequence length
            num_length_bins: number of discrete length options
                            (e.g., 8 bins: [16, 32, 48, 64, 80, 96, 112, 128])
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_length_bins = num_length_bins
        
        # Define length bins (evenly spaced)
        self.length_bins = torch.linspace(
            max_seq_len // num_length_bins,
            max_seq_len,
            num_length_bins,
            dtype=torch.long
        )
        
        # Lightweight predictor network
        # Uses global average pooling to get sequence-level representation
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_length_bins)
        )
    
    def forward(self, x_embedded, return_distribution=False):
        """
        Predict optimal sequence length for input.
        
        Args:
            x_embedded: (B, N, D) - embedded input tokens
            return_distribution: if True, return probability distribution over lengths
        
        Returns:
            predicted_lengths: (B,) - predicted optimal length for each sequence
            length_probs: (B, num_length_bins) - probability distribution (if return_distribution=True)
        """
        B, N, D = x_embedded.shape
        
        # Global average pooling to get sequence-level representation
        # This captures overall input characteristics
        x_pooled = x_embedded.mean(dim=1)  # (B, D)
        
        # Predict length logits
        length_logits = self.predictor(x_pooled)  # (B, num_length_bins)
        
        # Convert to probabilities
        length_probs = F.softmax(length_logits, dim=-1)  # (B, num_length_bins)
        
        # Select length bin (argmax for inference, Gumbel-Softmax for training)
        if self.training:
            # Gumbel-Softmax for differentiable sampling
            length_bin_onehot = F.gumbel_softmax(length_logits, tau=1.0, hard=True)  # (B, num_length_bins)
            length_bin_idx = length_bin_onehot.argmax(dim=-1)  # (B,)
        else:
            # Argmax for inference
            length_bin_idx = length_probs.argmax(dim=-1)  # (B,)
        
        # Map bin index to actual length
        length_bins_device = self.length_bins.to(x_embedded.device)
        predicted_lengths = length_bins_device[length_bin_idx]  # (B,)
        
        if return_distribution:
            return predicted_lengths, length_probs
        else:
            return predicted_lengths


class AdaptiveSequenceLengthWrapper(nn.Module):
    """
    Wrapper that adapts sequence length dynamically for each input.
    
    Predicts optimal length, truncates/pads input, processes with base model,
    then restores original length.
    """
    
    def __init__(self, base_model, max_seq_len=128, num_length_bins=8, length_penalty=0.01):
        """
        Args:
            base_model: underlying language model (e.g., ResNet-BK)
            max_seq_len: maximum sequence length
            num_length_bins: number of discrete length options
            length_penalty: penalty weight for using longer sequences
        """
        super().__init__()
        self.base_model = base_model
        self.max_seq_len = max_seq_len
        self.num_length_bins = num_length_bins
        self.length_penalty = length_penalty
        
        # Sequence length predictor
        self.length_predictor = SequenceLengthPredictor(
            d_model=base_model.d_model,
            max_seq_len=max_seq_len,
            num_length_bins=num_length_bins
        )
        
        # Track statistics
        self.register_buffer('avg_predicted_length', torch.tensor(0.0))
        self.register_buffer('length_distribution', torch.zeros(num_length_bins))
    
    def _pad_or_truncate(self, x, target_length):
        """
        Pad or truncate sequence to target length.
        
        Args:
            x: (B, N) - input token indices
            target_length: scalar - target sequence length
        
        Returns:
            x_adapted: (B, target_length) - adapted sequence
            original_length: scalar - original length N
        """
        B, N = x.shape
        original_length = N
        
        if target_length < N:
            # Truncate: keep first target_length tokens
            x_adapted = x[:, :target_length]
        elif target_length > N:
            # Pad: add padding tokens (assume 0 is padding token)
            padding = torch.zeros(B, target_length - N, dtype=x.dtype, device=x.device)
            x_adapted = torch.cat([x, padding], dim=1)
        else:
            # No change needed
            x_adapted = x
        
        return x_adapted, original_length
    
    def _restore_length(self, output, original_length, current_length):
        """
        Restore output to original sequence length.
        
        Args:
            output: (B, current_length, ...) - model output
            original_length: scalar - original sequence length
            current_length: scalar - current sequence length
        
        Returns:
            output_restored: (B, original_length, ...) - restored output
        """
        B = output.shape[0]
        
        if current_length < original_length:
            # Need to pad output
            # Use last token's output for padding positions
            last_output = output[:, -1:, ...]  # (B, 1, ...)
            padding_size = original_length - current_length
            padding = last_output.repeat(1, padding_size, *([1] * (output.dim() - 2)))
            output_restored = torch.cat([output, padding], dim=1)
        elif current_length > original_length:
            # Truncate output
            output_restored = output[:, :original_length, ...]
        else:
            # No change needed
            output_restored = output
        
        return output_restored
    
    def forward(self, x, use_adaptive_length=False):
        """
        Forward pass with optional adaptive sequence length.
        
        Args:
            x: (B, N) - input token indices
            use_adaptive_length: if True, predict and use optimal length
        
        Returns:
            logits: (B, N, vocab_size) - predictions (restored to original length)
            length_info: dict with length statistics (if use_adaptive_length=True)
        """
        B, N = x.shape
        
        if not use_adaptive_length:
            # Standard forward pass
            return self.base_model(x)
        
        # Adaptive length mode
        # Step 1: Get initial embeddings for length prediction
        tok_emb = self.base_model.token_embedding(x)  # (B, N, D)
        pos = torch.arange(0, N, dtype=torch.long, device=x.device).unsqueeze(0)
        pos_emb = self.base_model.position_embedding(pos)  # (1, N, D)
        x_embedded = tok_emb + pos_emb
        
        # Step 2: Predict optimal length
        predicted_lengths, length_probs = self.length_predictor(
            x_embedded, return_distribution=True
        )
        
        # Use maximum predicted length across batch for efficient batched processing
        # (alternative: process each sequence with its own length, but slower)
        max_predicted_length = predicted_lengths.max().item()
        
        # Ensure predicted length doesn't exceed model's n_seq
        max_predicted_length = min(max_predicted_length, self.base_model.n_seq)
        
        # Step 3: Adapt input to predicted length
        x_adapted, original_length = self._pad_or_truncate(x, max_predicted_length)
        
        # Step 4: Process with base model
        # Need to handle variable sequence length - process through model manually
        logits_adapted = self._forward_with_variable_length(x_adapted, max_predicted_length)
        
        # Step 5: Restore to original length
        logits = self._restore_length(logits_adapted, original_length, max_predicted_length)
        
        # Update statistics
        self.avg_predicted_length = predicted_lengths.float().mean()
        
        # Update length distribution
        for i in range(self.num_length_bins):
            length_bins_device = self.length_predictor.length_bins.to(x.device)
            count = (predicted_lengths == length_bins_device[i]).sum()
            self.length_distribution[i] += count
        
        # Prepare length info
        length_info = {
            'predicted_lengths': predicted_lengths,
            'length_probs': length_probs,
            'avg_predicted_length': self.avg_predicted_length.item(),
            'max_predicted_length': max_predicted_length,
            'original_length': original_length,
            'speedup_estimate': original_length / max_predicted_length
        }
        
        return logits, length_info
    
    def _forward_with_variable_length(self, x, seq_len):
        """
        Forward pass through base model with variable sequence length.
        
        Args:
            x: (B, seq_len) - input token indices
            seq_len: actual sequence length (may differ from model's n_seq)
        
        Returns:
            logits: (B, seq_len, vocab_size)
        """
        B, N = x.shape
        assert N == seq_len
        
        # Embeddings
        tok_emb = self.base_model.token_embedding(x)  # (B, N, D)
        pos = torch.arange(0, N, dtype=torch.long, device=x.device).unsqueeze(0)
        pos_emb = self.base_model.position_embedding(pos)  # (1, N, D)
        h = tok_emb + pos_emb
        
        # Process through blocks
        for block in self.base_model.blocks:
            # LayerNorm
            h_norm = block.layer_norm(h)
            
            # BK layer - need to handle variable length
            # The BK layer has fixed-size buffers, so we need to adjust them temporarily
            if hasattr(block.bk_layer, 'n_seq'):
                # Save original values
                original_n_seq = block.bk_layer.n_seq
                
                # Temporarily set n_seq to current length
                block.bk_layer.n_seq = seq_len
                
                # Also update h0 buffers if they exist
                if hasattr(block.bk_layer, 'h0_diag_base'):
                    original_h0_diag = block.bk_layer.h0_diag_base
                    original_h0_sub = block.bk_layer.h0_sub_base
                    original_h0_super = block.bk_layer.h0_super_base
                    
                    # Create new buffers for current sequence length
                    block.bk_layer.h0_diag_base = torch.full((seq_len,), -2.0, device=x.device)
                    block.bk_layer.h0_sub_base = torch.full((seq_len-1,), 1.0, device=x.device)
                    block.bk_layer.h0_super_base = torch.full((seq_len-1,), 1.0, device=x.device)
                    
                    # Forward pass
                    h_bk = block.bk_layer(h_norm)
                    
                    # Restore original buffers
                    block.bk_layer.h0_diag_base = original_h0_diag
                    block.bk_layer.h0_sub_base = original_h0_sub
                    block.bk_layer.h0_super_base = original_h0_super
                else:
                    # No h0 buffers, just forward
                    h_bk = block.bk_layer(h_norm)
                
                # Restore original n_seq
                block.bk_layer.n_seq = original_n_seq
            else:
                h_bk = block.bk_layer(h_norm)
            
            # Residual connection
            h = h + h_bk
        
        # Final output
        h_final = self.base_model.layer_norm_final(h)
        logits = self.base_model.lm_head(h_final)
        
        return logits
    
    def compute_loss(self, logits, targets, length_info=None):
        """
        Compute loss with optional length penalty.
        
        Loss = CE_loss + λ * length_penalty
        
        Args:
            logits: (B, N, vocab_size)
            targets: (B * N,) - flattened target indices
            length_info: dict with length information (if using adaptive length)
        
        Returns:
            total_loss: scalar
            ce_loss: scalar (for monitoring)
            length_cost: scalar (for monitoring)
        """
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets)
        
        if length_info is not None:
            # Penalize longer sequences to encourage efficiency
            # Normalize by max_seq_len so penalty is in [0, 1]
            avg_length_normalized = length_info['avg_predicted_length'] / self.max_seq_len
            length_cost = self.length_penalty * avg_length_normalized
            total_loss = ce_loss + length_cost
        else:
            length_cost = torch.tensor(0.0, device=ce_loss.device)
            total_loss = ce_loss
        
        return total_loss, ce_loss, length_cost
    
    def get_length_statistics(self):
        """
        Get statistics about predicted sequence lengths.
        
        Returns:
            stats: dict with average length and distribution
        """
        total_predictions = self.length_distribution.sum()
        
        if total_predictions == 0:
            return {
                'avg_predicted_length': 0.0,
                'length_distribution': [0.0] * self.num_length_bins,
                'avg_speedup': 1.0
            }
        
        # Length distribution (percentage at each bin)
        length_dist_pct = (self.length_distribution / total_predictions * 100).tolist()
        
        # Average predicted length
        avg_length = self.avg_predicted_length.item()
        
        # Average speedup estimate
        avg_speedup = self.max_seq_len / avg_length if avg_length > 0 else 1.0
        
        # Get actual length values for each bin
        length_bins_list = self.length_predictor.length_bins.tolist()
        
        return {
            'avg_predicted_length': avg_length,
            'length_distribution': length_dist_pct,
            'length_bins': length_bins_list,
            'avg_speedup': avg_speedup,
            'total_predictions': int(total_predictions.item())
        }
    
    def reset_length_statistics(self):
        """Reset length statistics counters."""
        self.avg_predicted_length = torch.tensor(0.0, device=self.avg_predicted_length.device)
        self.length_distribution = torch.zeros(
            self.num_length_bins,
            device=self.length_distribution.device
        )


class LearnedSequenceLengthTrainer:
    """
    Trainer for models with learned sequence length.
    Monitors length prediction and efficiency.
    """
    
    def __init__(self, model, optimizer, device='cuda'):
        """
        Args:
            model: AdaptiveSequenceLengthWrapper instance
            optimizer: PyTorch optimizer
            device: device to use
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
        # Statistics
        self.total_ce_loss = 0.0
        self.total_length_cost = 0.0
        self.num_batches = 0
    
    def train_step(self, x_batch, y_batch, use_adaptive_length=True):
        """
        Single training step.
        
        Args:
            x_batch: (B, N) - input token indices
            y_batch: (B * N,) - target token indices (flattened)
            use_adaptive_length: if True, use adaptive sequence length
        
        Returns:
            metrics: dict with loss components and length statistics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        if use_adaptive_length:
            logits, length_info = self.model(x_batch, use_adaptive_length=True)
        else:
            logits = self.model(x_batch, use_adaptive_length=False)
            length_info = None
        
        # Compute loss
        total_loss, ce_loss, length_cost = self.model.compute_loss(
            logits, y_batch, length_info
        )
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        
        # Update statistics
        self.total_ce_loss += ce_loss.item()
        self.total_length_cost += length_cost.item() if isinstance(length_cost, torch.Tensor) else length_cost
        self.num_batches += 1
        
        # Prepare metrics
        metrics = {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'length_cost': length_cost.item() if isinstance(length_cost, torch.Tensor) else length_cost
        }
        
        if length_info is not None:
            metrics.update({
                'avg_predicted_length': length_info['avg_predicted_length'],
                'speedup_estimate': length_info['speedup_estimate']
            })
        
        return metrics
    
    def evaluate(self, dataloader, use_adaptive_length=True):
        """
        Evaluate model on validation set.
        
        Args:
            dataloader: DataLoader with (input, target) batches
            use_adaptive_length: if True, use adaptive sequence length
        
        Returns:
            metrics: dict with evaluation metrics
        """
        self.model.eval()
        self.model.reset_length_statistics()
        
        total_loss = 0.0
        total_ce_loss = 0.0
        total_length_cost = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                if use_adaptive_length:
                    logits, length_info = self.model(x_batch, use_adaptive_length=True)
                else:
                    logits = self.model(x_batch, use_adaptive_length=False)
                    length_info = None
                
                # Compute loss
                total_loss_batch, ce_loss, length_cost = self.model.compute_loss(
                    logits, y_batch, length_info
                )
                
                total_loss += total_loss_batch.item()
                total_ce_loss += ce_loss.item()
                total_length_cost += length_cost.item() if isinstance(length_cost, torch.Tensor) else length_cost
                num_batches += 1
        
        # Compute averages
        avg_loss = total_loss / num_batches
        avg_ce_loss = total_ce_loss / num_batches
        avg_length_cost = total_length_cost / num_batches
        perplexity = torch.exp(torch.tensor(avg_ce_loss)).item()
        
        # Get length statistics
        length_stats = self.model.get_length_statistics()
        
        return {
            'loss': avg_loss,
            'ce_loss': avg_ce_loss,
            'length_cost': avg_length_cost,
            'perplexity': perplexity,
            **length_stats
        }
    
    def get_average_metrics(self):
        """Get average metrics across all training steps."""
        if self.num_batches == 0:
            return {'avg_ce_loss': 0.0, 'avg_length_cost': 0.0}
        
        return {
            'avg_ce_loss': self.total_ce_loss / self.num_batches,
            'avg_length_cost': self.total_length_cost / self.num_batches
        }
    
    def reset_statistics(self):
        """Reset training statistics."""
        self.total_ce_loss = 0.0
        self.total_length_cost = 0.0
        self.num_batches = 0
