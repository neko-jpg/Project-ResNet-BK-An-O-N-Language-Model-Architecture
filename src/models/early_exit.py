"""
Early Exiting for Inference
Implements dynamic layer execution based on output confidence.
Halts computation when confidence threshold is reached.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_bk import MoEResNetBKLayer


class EarlyExitResNetBKBlock(nn.Module):
    """
    ResNet-BK block with early exit capability.
    Each layer can produce predictions, and computation halts when confidence is high enough.
    
    Architecture:
        Input -> LayerNorm -> MoEResNetBKLayer -> Exit Classifier (optional)
    """
    
    def __init__(self, d_model, n_seq, vocab_size, num_experts=4, top_k=1, dropout_p=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_seq = n_seq
        self.vocab_size = vocab_size
        
        # Standard ResNet-BK components
        self.layer_norm = nn.LayerNorm(d_model)
        self.bk_layer = MoEResNetBKLayer(d_model, n_seq, num_experts, top_k, dropout_p)
        
        # Exit classifier: produces predictions at this layer
        # Lightweight classifier to minimize overhead
        self.exit_classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )
    
    def forward(self, x):
        """
        Forward pass with exit point.
        
        Args:
            x: (B, N, D) - input tensor
        
        Returns:
            output: (B, N, D) - processed output
            exit_logits: (B, N, vocab_size) - predictions at this layer
        """
        # Process through ResNet-BK layer
        x_normalized = self.layer_norm(x)
        x_processed = x + self.bk_layer(x_normalized)  # Residual connection
        
        # Compute exit predictions
        exit_logits = self.exit_classifier(x_processed)
        
        return x_processed, exit_logits


class EarlyExitLanguageModel(nn.Module):
    """
    Language model with early exiting capability.
    
    Dynamically determines when to stop processing based on prediction confidence.
    Each layer can produce predictions, and computation halts when confidence exceeds threshold.
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
        confidence_threshold=0.9
    ):
        super().__init__()
        self.d_model = d_model
        self.n_seq = n_seq
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.confidence_threshold = confidence_threshold
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(n_seq, d_model)
        
        # Early exit ResNet-BK blocks
        self.blocks = nn.ModuleList([
            EarlyExitResNetBKBlock(
                d_model=d_model,
                n_seq=n_seq,
                vocab_size=vocab_size,
                num_experts=num_experts,
                top_k=top_k,
                dropout_p=dropout_p
            )
            for _ in range(n_layers)
        ])
        
        # Final output layers (used if no early exit)
        self.layer_norm_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Track statistics
        self.register_buffer('avg_exit_layer', torch.tensor(0.0))
        self.register_buffer('exit_layer_counts', torch.zeros(n_layers + 1))
    
    def forward(self, x, use_early_exit=False):
        """
        Forward pass with optional early exiting.
        
        Args:
            x: (B, N) - token indices
            use_early_exit: if True, use early exiting based on confidence
        
        Returns:
            logits: (B, N, vocab_size)
            exit_info: dict with exit statistics (if use_early_exit=True)
        """
        B, N = x.shape
        assert N == self.n_seq, f"Sequence length mismatch: expected {self.n_seq}, got {N}"
        
        # Embeddings
        tok_emb = self.token_embedding(x)  # (B, N, D)
        pos = torch.arange(0, N, dtype=torch.long, device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)  # (1, N, D)
        h = tok_emb + pos_emb
        
        if not use_early_exit:
            # Standard forward pass through all layers
            for block in self.blocks:
                h, _ = block(h)
            
            # Final output
            h_final = self.layer_norm_final(h)
            logits = self.lm_head(h_final)
            
            return logits
        
        else:
            # Early exit mode: check confidence at each layer
            exit_info = {
                'exit_layers': torch.full((B, N), self.n_layers, dtype=torch.long, device=x.device),
                'exit_confidences': torch.zeros(B, N, device=x.device),
                'exited_mask': torch.zeros(B, N, dtype=torch.bool, device=x.device)
            }
            
            # Accumulate final predictions
            final_logits = torch.zeros(B, N, self.vocab_size, device=x.device)
            
            for layer_idx, block in enumerate(self.blocks):
                # Process layer
                h, exit_logits = block(h)
                
                # Compute confidence (max probability)
                probs = F.softmax(exit_logits, dim=-1)
                max_probs, _ = probs.max(dim=-1)  # (B, N)
                
                # Determine which tokens should exit at this layer
                should_exit = (max_probs >= self.confidence_threshold) & (~exit_info['exited_mask'])
                
                # Update exit info for tokens that exit
                exit_info['exit_layers'][should_exit] = layer_idx
                exit_info['exit_confidences'][should_exit] = max_probs[should_exit]
                exit_info['exited_mask'] |= should_exit
                
                # Store predictions for exited tokens
                final_logits[should_exit] = exit_logits[should_exit]
                
                # Early termination if all tokens exited
                if exit_info['exited_mask'].all():
                    break
            
            # For tokens that didn't exit, use final layer predictions
            if not exit_info['exited_mask'].all():
                h_final = self.layer_norm_final(h)
                final_layer_logits = self.lm_head(h_final)
                
                not_exited = ~exit_info['exited_mask']
                final_logits[not_exited] = final_layer_logits[not_exited]
                
                # Update exit info for remaining tokens
                probs = F.softmax(final_layer_logits, dim=-1)
                max_probs, _ = probs.max(dim=-1)
                exit_info['exit_confidences'][not_exited] = max_probs[not_exited]
            
            # Update statistics
            avg_exit = exit_info['exit_layers'].float().mean()
            self.avg_exit_layer = avg_exit
            
            # Update exit layer counts
            for layer_idx in range(self.n_layers + 1):
                count = (exit_info['exit_layers'] == layer_idx).sum()
                self.exit_layer_counts[layer_idx] += count
            
            return final_logits, exit_info
    
    def get_exit_statistics(self):
        """
        Get statistics about early exiting behavior.
        
        Returns:
            stats: dict with average exit layer and distribution
        """
        total_tokens = self.exit_layer_counts.sum()
        
        if total_tokens == 0:
            return {
                'avg_exit_layer': 0.0,
                'exit_distribution': [0.0] * (self.n_layers + 1),
                'speedup_estimate': 1.0
            }
        
        # Exit distribution (percentage at each layer)
        exit_distribution = (self.exit_layer_counts / total_tokens * 100).tolist()
        
        # Average exit layer
        avg_exit = self.avg_exit_layer.item()
        
        # Speedup estimate: (avg_exit_layer + 1) / n_layers
        # +1 because layer indices are 0-based
        speedup_estimate = self.n_layers / (avg_exit + 1)
        
        return {
            'avg_exit_layer': avg_exit,
            'exit_distribution': exit_distribution,
            'speedup_estimate': speedup_estimate,
            'total_tokens_processed': int(total_tokens.item())
        }
    
    def reset_exit_statistics(self):
        """Reset exit statistics counters."""
        self.avg_exit_layer = torch.tensor(0.0, device=self.avg_exit_layer.device)
        self.exit_layer_counts = torch.zeros(self.n_layers + 1, device=self.exit_layer_counts.device)


class EarlyExitEvaluator:
    """
    Evaluator for early exit models.
    Measures performance and efficiency of early exiting.
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    def evaluate(self, dataloader, confidence_thresholds=[0.7, 0.8, 0.9, 0.95]):
        """
        Evaluate early exit performance across different confidence thresholds.
        
        Args:
            dataloader: DataLoader with (input, target) batches
            confidence_thresholds: list of thresholds to test
        
        Returns:
            results: dict mapping threshold -> metrics
        """
        self.model.eval()
        results = {}
        
        for threshold in confidence_thresholds:
            # Set threshold
            self.model.confidence_threshold = threshold
            self.model.reset_exit_statistics()
            
            total_loss = 0.0
            total_tokens = 0
            num_batches = 0
            
            with torch.no_grad():
                for x_batch, y_batch in dataloader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    # Forward with early exit
                    logits, exit_info = self.model(x_batch, use_early_exit=True)
                    
                    # Compute loss
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y_batch.view(-1)
                    )
                    
                    total_loss += loss.item()
                    total_tokens += x_batch.numel()
                    num_batches += 1
            
            # Get statistics
            stats = self.model.get_exit_statistics()
            
            # Compute perplexity
            avg_loss = total_loss / num_batches
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            
            results[threshold] = {
                'perplexity': perplexity,
                'avg_exit_layer': stats['avg_exit_layer'],
                'exit_distribution': stats['exit_distribution'],
                'speedup_estimate': stats['speedup_estimate'],
                'total_tokens': stats['total_tokens_processed']
            }
        
        return results
    
    def benchmark_speedup(self, dataloader, num_batches=100):
        """
        Benchmark actual speedup from early exiting.
        
        Args:
            dataloader: DataLoader with (input, target) batches
            num_batches: number of batches to benchmark
        
        Returns:
            speedup_metrics: dict with timing comparisons
        """
        import time
        
        self.model.eval()
        
        # Benchmark without early exit
        start_time = time.time()
        with torch.no_grad():
            for i, (x_batch, _) in enumerate(dataloader):
                if i >= num_batches:
                    break
                x_batch = x_batch.to(self.device)
                _ = self.model(x_batch, use_early_exit=False)
        time_no_exit = time.time() - start_time
        
        # Benchmark with early exit
        self.model.reset_exit_statistics()
        start_time = time.time()
        with torch.no_grad():
            for i, (x_batch, _) in enumerate(dataloader):
                if i >= num_batches:
                    break
                x_batch = x_batch.to(self.device)
                _, _ = self.model(x_batch, use_early_exit=True)
        time_with_exit = time.time() - start_time
        
        # Get exit statistics
        stats = self.model.get_exit_statistics()
        
        # Compute speedup
        actual_speedup = time_no_exit / time_with_exit if time_with_exit > 0 else 1.0
        
        return {
            'time_no_exit': time_no_exit,
            'time_with_exit': time_with_exit,
            'actual_speedup': actual_speedup,
            'theoretical_speedup': stats['speedup_estimate'],
            'avg_exit_layer': stats['avg_exit_layer'],
            'exit_distribution': stats['exit_distribution']
        }
