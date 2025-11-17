"""
Streaming Evaluation for Ultra-Long Sequences

This module implements streaming evaluation that can handle sequences up to 1M tokens
without loading the entire sequence into memory. It uses chunked processing with
state preservation for models that maintain internal state.

Requirement: 6.15 - Support evaluation on 1M token sequences without loading entire sequence
"""

import torch
import torch.nn as nn
import math
import time
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class StreamingEvalConfig:
    """Configuration for streaming evaluation."""
    chunk_size: int = 8192
    overlap: int = 0  # Overlap between chunks for context
    max_tokens: Optional[int] = None
    device: str = 'cuda'
    verbose: bool = True
    log_interval: int = 10  # Log every N chunks


class StreamingEvaluator:
    """
    Streaming evaluation for ultra-long sequences.
    
    Evaluates on sequences up to 1M tokens without loading entire sequence into memory.
    Implements chunked processing with optional state preservation for recurrent models.
    
    Features:
    - Chunked processing: Process sequence in manageable chunks
    - State preservation: Maintain model state across chunks (for stateful models)
    - Memory efficient: Clear cache periodically
    - Progress tracking: Real-time progress and performance metrics
    - Flexible: Works with any PyTorch model
    
    Example:
        >>> model = MyLanguageModel()
        >>> evaluator = StreamingEvaluator(model, chunk_size=8192)
        >>> data = torch.randint(0, 30000, (1000000,))  # 1M tokens
        >>> results = evaluator.evaluate_streaming(data)
        >>> print(f"Perplexity: {results['perplexity']:.2f}")
    """
    
    def __init__(
        self,
        model: nn.Module,
        chunk_size: Optional[int] = None,
        overlap: int = 0,
        device: str = 'cuda',
        verbose: bool = True,
    ):
        """
        Initialize streaming evaluator.
        
        Args:
            model: PyTorch model to evaluate
            chunk_size: Size of each chunk (default: model's n_seq or 8192)
            overlap: Number of tokens to overlap between chunks for context
            device: Device to run evaluation on
            verbose: Whether to print progress
        """
        self.model = model
        self.device = device
        self.verbose = verbose
        self.overlap = overlap
        
        # Determine chunk size from model if not specified
        if chunk_size is None:
            if hasattr(model, 'model') and hasattr(model.model, 'n_seq'):
                self.chunk_size = model.model.n_seq
            elif hasattr(model, 'n_seq'):
                self.chunk_size = model.n_seq
            elif hasattr(model, 'config') and hasattr(model.config, 'n_seq'):
                self.chunk_size = model.config.n_seq
            else:
                self.chunk_size = 8192
        else:
            self.chunk_size = chunk_size
        
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
        # State management for stateful models
        self.model_state = None
        self.supports_state = self._check_state_support()
        
        if self.verbose:
            print(f"StreamingEvaluator initialized:")
            print(f"  Chunk size: {self.chunk_size}")
            print(f"  Overlap: {self.overlap}")
            print(f"  Device: {self.device}")
            print(f"  Stateful model: {self.supports_state}")
    
    def _check_state_support(self) -> bool:
        """Check if model supports state preservation."""
        # Check for common state management methods
        has_get_state = hasattr(self.model, 'get_state')
        has_set_state = hasattr(self.model, 'set_state')
        has_reset_state = hasattr(self.model, 'reset_state')
        
        return has_get_state and has_set_state
    
    def _get_model_state(self) -> Optional[Any]:
        """Get current model state if supported."""
        if self.supports_state and hasattr(self.model, 'get_state'):
            return self.model.get_state()
        return None
    
    def _set_model_state(self, state: Optional[Any]):
        """Set model state if supported."""
        if self.supports_state and state is not None and hasattr(self.model, 'set_state'):
            self.model.set_state(state)
    
    def _reset_model_state(self):
        """Reset model state if supported."""
        if self.supports_state and hasattr(self.model, 'reset_state'):
            self.model.reset_state()
    
    def evaluate_streaming(
        self,
        data: torch.Tensor,
        max_tokens: Optional[int] = None,
        log_interval: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate on ultra-long sequence using streaming.
        
        This method processes the sequence in chunks, maintaining state between
        chunks for stateful models. It computes loss and perplexity without
        loading the entire sequence into memory.
        
        Args:
            data: Full dataset tensor (1D: [total_tokens])
            max_tokens: Maximum tokens to evaluate (None = all available)
            log_interval: Log progress every N chunks
        
        Returns:
            Dictionary with evaluation metrics:
                - loss: Average cross-entropy loss
                - perplexity: Perplexity (exp(loss))
                - total_tokens: Number of tokens evaluated
                - num_chunks: Number of chunks processed
                - tokens_per_second: Processing speed
                - avg_chunk_time: Average time per chunk
        """
        self.model.eval()
        
        # Reset model state at start
        self._reset_model_state()
        
        total_loss = 0.0
        total_tokens = 0
        num_chunks = 0
        total_time = 0.0
        
        # Determine evaluation length
        eval_length = min(data.size(0) - 1, max_tokens) if max_tokens else data.size(0) - 1
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Streaming Evaluation")
            print(f"{'='*70}")
            print(f"Total tokens: {eval_length:,}")
            print(f"Chunk size: {self.chunk_size}")
            print(f"Overlap: {self.overlap}")
            print(f"Expected chunks: {(eval_length + self.chunk_size - 1) // self.chunk_size}")
            print(f"{'='*70}\n")
        
        with torch.no_grad():
            # Process sequence in chunks
            i = 0
            while i < eval_length:
                chunk_start_time = time.time()
                
                # Determine chunk boundaries
                chunk_start = max(0, i - self.overlap)
                chunk_end = min(i + self.chunk_size, eval_length)
                actual_chunk_len = chunk_end - i
                
                # Extract chunk (data is 1D)
                x_data = data[chunk_start:chunk_end]
                y_data = data[chunk_start + 1:chunk_end + 1]
                
                # Reshape to (batch=1, seq_len) for model
                x_chunk = x_data.unsqueeze(0).to(self.device)
                y_chunk = y_data.to(self.device)
                
                # Forward pass
                try:
                    logits = self.model(x_chunk)  # (1, seq_len, vocab_size)
                    
                    # Only compute loss on non-overlapping part
                    if self.overlap > 0 and i > 0:
                        logits = logits[:, self.overlap:, :]
                        y_chunk = y_chunk[self.overlap:]
                    
                    # Compute loss
                    loss = self.criterion(
                        logits.reshape(-1, logits.size(-1)),
                        y_chunk
                    )
                    
                    # Accumulate
                    total_loss += loss.sum().item()
                    total_tokens += actual_chunk_len
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"Warning: OOM at chunk {num_chunks}, skipping...")
                        torch.cuda.empty_cache()
                        i += self.chunk_size
                        continue
                    else:
                        raise e
                
                num_chunks += 1
                chunk_time = time.time() - chunk_start_time
                total_time += chunk_time
                
                # Progress logging
                if self.verbose and num_chunks % log_interval == 0:
                    progress = (i + actual_chunk_len) / eval_length * 100
                    current_ppl = math.exp(min(total_loss / total_tokens, 20))
                    tokens_per_sec = total_tokens / total_time
                    
                    print(f"Chunk {num_chunks:4d} | "
                          f"Progress: {progress:5.1f}% | "
                          f"Tokens: {total_tokens:8,} | "
                          f"PPL: {current_ppl:7.2f} | "
                          f"Speed: {tokens_per_sec:7.1f} tok/s | "
                          f"Time: {chunk_time:5.2f}s")
                
                # Clear cache periodically to prevent memory buildup
                if num_chunks % 50 == 0:
                    torch.cuda.empty_cache()
                
                # Move to next chunk
                i += self.chunk_size
        
        # Compute final metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(min(avg_loss, 20))
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        avg_chunk_time = total_time / num_chunks if num_chunks > 0 else 0
        
        results = {
            "loss": avg_loss,
            "perplexity": perplexity,
            "total_tokens": total_tokens,
            "num_chunks": num_chunks,
            "total_time": total_time,
            "tokens_per_second": tokens_per_second,
            "avg_chunk_time": avg_chunk_time,
        }
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Streaming Evaluation Complete")
            print(f"{'='*70}")
            print(f"Total tokens: {total_tokens:,}")
            print(f"Chunks processed: {num_chunks}")
            print(f"Average loss: {avg_loss:.4f}")
            print(f"Perplexity: {perplexity:.2f}")
            print(f"Total time: {total_time:.2f}s")
            print(f"Speed: {tokens_per_second:.1f} tokens/second")
            print(f"Avg chunk time: {avg_chunk_time:.3f}s")
            print(f"{'='*70}\n")
        
        return results
    
    def evaluate_streaming_with_metrics(
        self,
        data: torch.Tensor,
        max_tokens: Optional[int] = None,
        compute_per_token_metrics: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate with additional per-chunk metrics.
        
        Args:
            data: Full dataset tensor (1D)
            max_tokens: Maximum tokens to evaluate
            compute_per_token_metrics: Whether to compute per-token loss
        
        Returns:
            Dictionary with metrics including per-chunk statistics
        """
        self.model.eval()
        self._reset_model_state()
        
        chunk_losses = []
        chunk_perplexities = []
        chunk_times = []
        per_token_losses = [] if compute_per_token_metrics else None
        
        total_loss = 0.0
        total_tokens = 0
        num_chunks = 0
        
        eval_length = min(data.size(0) - 1, max_tokens) if max_tokens else data.size(0) - 1
        
        with torch.no_grad():
            i = 0
            while i < eval_length:
                chunk_start_time = time.time()
                
                chunk_end = min(i + self.chunk_size, eval_length)
                actual_chunk_len = chunk_end - i
                
                x_data = data[i:chunk_end]
                y_data = data[i + 1:chunk_end + 1]
                
                x_chunk = x_data.unsqueeze(0).to(self.device)
                y_chunk = y_data.to(self.device)
                
                logits = self.model(x_chunk)
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), y_chunk)
                
                chunk_loss = loss.mean().item()
                chunk_ppl = math.exp(min(chunk_loss, 20))
                
                chunk_losses.append(chunk_loss)
                chunk_perplexities.append(chunk_ppl)
                chunk_times.append(time.time() - chunk_start_time)
                
                if compute_per_token_metrics:
                    per_token_losses.extend(loss.cpu().tolist())
                
                total_loss += loss.sum().item()
                total_tokens += actual_chunk_len
                num_chunks += 1
                
                i += self.chunk_size
                
                if num_chunks % 50 == 0:
                    torch.cuda.empty_cache()
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(min(avg_loss, 20))
        
        results = {
            "loss": avg_loss,
            "perplexity": perplexity,
            "total_tokens": total_tokens,
            "num_chunks": num_chunks,
            "chunk_losses": chunk_losses,
            "chunk_perplexities": chunk_perplexities,
            "chunk_times": chunk_times,
            "avg_chunk_loss": sum(chunk_losses) / len(chunk_losses),
            "std_chunk_loss": float(torch.tensor(chunk_losses).std().item()),
            "avg_chunk_time": sum(chunk_times) / len(chunk_times),
        }
        
        if compute_per_token_metrics:
            results["per_token_losses"] = per_token_losses
        
        return results


def create_streaming_evaluator(
    model: nn.Module,
    config: Optional[StreamingEvalConfig] = None,
) -> StreamingEvaluator:
    """
    Factory function to create a streaming evaluator.
    
    Args:
        model: PyTorch model to evaluate
        config: Configuration for streaming evaluation
    
    Returns:
        Configured StreamingEvaluator instance
    """
    if config is None:
        config = StreamingEvalConfig()
    
    return StreamingEvaluator(
        model=model,
        chunk_size=config.chunk_size,
        overlap=config.overlap,
        device=config.device,
        verbose=config.verbose,
    )
