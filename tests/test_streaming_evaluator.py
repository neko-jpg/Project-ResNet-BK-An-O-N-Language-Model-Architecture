"""
Tests for Streaming Evaluator

Tests the streaming evaluation functionality for ultra-long sequences.
"""

import torch
import torch.nn as nn
import pytest
import math
from src.benchmarks.streaming_evaluator import (
    StreamingEvaluator,
    StreamingEvalConfig,
    create_streaming_evaluator,
)


class SimpleLanguageModel(nn.Module):
    """Simple language model for testing."""
    
    def __init__(self, vocab_size=1000, d_model=128, n_seq=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_seq = n_seq
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=512, batch_first=True),
            num_layers=2
        )
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        """Forward pass."""
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, d_model)
        hidden = self.transformer(emb)  # (batch, seq_len, d_model)
        logits = self.output(hidden)  # (batch, seq_len, vocab_size)
        return logits


class StatefulLanguageModel(nn.Module):
    """Stateful language model for testing state preservation."""
    
    def __init__(self, vocab_size=1000, d_model=128, n_seq=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_seq = n_seq
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.LSTM(d_model, d_model, num_layers=2, batch_first=True)
        self.output = nn.Linear(d_model, vocab_size)
        
        self.hidden_state = None
    
    def forward(self, x):
        """Forward pass with state."""
        emb = self.embedding(x)
        
        if self.hidden_state is not None:
            hidden, self.hidden_state = self.rnn(emb, self.hidden_state)
        else:
            hidden, self.hidden_state = self.rnn(emb)
        
        logits = self.output(hidden)
        return logits
    
    def get_state(self):
        """Get current state."""
        return self.hidden_state
    
    def set_state(self, state):
        """Set state."""
        self.hidden_state = state
    
    def reset_state(self):
        """Reset state."""
        self.hidden_state = None


@pytest.fixture
def simple_model():
    """Create simple model for testing."""
    return SimpleLanguageModel(vocab_size=1000, d_model=64, n_seq=256)


@pytest.fixture
def stateful_model():
    """Create stateful model for testing."""
    return StatefulLanguageModel(vocab_size=1000, d_model=64, n_seq=256)


@pytest.fixture
def test_data():
    """Create test data."""
    # Create 10K tokens
    return torch.randint(0, 1000, (10000,))


@pytest.fixture
def long_test_data():
    """Create long test data (100K tokens)."""
    return torch.randint(0, 1000, (100000,))


def test_streaming_evaluator_initialization(simple_model):
    """Test StreamingEvaluator initialization."""
    evaluator = StreamingEvaluator(simple_model, chunk_size=512, verbose=False)
    
    assert evaluator.chunk_size == 512
    assert evaluator.overlap == 0
    assert evaluator.device == 'cuda'
    assert evaluator.model == simple_model


def test_streaming_evaluator_auto_chunk_size(simple_model):
    """Test automatic chunk size detection."""
    evaluator = StreamingEvaluator(simple_model, verbose=False)
    
    # Should use model's n_seq
    assert evaluator.chunk_size == 256


def test_streaming_evaluation_basic(simple_model, test_data):
    """Test basic streaming evaluation."""
    evaluator = StreamingEvaluator(
        simple_model,
        chunk_size=512,
        device='cpu',
        verbose=False
    )
    
    results = evaluator.evaluate_streaming(test_data)
    
    # Check results structure
    assert 'loss' in results
    assert 'perplexity' in results
    assert 'total_tokens' in results
    assert 'num_chunks' in results
    assert 'tokens_per_second' in results
    
    # Check values are reasonable
    assert results['total_tokens'] == len(test_data) - 1
    assert results['num_chunks'] > 0
    assert results['loss'] > 0
    assert results['perplexity'] > 1.0
    assert results['tokens_per_second'] > 0


def test_streaming_evaluation_max_tokens(simple_model, test_data):
    """Test streaming evaluation with max_tokens limit."""
    evaluator = StreamingEvaluator(
        simple_model,
        chunk_size=512,
        device='cpu',
        verbose=False
    )
    
    max_tokens = 2000
    results = evaluator.evaluate_streaming(test_data, max_tokens=max_tokens)
    
    assert results['total_tokens'] == max_tokens


def test_streaming_evaluation_chunking(simple_model, test_data):
    """Test that chunking produces correct number of chunks."""
    chunk_size = 512
    evaluator = StreamingEvaluator(
        simple_model,
        chunk_size=chunk_size,
        device='cpu',
        verbose=False
    )
    
    results = evaluator.evaluate_streaming(test_data)
    
    expected_chunks = (len(test_data) - 1 + chunk_size - 1) // chunk_size
    assert results['num_chunks'] == expected_chunks


def test_streaming_evaluation_with_overlap(simple_model, test_data):
    """Test streaming evaluation with overlap."""
    evaluator = StreamingEvaluator(
        simple_model,
        chunk_size=512,
        overlap=64,
        device='cpu',
        verbose=False
    )
    
    results = evaluator.evaluate_streaming(test_data)
    
    # Should still process all tokens
    assert results['total_tokens'] == len(test_data) - 1
    assert results['perplexity'] > 1.0


def test_streaming_evaluation_long_sequence(simple_model, long_test_data):
    """Test streaming evaluation on long sequence (100K tokens)."""
    evaluator = StreamingEvaluator(
        simple_model,
        chunk_size=1024,
        device='cpu',
        verbose=False
    )
    
    results = evaluator.evaluate_streaming(long_test_data)
    
    assert results['total_tokens'] == len(long_test_data) - 1
    assert results['num_chunks'] > 90  # Should be ~97 chunks
    assert results['perplexity'] > 1.0


def test_streaming_evaluation_stateful_model(stateful_model, test_data):
    """Test streaming evaluation with stateful model."""
    evaluator = StreamingEvaluator(
        stateful_model,
        chunk_size=512,
        device='cpu',
        verbose=False
    )
    
    # Check state support detection
    assert evaluator.supports_state
    
    results = evaluator.evaluate_streaming(test_data)
    
    assert results['total_tokens'] == len(test_data) - 1
    assert results['perplexity'] > 1.0


def test_streaming_evaluation_with_metrics(simple_model, test_data):
    """Test streaming evaluation with additional metrics."""
    evaluator = StreamingEvaluator(
        simple_model,
        chunk_size=512,
        device='cpu',
        verbose=False
    )
    
    results = evaluator.evaluate_streaming_with_metrics(test_data)
    
    # Check additional metrics
    assert 'chunk_losses' in results
    assert 'chunk_perplexities' in results
    assert 'chunk_times' in results
    assert 'avg_chunk_loss' in results
    assert 'std_chunk_loss' in results
    assert 'avg_chunk_time' in results
    
    # Check lengths match
    assert len(results['chunk_losses']) == results['num_chunks']
    assert len(results['chunk_perplexities']) == results['num_chunks']
    assert len(results['chunk_times']) == results['num_chunks']


def test_streaming_evaluation_per_token_metrics(simple_model, test_data):
    """Test streaming evaluation with per-token metrics."""
    evaluator = StreamingEvaluator(
        simple_model,
        chunk_size=512,
        device='cpu',
        verbose=False
    )
    
    results = evaluator.evaluate_streaming_with_metrics(
        test_data,
        compute_per_token_metrics=True
    )
    
    assert 'per_token_losses' in results
    assert len(results['per_token_losses']) == results['total_tokens']


def test_streaming_evaluator_factory():
    """Test factory function."""
    model = SimpleLanguageModel()
    config = StreamingEvalConfig(
        chunk_size=1024,
        overlap=128,
        device='cpu',
        verbose=False
    )
    
    evaluator = create_streaming_evaluator(model, config)
    
    assert evaluator.chunk_size == 1024
    assert evaluator.overlap == 128
    assert evaluator.device == 'cpu'


def test_streaming_evaluator_factory_default_config():
    """Test factory function with default config."""
    model = SimpleLanguageModel()
    evaluator = create_streaming_evaluator(model)
    
    assert evaluator.chunk_size > 0
    assert evaluator.overlap == 0


def test_streaming_evaluation_consistency(simple_model, test_data):
    """Test that streaming evaluation is consistent across runs."""
    evaluator = StreamingEvaluator(
        simple_model,
        chunk_size=512,
        device='cpu',
        verbose=False
    )
    
    # Run twice
    results1 = evaluator.evaluate_streaming(test_data)
    results2 = evaluator.evaluate_streaming(test_data)
    
    # Should get same results (model in eval mode, no dropout)
    assert abs(results1['loss'] - results2['loss']) < 1e-5
    assert abs(results1['perplexity'] - results2['perplexity']) < 1e-3


def test_streaming_evaluation_memory_efficiency(simple_model):
    """Test that streaming evaluation doesn't accumulate memory."""
    evaluator = StreamingEvaluator(
        simple_model,
        chunk_size=512,
        device='cpu',
        verbose=False
    )
    
    # Create very long sequence
    long_data = torch.randint(0, 1000, (50000,))
    
    # Should complete without OOM
    results = evaluator.evaluate_streaming(long_data)
    
    assert results['total_tokens'] == len(long_data) - 1


def test_streaming_evaluation_small_sequence(simple_model):
    """Test streaming evaluation on sequence smaller than chunk size."""
    evaluator = StreamingEvaluator(
        simple_model,
        chunk_size=1024,
        device='cpu',
        verbose=False
    )
    
    small_data = torch.randint(0, 1000, (500,))
    results = evaluator.evaluate_streaming(small_data)
    
    assert results['total_tokens'] == len(small_data) - 1
    assert results['num_chunks'] == 1


def test_streaming_evaluation_exact_chunk_size(simple_model):
    """Test streaming evaluation when data is exact multiple of chunk size."""
    chunk_size = 512
    evaluator = StreamingEvaluator(
        simple_model,
        chunk_size=chunk_size,
        device='cpu',
        verbose=False
    )
    
    # Create data that's exactly 2 chunks
    exact_data = torch.randint(0, 1000, (chunk_size * 2 + 1,))
    results = evaluator.evaluate_streaming(exact_data)
    
    assert results['num_chunks'] == 2
    assert results['total_tokens'] == chunk_size * 2


def test_streaming_evaluation_progress_logging(simple_model, test_data, capsys):
    """Test that progress logging works."""
    evaluator = StreamingEvaluator(
        simple_model,
        chunk_size=512,
        device='cpu',
        verbose=True  # Enable verbose
    )
    
    results = evaluator.evaluate_streaming(test_data, log_interval=5)
    
    # Check that something was printed
    captured = capsys.readouterr()
    assert "Streaming Evaluation" in captured.out
    assert "Complete" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
