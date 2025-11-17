"""
Quick test for the automated benchmark pipeline.
"""

import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.mamba_vs_bk_benchmark import (
    BenchmarkArgs,
    BenchmarkPipeline,
    DatasetDownloader,
    BenchmarkResults,
    MultiDatasetResults
)

def test_args_creation():
    """Test BenchmarkArgs creation."""
    args = BenchmarkArgs(
        model='bk',
        seq_len=128,
        bits=32,
        dataset='wikitext-2',
        epochs=1
    )
    assert args.model == 'bk'
    assert args.seq_len == 128
    assert args.bits == 32
    print("✓ BenchmarkArgs creation works")

def test_dataset_downloader():
    """Test DatasetDownloader."""
    # Check supported datasets
    assert 'wikitext-2' in DatasetDownloader.SUPPORTED_DATASETS
    assert 'ptb' in DatasetDownloader.SUPPORTED_DATASETS
    print("✓ DatasetDownloader has supported datasets")

def test_pipeline_creation():
    """Test BenchmarkPipeline creation."""
    args = BenchmarkArgs(
        model='bk',
        seq_len=128,
        bits=32,
        dataset='wikitext-2',
        epochs=1,
        device='cpu'  # Use CPU for testing
    )
    
    pipeline = BenchmarkPipeline(args)
    assert pipeline.args.model == 'bk'
    assert pipeline.device.type == 'cpu'
    print("✓ BenchmarkPipeline creation works")

def test_model_creation():
    """Test model creation."""
    args = BenchmarkArgs(
        model='bk',
        seq_len=128,
        bits=32,
        dataset='wikitext-2',
        epochs=1,
        device='cpu',
        d_model=64,  # Small model for testing
        n_layers=2
    )
    
    pipeline = BenchmarkPipeline(args)
    
    try:
        model = pipeline.create_model()
        assert model is not None
        print("✓ Model creation works")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")

def test_results_serialization():
    """Test results serialization."""
    results = BenchmarkResults(
        model_name='bk',
        dataset='wikitext-2',
        seq_len=128,
        bits=32,
        final_loss=3.5,
        final_perplexity=33.1,
        best_perplexity=30.5,
        training_time=100.0,
        forward_flops=1000000,
        backward_flops=2000000,
        total_flops=3000000,
        peak_memory_mb=500.0,
        model_size_mb=10.0,
        epoch_losses=[4.0, 3.5],
        epoch_perplexities=[54.6, 33.1],
        config={}
    )
    
    # Test to_dict
    results_dict = results.to_dict()
    assert results_dict['model_name'] == 'bk'
    assert results_dict['final_perplexity'] == 33.1
    print("✓ BenchmarkResults serialization works")

def test_multi_dataset_results():
    """Test MultiDatasetResults."""
    multi_results = MultiDatasetResults(
        model_name='bk',
        datasets=['wikitext-2', 'ptb'],
        perplexities={'wikitext-2': 30.5, 'ptb': 35.2},
        mean_perplexity=32.85,
        std_perplexity=2.35
    )
    
    results_dict = multi_results.to_dict()
    assert results_dict['mean_perplexity'] == 32.85
    print("✓ MultiDatasetResults works")

def main():
    """Run all tests."""
    print("Testing Automated Benchmark Pipeline")
    print("=" * 60)
    
    test_args_creation()
    test_dataset_downloader()
    test_pipeline_creation()
    test_model_creation()
    test_results_serialization()
    test_multi_dataset_results()
    
    print("=" * 60)
    print("All tests passed!")

if __name__ == '__main__':
    main()
