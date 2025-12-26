#!/usr/bin/env python3
"""Quick data sampling test - simpler version without full tokenizer."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import numpy as np
from src.utils.data_utils import BinaryIndexedDataset, MixedBinaryDataset

def test_sampling():
    """Test that sample_sequence works with short document concatenation."""
    
    print("=" * 70)
    print("データサンプリングテスト")
    print("=" * 70)
    
    # Test individual datasets
    datasets = [
        ("data/japanese_instruct", "Japanese Instruct"),
        ("data/wiki_ja", "Japanese Wikipedia"),
        ("data/mc4_ja", "mC4 Japanese"),
    ]
    
    for path, name in datasets:
        idx_path = Path(path) / "train.idx"
        if not idx_path.exists():
            print(f"\n{name}: NOT FOUND")
            continue
        
        try:
            ds = BinaryIndexedDataset(path, split="train")
            rng = random.Random(42)
            
            # Test sampling at different seq_len values
            print(f"\n{name}:")
            print(f"  Total docs: {ds.num_docs}")
            
            for seq_len in [256, 512, 1024]:
                success = 0
                for _ in range(10):
                    result = ds.sample_sequence(seq_len, rng)
                    if result is not None:
                        x, y = result
                        if len(x) == seq_len and len(y) == seq_len:
                            success += 1
                print(f"  seq_len={seq_len}: {success}/10 successes")
                
        except Exception as e:
            print(f"\n{name}: ERROR - {e}")
    
    # Test MixedBinaryDataset with config
    print("\n" + "=" * 70)
    print("MixedBinaryDataset テスト")
    print("=" * 70)
    
    config_path = "configs/dataset_japanese_chat_optimized.yaml"
    if not Path(config_path).exists():
        print(f"Config not found: {config_path}")
        return
    
    try:
        ds = MixedBinaryDataset(
            config_path=config_path,
            batch_size=4,
            seq_len=512,
            total_tokens=100000,
            seed=42,
            vocab_size=32768,
        )
        
        print(f"Dataset loaded successfully")
        print(f"  Datasets: {ds.dataset_names}")
        print(f"  Weights: {ds.weights}")
        print(f"  Steps per epoch: {ds.steps_per_epoch}")
        
        # Sample a few batches
        batch_count = 0
        for x_batch, y_batch in ds.iter_epoch(epoch=0):
            batch_count += 1
            if batch_count <= 3:
                print(f"\n  Batch {batch_count}: x={x_batch.shape}, y={y_batch.shape}")
                print(f"    x sample tokens: {x_batch[0, :20].tolist()}")
            if batch_count >= 5:
                break
        
        print(f"\n  Successfully sampled {batch_count} batches!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_sampling()
