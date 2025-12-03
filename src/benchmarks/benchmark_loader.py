
import time
import argparse
import torch
import numpy as np
from src.utils.data_utils import BinaryIndexedDataset
import os

def create_dummy_dataset(path, num_docs=1000, doc_len=2048):
    """Create a dummy .bin/.idx dataset for benchmarking."""
    os.makedirs(path, exist_ok=True)
    bin_path = os.path.join(path, "train.bin")
    idx_path = os.path.join(path, "train.idx")
    
    if os.path.exists(bin_path) and os.path.exists(idx_path):
        return

    print(f"Creating dummy dataset at {path}...")
    
    # Create random tokens
    total_tokens = num_docs * doc_len
    tokens = np.random.randint(0, 10000, size=total_tokens, dtype=np.uint32)
    tokens.tofile(bin_path)
    
    # Create index
    with open(idx_path, "wb") as f:
        f.write(b"MUSE")
        f.write((1).to_bytes(4, 'little'))
        
        for i in range(num_docs):
            offset = i * doc_len
            length = doc_len
            f.write(offset.to_bytes(8, 'little'))
            f.write(length.to_bytes(8, 'little'))

def benchmark_loader(dataset_path, batch_size=32, seq_len=1024, steps=1000):
    print(f"\nBenchmarking Data Loader (B={batch_size}, L={seq_len}, Steps={steps})...")
    
    # Create dummy dataset if not exists
    create_dummy_dataset(dataset_path)
    
    # Load Dataset
    try:
        ds = BinaryIndexedDataset(dataset_path, split="train")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print(f"Using Rust: {ds.use_rust}")
    
    import random
    rng = random.Random(42)
    
    # Warmup
    for _ in range(10):
        ds.sample_sequence(seq_len, rng)
        
    start = time.time()
    
    for _ in range(steps):
        # Simulate batch construction
        batch = []
        for _ in range(batch_size):
            res = ds.sample_sequence(seq_len, rng)
            if res:
                batch.append(res)
        
        # Stack (simulate collation)
        if batch:
            x = torch.stack([torch.from_numpy(b[0].astype(np.int64)) for b in batch])
            y = torch.stack([torch.from_numpy(b[1].astype(np.int64)) for b in batch])
            
    end = time.time()
    duration = end - start
    throughput = (steps * batch_size * seq_len) / duration
    
    print(f"Time: {duration:.2f} s")
    print(f"Throughput: {throughput/1e6:.2f} M tokens/sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data/dummy_benchmark')
    args = parser.parse_args()
    
    benchmark_loader(args.path)
