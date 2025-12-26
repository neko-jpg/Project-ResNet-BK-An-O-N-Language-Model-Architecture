#!/usr/bin/env python3
"""Analyze dataset document lengths to diagnose short document skipping issue."""

import numpy as np
import struct
from pathlib import Path

def analyze_dataset(data_path: str, split: str = "train"):
    """Analyze document lengths in a binary indexed dataset."""
    root = Path(data_path)
    idx_path = root / f"{split}.idx"
    
    if not idx_path.exists():
        print(f"  {data_path}/{split}: NOT FOUND")
        return None
    
    with open(idx_path, "rb") as f:
        magic = f.read(4)
        if magic != b"MUSE":
            print(f"  Invalid magic: {magic}")
            return None
        _version = struct.unpack("<I", f.read(4))[0]
        idx_data = np.fromfile(f, dtype=np.uint64)
    
    if idx_data.size % 2 != 0:
        print(f"  Corrupted idx file")
        return None
    
    index = idx_data.reshape(-1, 2)  # (offset, length)
    lengths = index[:, 1]
    
    return {
        "num_docs": len(lengths),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "lengths": lengths,
    }

def main():
    datasets = [
        ("data/japanese_instruct", "Japanese Instruct"),
        ("data/dolly_ja", "Dolly Japanese"),
        ("data/wiki_ja", "Japanese Wikipedia"),
        ("data/mc4_ja", "mC4 Japanese"),
    ]
    
    seq_lengths_to_check = [512, 1024, 2048]
    
    print("=" * 70)
    print("Document Length Analysis for Conversation Data Pipeline")
    print("=" * 70)
    
    results = {}
    for path, name in datasets:
        print(f"\n### {name} ({path})")
        stats = analyze_dataset(path)
        if stats is None:
            continue
        
        results[path] = stats
        print(f"  Documents: {stats['num_docs']:,}")
        print(f"  Length: min={stats['min']}, max={stats['max']}, mean={stats['mean']:.1f}, median={stats['median']:.1f}")
        
        for seq_len in seq_lengths_to_check:
            usable = (stats['lengths'] > seq_len).sum()
            pct = usable / stats['num_docs'] * 100
            print(f"  Usable at n_seq={seq_len}: {usable:,} ({pct:.1f}%)")
    
    # Summary diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)
    
    for path, name in datasets:
        if path not in results:
            continue
        stats = results[path]
        # Current n_seq in config is 512
        usable_512 = (stats['lengths'] > 512).sum()
        pct_512 = usable_512 / stats['num_docs'] * 100
        
        if pct_512 < 50:
            print(f"⚠️  {name}: Only {pct_512:.1f}% usable at n_seq=512 - NEEDS CONCATENATION")
        else:
            print(f"✅ {name}: {pct_512:.1f}% usable at n_seq=512")
    
    # Check for Human/Assistant patterns in tokenized data
    print("\n" + "=" * 70)
    print("CONVERSATION PATTERN CHECK")
    print("=" * 70)
    
    for path, name in datasets:
        if "instruct" in path or "dolly" in path:
            bin_path = Path(path) / "train.bin"
            if bin_path.exists():
                tokens = np.memmap(bin_path, dtype=np.uint32, mode="r")
                # Check first 10000 tokens for patterns
                sample = tokens[:min(10000, len(tokens))]
                print(f"\n{name}: First 20 unique token IDs: {np.unique(sample)[:20].tolist()}")

if __name__ == "__main__":
    main()
