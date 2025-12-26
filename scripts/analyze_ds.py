#!/usr/bin/env python3
"""Analyze dataset document lengths with robust error handling."""

import numpy as np
from pathlib import Path
import struct

datasets = ['japanese_instruct', 'dolly_ja', 'wiki_ja', 'mc4_ja']

print("=" * 70)
print("Document Length Analysis")
print("=" * 70)

for ds in datasets:
    idx_path = Path(f'data/{ds}/train.idx')
    if idx_path.exists():
        try:
            with open(idx_path, 'rb') as f:
                magic = f.read(4)
                ver = struct.unpack('<I', f.read(4))[0]
                data = np.fromfile(f, dtype=np.uint64)
            
            print(f"\n{ds}:")
            print(f"  magic={magic}, version={ver}")
            print(f"  raw data size: {data.size} uint64 values")
            
            if data.size >= 2 and data.size % 2 == 0:
                lengths = data.reshape(-1, 2)[:, 1]
                num_docs = len(lengths)
                
                # Analysis for different n_seq values
                for n_seq in [256, 512, 1024]:
                    usable = (lengths > n_seq).sum()
                    pct = usable / num_docs * 100
                    print(f"  n_seq={n_seq}: {usable:,}/{num_docs:,} ({pct:.1f}%) usable")
                
                print(f"  Length stats: min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.1f}")
            else:
                print(f"  WARNING: Invalid data shape, size={data.size}")
                
        except Exception as e:
            print(f"  ERROR: {e}")
    else:
        print(f"\n{ds}: NOT FOUND")

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)
