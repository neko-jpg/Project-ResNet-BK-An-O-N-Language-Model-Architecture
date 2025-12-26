#!/usr/bin/env python3
"""Detailed dataset analysis with full output."""

import numpy as np
from pathlib import Path
import struct
import sys

datasets = ['japanese_instruct', 'dolly_ja', 'wiki_ja', 'mc4_ja']

for ds in datasets:
    idx_path = Path(f'data/{ds}/train.idx')
    if idx_path.exists():
        with open(idx_path, 'rb') as f:
            magic = f.read(4)
            ver = struct.unpack('<I', f.read(4))[0]
            data = np.fromfile(f, dtype=np.uint64)
        
        if data.size >= 2 and data.size % 2 == 0:
            lengths = data.reshape(-1, 2)[:, 1]
            num_docs = len(lengths)
            
            print(f"DATASET: {ds}")
            print(f"  docs={num_docs}, min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.1f}")
            
            for n_seq in [256, 512, 1024, 2048]:
                usable = (lengths > n_seq).sum()
                pct = usable / num_docs * 100
                status = "OK" if pct > 50 else "NEEDS_CONCAT"
                print(f"  n_seq={n_seq}: {usable}/{num_docs} ({pct:.1f}%) [{status}]")
            print()
