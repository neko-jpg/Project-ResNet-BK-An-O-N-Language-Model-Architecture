#!/usr/bin/env python3
import numpy as np
from pathlib import Path

print('=== 全データのmax token確認 ===')
for subdir in sorted(Path('data').iterdir()):
    if not subdir.is_dir():
        continue
    bin_path = subdir / 'train.bin'
    if bin_path.exists():
        tokens = np.memmap(bin_path, dtype=np.uint32, mode='r')
        max_tok = int(tokens.max())
        if max_tok < 32000:
            status = '✅ 32000互換'
        elif max_tok < 32768:
            status = '✅ 32768互換'
        else:
            status = '❌ 対象外'
        print(f'{subdir.name:<45} max={max_tok:<8} {status}')
