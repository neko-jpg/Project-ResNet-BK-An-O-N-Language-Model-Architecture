#!/usr/bin/env python3
"""
全データディレクトリのトークナイザー整合性を一括検査
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

print("=" * 70)
print("全データディレクトリのトークナイザー整合性検査")
print("=" * 70)

RINNA_VOCAB_SIZE = 32768

# dataフォルダ内のすべてのディレクトリを検査
data_dir = Path("data")
results = []

for subdir in sorted(data_dir.iterdir()):
    if not subdir.is_dir():
        continue
    
    bin_path = subdir / "train.bin"
    if not bin_path.exists():
        continue
    
    try:
        tokens = np.memmap(bin_path, dtype=np.uint32, mode='r')
        min_token = int(tokens.min())
        max_token = int(tokens.max())
        total_tokens = len(tokens)
        
        # 判定
        if max_token >= RINNA_VOCAB_SIZE:
            status = "❌ GPT-2?"
            compatible = False
        elif max_token < 1000:
            status = "⚠️ 異常"
            compatible = False
        else:
            status = "✅ rinna互換"
            compatible = True
        
        results.append({
            "name": subdir.name,
            "tokens": total_tokens,
            "min": min_token,
            "max": max_token,
            "status": status,
            "compatible": compatible
        })
        
    except Exception as e:
        results.append({
            "name": subdir.name,
            "status": f"❌ エラー: {e}",
            "compatible": False
        })

# 結果表示
print(f"\n{'ディレクトリ':<45} {'トークン数':<15} {'min-max':<20} {'状態'}")
print("-" * 100)

compatible_list = []
for r in results:
    if "tokens" in r:
        print(f"{r['name']:<45} {r['tokens']:>12,} {r['min']:>8} - {r['max']:<8} {r['status']}")
    else:
        print(f"{r['name']:<45} {'-':<15} {'-':<20} {r['status']}")
    
    if r.get("compatible"):
        compatible_list.append(r["name"])

print("\n" + "=" * 70)
print("使用可能なデータセット（rinna互換）:")
print("=" * 70)
if compatible_list:
    for name in compatible_list:
        print(f"  ✅ {name}")
else:
    print("  なし - データ再生成が必要です")

print("\n推奨アクション:")
if compatible_list:
    print(f"  使用可能なデータ: {', '.join(compatible_list)}")
else:
    print("  make regenerate-data を実行してください")
