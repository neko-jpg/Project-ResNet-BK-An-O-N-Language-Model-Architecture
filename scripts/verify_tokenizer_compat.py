#!/usr/bin/env python3
"""
データセットのトークナイザー整合性検証スクリプト

rinnaトークナイザー（vocab_size=32768）で生成されたデータかどうかを確認
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import struct

print("=" * 70)
print("データセット トークナイザー整合性検証")
print("=" * 70)

# rinnaトークナイザーのvocab_size
RINNA_VOCAB_SIZE = 32768

datasets = ["japanese_instruct", "dolly_ja", "wiki_ja", "mc4_ja"]

results = {}

for ds_name in datasets:
    bin_path = Path(f"data/{ds_name}/train.bin")
    idx_path = Path(f"data/{ds_name}/train.idx")
    
    if not bin_path.exists():
        print(f"\n❌ {ds_name}: train.bin が見つかりません")
        results[ds_name] = "NOT_FOUND"
        continue
    
    try:
        # トークンを読み込み
        tokens = np.memmap(bin_path, dtype=np.uint32, mode='r')
        
        # 統計を計算
        min_token = int(tokens.min())
        max_token = int(tokens.max())
        unique_count = len(np.unique(tokens[:100000]))  # 先頭10万トークンでユニーク数
        
        print(f"\n📊 {ds_name}:")
        print(f"   総トークン数: {len(tokens):,}")
        print(f"   トークン値範囲: {min_token} - {max_token}")
        print(f"   ユニークトークン数 (先頭100K): {unique_count:,}")
        
        # rinnaとの整合性チェック
        if max_token >= RINNA_VOCAB_SIZE:
            print(f"   ⚠️  警告: max_token ({max_token}) >= rinna vocab_size ({RINNA_VOCAB_SIZE})")
            print(f"   → このデータセットはrinnaトークナイザーで生成されていない可能性があります")
            results[ds_name] = "MISMATCH"
        elif max_token < 1000:
            print(f"   ⚠️  警告: max_token ({max_token}) が異常に小さい")
            results[ds_name] = "SUSPICIOUS"
        else:
            print(f"   ✅ rinnaトークナイザー (vocab_size={RINNA_VOCAB_SIZE}) と整合性あり")
            results[ds_name] = "OK"
            
    except Exception as e:
        print(f"\n❌ {ds_name}: エラー - {e}")
        results[ds_name] = "ERROR"

# サマリー
print("\n" + "=" * 70)
print("検証結果サマリー")
print("=" * 70)

ok_count = sum(1 for v in results.values() if v == "OK")
total = len(results)

for ds_name, status in results.items():
    icon = "✅" if status == "OK" else "❌" if status in ["MISMATCH", "NOT_FOUND", "ERROR"] else "⚠️"
    print(f"{icon} {ds_name}: {status}")

print(f"\n合計: {ok_count}/{total} データセットがrinnaトークナイザーと整合")

if ok_count == total:
    print("\n🎉 すべてのデータセットがrinnaトークナイザーで生成されています！")
    sys.exit(0)
else:
    print("\n⚠️  一部のデータセットで問題があります。再生成を検討してください。")
    sys.exit(1)
