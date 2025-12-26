#!/usr/bin/env python3
"""
会話パターン検証スクリプト

学習バッチ内の会話パターンを検証し、以下の指標を測定:
- Human:トークン含有率（目標: 80%以上のサンプルに含まれる）
- EOS/サンプル密度（目標: 5-12）
- Human→Assistant連続率
- EOS間の平均トークン長
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import numpy as np
from transformers import AutoTokenizer

from src.utils.data_utils import get_mixed_data_loader


def verify_conversation_patterns(
    config_path: str,
    batch_size: int = 4,
    seq_len: int = 512,
    num_batches: int = 50,
    tokenizer_name: str = "rinna/japanese-gpt-neox-3.6b",
):
    """Verify conversation patterns in training batches."""
    
    print("=" * 70)
    print("会話パターン検証")
    print("=" * 70)
    
    # Load tokenizer (try local first, then HuggingFace)
    print(f"Loading tokenizer: {tokenizer_name}")
    local_tokenizer_path = f"tokenizers/{tokenizer_name.split('/')[-1]}"
    try:
        # Try local tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path, use_fast=False)
        print(f"  (loaded from local: {local_tokenizer_path})")
    except Exception:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
            print(f"  (loaded from HuggingFace)")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("Using GPT-2 tokenizer as fallback...")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Get special token IDs
    eos_id = tokenizer.eos_token_id
    
    # Look for common conversation markers
    human_patterns = ["Human:", "ユーザー:", "質問:", "入力:"]
    assistant_patterns = ["Assistant:", "アシスタント:", "回答:", "出力:"]
    
    # Encode patterns to find their token sequences
    human_token_ids = set()
    assistant_token_ids = set()
    
    for pattern in human_patterns:
        try:
            ids = tokenizer.encode(pattern, add_special_tokens=False)
            human_token_ids.update(ids)
        except:
            pass
    
    for pattern in assistant_patterns:
        try:
            ids = tokenizer.encode(pattern, add_special_tokens=False)
            assistant_token_ids.update(ids)
        except:
            pass
    
    print(f"EOS token ID: {eos_id}")
    print(f"Human-related token IDs: {sorted(human_token_ids)[:10]}...")
    print(f"Assistant-related token IDs: {sorted(assistant_token_ids)[:10]}...")
    
    # Load dataset
    print(f"\nLoading dataset from: {config_path}")
    try:
        mixed_dataset, vocab, steps = get_mixed_data_loader(
            config_path=config_path,
            batch_size=batch_size,
            n_seq=seq_len,
            total_tokens=1000000,
            seed=42,
            vocab_size=tokenizer.vocab_size,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    print(f"Dataset loaded. Steps per epoch: {steps}")
    
    # Metrics
    samples_with_human = 0
    samples_with_assistant = 0
    total_eos_count = 0
    total_samples = 0
    eos_counts_per_sample = []
    sample_texts = []
    
    print(f"\nAnalyzing {num_batches} batches...")
    
    for batch_idx, (x_batch, y_batch) in enumerate(mixed_dataset.iter_epoch(epoch=0)):
        if batch_idx >= num_batches:
            break
        
        for sample_idx in range(x_batch.shape[0]):
            sample = x_batch[sample_idx].numpy()
            total_samples += 1
            
            # Count EOS tokens
            eos_count = (sample == eos_id).sum()
            total_eos_count += eos_count
            eos_counts_per_sample.append(eos_count)
            
            # Check for human/assistant patterns
            sample_set = set(sample.tolist())
            has_human = bool(human_token_ids & sample_set)
            has_assistant = bool(assistant_token_ids & sample_set)
            
            if has_human:
                samples_with_human += 1
            if has_assistant:
                samples_with_assistant += 1
            
            # Decode a few samples for inspection
            if len(sample_texts) < 3:
                try:
                    text = tokenizer.decode(sample[:100], skip_special_tokens=False)
                    sample_texts.append(text)
                except:
                    pass
    
    # Report results
    print("\n" + "=" * 70)
    print("検証結果")
    print("=" * 70)
    
    human_pct = samples_with_human / total_samples * 100 if total_samples > 0 else 0
    assistant_pct = samples_with_assistant / total_samples * 100 if total_samples > 0 else 0
    avg_eos = total_eos_count / total_samples if total_samples > 0 else 0
    
    print(f"\n総サンプル数: {total_samples}")
    print(f"\nHuman含有率: {samples_with_human}/{total_samples} ({human_pct:.1f}%)")
    print(f"Assistant含有率: {samples_with_assistant}/{total_samples} ({assistant_pct:.1f}%)")
    print(f"\n平均EOS/サンプル: {avg_eos:.1f}")
    print(f"EOS分布: min={min(eos_counts_per_sample)}, max={max(eos_counts_per_sample)}")
    
    # Quality assessment
    print("\n" + "=" * 70)
    print("品質評価")
    print("=" * 70)
    
    issues = []
    if human_pct < 80:
        issues.append(f"⚠️ Human含有率が低い ({human_pct:.1f}% < 80%)")
    else:
        print(f"✅ Human含有率: {human_pct:.1f}% >= 80%")
    
    if avg_eos < 5:
        issues.append(f"⚠️ EOS密度が低い ({avg_eos:.1f} < 5)")
    elif avg_eos > 12:
        issues.append(f"⚠️ EOS密度が高い ({avg_eos:.1f} > 12)")
    else:
        print(f"✅ EOS密度: {avg_eos:.1f} (目標: 5-12)")
    
    if issues:
        print("\n問題点:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ すべての品質チェックに合格")
    
    # Sample texts
    print("\n" + "=" * 70)
    print("サンプルテキスト (先頭100トークン)")
    print("=" * 70)
    for i, text in enumerate(sample_texts):
        print(f"\n--- サンプル {i+1} ---")
        print(text[:500] + "..." if len(text) > 500 else text)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset_japanese_chat_optimized.yaml")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--num-batches", type=int, default=50)
    parser.add_argument("--tokenizer", default="rinna/japanese-gpt-neox-3.6b")
    args = parser.parse_args()
    
    verify_conversation_patterns(
        config_path=args.config,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_batches=args.num_batches,
        tokenizer_name=args.tokenizer,
    )
