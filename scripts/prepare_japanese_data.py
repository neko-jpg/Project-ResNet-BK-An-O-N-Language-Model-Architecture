#!/usr/bin/env python3
"""
Japanese Dataset Preparation Script
====================================
Downloads and prepares Japanese datasets for LLM training.

Datasets:
- Wikipedia Japanese
- CC-100 Japanese
- Oscar Japanese
- Dolly Japanese (Instruction)
- Alpaca Japanese (Instruction)
"""

import os
import sys
import argparse
from pathlib import Path

try:
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoTokenizer
    from tqdm import tqdm
except ImportError:
    print("Installing required packages...")
    os.system(f"{sys.executable} -m pip install datasets transformers tqdm")
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoTokenizer
    from tqdm import tqdm


def download_wikipedia_ja(max_samples: int = 500000):
    """Download Japanese Wikipedia."""
    print("📚 Downloading Wikipedia Japanese...")
    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)
        samples = []
        for i, item in enumerate(tqdm(ds, total=max_samples, desc="Wikipedia JP")):
            if i >= max_samples:
                break
            samples.append({"text": item["text"], "source": "wikipedia_ja"})
        return samples
    except Exception as e:
        print(f"⚠️ Wikipedia JP failed: {e}")
        return []


def download_cc100_ja(max_samples: int = 500000):
    """Download CC-100 Japanese."""
    print("🌐 Downloading CC-100 Japanese...")
    try:
        ds = load_dataset("cc100", "ja", split="train", streaming=True)
        samples = []
        for i, item in enumerate(tqdm(ds, total=max_samples, desc="CC-100 JP")):
            if i >= max_samples:
                break
            samples.append({"text": item["text"], "source": "cc100_ja"})
        return samples
    except Exception as e:
        print(f"⚠️ CC-100 JP failed: {e}")
        return []


def download_dolly_ja(max_samples: int = 15000):
    """Download Dolly Japanese (Instruction Tuning)."""
    print("💬 Downloading Dolly Japanese...")
    try:
        ds = load_dataset("kunishou/databricks-dolly-15k-ja", split="train")
        samples = []
        for i, item in enumerate(tqdm(ds, total=min(len(ds), max_samples), desc="Dolly JP")):
            if i >= max_samples:
                break
            # Format as instruction-response pair
            text = f"### 指示:\n{item['instruction']}\n\n### 回答:\n{item['output']}"
            samples.append({"text": text, "source": "dolly_ja"})
        return samples
    except Exception as e:
        print(f"⚠️ Dolly JP failed: {e}")
        return []


def download_alpaca_ja(max_samples: int = 50000):
    """Download Japanese Alpaca (Instruction Tuning)."""
    print("🦙 Downloading Japanese Alpaca...")
    try:
        ds = load_dataset("fujiki/japanese_alpaca_data", split="train")
        samples = []
        for i, item in enumerate(tqdm(ds, total=min(len(ds), max_samples), desc="Alpaca JP")):
            if i >= max_samples:
                break
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "")
            
            if input_text:
                text = f"### 指示:\n{instruction}\n\n### 入力:\n{input_text}\n\n### 回答:\n{output}"
            else:
                text = f"### 指示:\n{instruction}\n\n### 回答:\n{output}"
            samples.append({"text": text, "source": "alpaca_ja"})
        return samples
    except Exception as e:
        print(f"⚠️ Alpaca JP failed: {e}")
        return []


def save_dataset(samples: list, output_dir: Path, name: str):
    """Save dataset to disk."""
    output_path = output_dir / name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as JSONL
    import json
    jsonl_path = output_path / "data.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"✅ Saved {len(samples)} samples to {jsonl_path}")
    return jsonl_path


def main():
    parser = argparse.ArgumentParser(description="Prepare Japanese datasets for LLM training")
    parser.add_argument("--output-dir", type=str, default="data/japanese", help="Output directory")
    parser.add_argument("--max-pretrain", type=int, default=500000, help="Max samples for pre-training")
    parser.add_argument("--max-instruct", type=int, default=50000, help="Max samples for instruction tuning")
    parser.add_argument("--skip-pretrain", action="store_true", help="Skip pre-training data")
    parser.add_argument("--skip-instruct", action="store_true", help="Skip instruction data")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("🇯🇵 Japanese Dataset Preparation")
    print("=" * 50)
    
    all_pretrain = []
    all_instruct = []
    
    if not args.skip_pretrain:
        # Pre-training data
        wiki_samples = download_wikipedia_ja(args.max_pretrain)
        if wiki_samples:
            save_dataset(wiki_samples, output_dir, "wikipedia_ja")
            all_pretrain.extend(wiki_samples)
        
        cc100_samples = download_cc100_ja(args.max_pretrain)
        if cc100_samples:
            save_dataset(cc100_samples, output_dir, "cc100_ja")
            all_pretrain.extend(cc100_samples)
    
    if not args.skip_instruct:
        # Instruction tuning data
        dolly_samples = download_dolly_ja(15000)
        if dolly_samples:
            save_dataset(dolly_samples, output_dir, "dolly_ja")
            all_instruct.extend(dolly_samples)
        
        alpaca_samples = download_alpaca_ja(args.max_instruct)
        if alpaca_samples:
            save_dataset(alpaca_samples, output_dir, "alpaca_ja")
            all_instruct.extend(alpaca_samples)
    
    # Save combined datasets
    if all_pretrain:
        save_dataset(all_pretrain, output_dir, "pretrain_combined")
    if all_instruct:
        save_dataset(all_instruct, output_dir, "instruct_combined")
    
    print("\n" + "=" * 50)
    print("📊 Summary:")
    print(f"  Pre-training samples: {len(all_pretrain):,}")
    print(f"  Instruction samples:  {len(all_instruct):,}")
    print(f"  Total samples:        {len(all_pretrain) + len(all_instruct):,}")
    print(f"  Output directory:     {output_dir.absolute()}")
    print("=" * 50)
    
    # Download tokenizer
    print("\n📝 Downloading Japanese tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b")
        tokenizer.save_pretrained(output_dir / "tokenizer")
        print(f"✅ Tokenizer saved to {output_dir / 'tokenizer'}")
    except Exception as e:
        print(f"⚠️ Tokenizer download failed: {e}")
    
    print("\n🎉 Done! Ready for training.")


if __name__ == "__main__":
    main()
