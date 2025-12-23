#!/usr/bin/env python3
"""
日本語会話データセット - GitHubからダウンロード
==============================================
HuggingFace APIを使わず、GitHubから直接クローンして処理します。

Usage:
    python scripts/download_japanese_chat_data.py
"""

import os
import sys
import json
import struct
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

def clone_or_update_repo(url: str, target_dir: Path) -> bool:
    """Clone or update a git repository."""
    if target_dir.exists():
        print(f"   Repository already exists, pulling updates...")
        result = subprocess.run(
            ["git", "-C", str(target_dir), "pull"],
            capture_output=True, text=True
        )
        return result.returncode == 0
    else:
        print(f"   Cloning repository...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, str(target_dir)],
            capture_output=True, text=True
        )
        return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Download Japanese Chat Datasets from GitHub")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--tokenizer", type=str, default="rinna/japanese-gpt-neox-3.6b", help="Tokenizer name")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per dataset")
    parser.add_argument("--temp-dir", type=str, default="/tmp/japanese_datasets", help="Temp dir for cloning")
    args = parser.parse_args()
    
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("❌ Please install: pip install transformers")
        sys.exit(1)
    
    print("=" * 60)
    print("🇯🇵 Japanese Chat Dataset Downloader (GitHub)")
    print("=" * 60)
    
    # Load tokenizer (try local cache first)
    print(f"\n📦 Loading tokenizer: {args.tokenizer}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        print("   (loaded from local cache)")
    except:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        except Exception as e:
            print(f"   ⚠️ Could not load {args.tokenizer}: {e}")
            print("   Falling back to GPT-2 tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    
    # Define GitHub datasets
    datasets_config = [
        {
            "name": "dolly_ja",
            "git_url": "https://github.com/kunishou/databricks-dolly-15k-ja.git",
            "data_file": "databricks-dolly-15k-ja.json",
            "format": "instruction",
            "instruction_field": "instruction",
            "input_field": "input",
            "output_field": "output",
            "description": "Dolly日本語 (15K 指示追従データ)"
        },
        {
            "name": "jglue",
            "git_url": "https://github.com/yahoojapan/JGLUE.git",
            "data_dir": "datasets/jcommonsenseqa-v1.1",
            "data_file": "train-v1.1.json",
            "format": "jglue_qa",
            "description": "JGLUE JCommonsenseQA (日本語常識QA)"
        },
    ]
    
    for ds_config in datasets_config:
        name = ds_config["name"]
        git_url = ds_config["git_url"]
        desc = ds_config["description"]
        
        print(f"\n{'='*60}")
        print(f"📥 Downloading: {desc}")
        print(f"   Source: {git_url}")
        print("=" * 60)
        
        try:
            # Clone repository
            repo_dir = temp_dir / name
            if not clone_or_update_repo(git_url, repo_dir):
                print(f"   ⚠️ Failed to clone repository")
                continue
            
            # Find data file
            if "data_dir" in ds_config:
                data_path = repo_dir / ds_config["data_dir"] / ds_config["data_file"]
            else:
                data_path = repo_dir / ds_config["data_file"]
            
            if not data_path.exists():
                # Try to find the file
                print(f"   Looking for data files in {repo_dir}...")
                json_files = list(repo_dir.rglob("*.json"))
                print(f"   Found JSON files: {[f.name for f in json_files[:5]]}")
                if json_files:
                    data_path = json_files[0]
                    print(f"   Using: {data_path}")
                else:
                    print(f"   ⚠️ No JSON files found")
                    continue
            
            # Load data
            print(f"   Loading: {data_path.name}")
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and 'data' in data:
                data = data['data']
            
            if args.max_samples and len(data) > args.max_samples:
                data = data[:args.max_samples]
            
            print(f"   Samples: {len(data)}")
            
            # Convert to text format
            texts = []
            for item in data:
                if ds_config["format"] == "instruction":
                    instruction = item.get(ds_config["instruction_field"], "")
                    input_text = item.get(ds_config.get("input_field", "input"), "")
                    output_text = item.get(ds_config["output_field"], "")
                    
                    if input_text:
                        text = f"### 指示:\n{instruction}\n\n### 入力:\n{input_text}\n\n### 回答:\n{output_text}"
                    else:
                        text = f"### 指示:\n{instruction}\n\n### 回答:\n{output_text}"
                    texts.append(text)
                    
                elif ds_config["format"] == "jglue_qa":
                    # JGLUE JCommonsenseQA format
                    question = item.get("question", "")
                    choices = [item.get(f"choice{i}", "") for i in range(5)]
                    label = item.get("label", 0)
                    answer = choices[label] if label < len(choices) else ""
                    
                    choices_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices) if c])
                    text = f"### 質問:\n{question}\n\n選択肢:\n{choices_text}\n\n### 正解:\n{answer}"
                    texts.append(text)
            
            print(f"   Processed: {len(texts)} texts")
            
            if len(texts) == 0:
                print(f"   ⚠️ No texts extracted")
                continue
            
            # Tokenize all texts
            print("   Tokenizing...")
            all_tokens = []
            for i, text in enumerate(texts):
                tokens = tokenizer.encode(text, add_special_tokens=True)
                all_tokens.extend(tokens)
                all_tokens.append(tokenizer.eos_token_id)
                
                if (i + 1) % 5000 == 0:
                    print(f"      {i+1}/{len(texts)} texts processed")
            
            print(f"   Total tokens: {len(all_tokens):,}")
            
            # Save as .bin/.idx format
            ds_output_dir = output_dir / name
            ds_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Split into train/val (95/5)
            split_point = int(len(all_tokens) * 0.95)
            train_tokens = all_tokens[:split_point]
            val_tokens = all_tokens[split_point:]
            
            save_binary_dataset(ds_output_dir / "train.bin", ds_output_dir / "train.idx", train_tokens)
            save_binary_dataset(ds_output_dir / "validation.bin", ds_output_dir / "validation.idx", val_tokens)
            
            # Save metadata
            metadata = {
                "source": git_url,
                "description": desc,
                "num_samples": len(texts),
                "num_tokens": len(all_tokens),
                "tokenizer": args.tokenizer,
            }
            with open(ds_output_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            print(f"   ✅ Saved to {ds_output_dir}")
            
        except Exception as e:
            print(f"   ⚠️ Failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 60)
    print("✅ Download Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Update configs/dataset_japanese_chat_optimized.yaml to include new datasets")
    print("2. Run: make train-japanese-chat")


def save_binary_dataset(bin_path: Path, idx_path: Path, tokens: List[int]):
    """Save tokens as .bin/.idx pair (MUSE format)."""
    import numpy as np
    
    token_array = np.array(tokens, dtype=np.uint32)
    token_array.tofile(bin_path)
    
    with open(idx_path, "wb") as f:
        f.write(b"MUSE")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<Q", 0))
        f.write(struct.pack("<Q", len(tokens)))


if __name__ == "__main__":
    main()
