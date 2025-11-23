#!/usr/bin/env python3
"""
MUSE User Data Importer
Imports text/json/csv files from data/import/ directory.
"""
import sys
import os
import json
import logging
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.progress import track

# Add root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from scripts.prepare_datasets import BinaryWriter
    from transformers import AutoTokenizer
except ImportError:
    print("Error: modules not found. Run from repo root.")
    sys.exit(1)

# Language Config
LANG = "1"
if os.path.exists(".muse_config"):
    with open(".muse_config") as f:
        for line in f:
            if "MUSE_LANG" in line:
                LANG = line.strip().split("=")[1].strip("'\"")
IS_JP = (LANG == "2")

def t(en, jp): return jp if IS_JP else en

console = Console()
logger = logging.getLogger(__name__)

def main():
    console.print(t("[bold blue]MUSE Data Importer[/bold blue]", "[bold blue]MUSE データインポーター[/bold blue]"))

    import_dir = Path("data/import")
    output_dir = Path("data/user_imported")
    import_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Scan Files
    files = list(import_dir.glob("*"))
    valid_extensions = {'.txt', '.json', '.csv', '.parquet', '.jsonl'}
    target_files = [f for f in files if f.suffix.lower() in valid_extensions]

    if not target_files:
        console.print(t(f"No files found in {import_dir}.", f"{import_dir} にファイルが見つかりません。"))
        console.print(t("Please drop your .txt, .json, .csv files there.", "学習させたい .txt, .json, .csv ファイルをそこに置いてください。"))
        return

    console.print(t(f"Found {len(target_files)} files.", f"{len(target_files)} 個のファイルを検出しました。"))
    for f in target_files:
        console.print(f" - {f.name}")

    if input(t("Start import? [Y/n]: ", "インポートを開始しますか？ [Y/n]: ")).lower() == 'n':
        console.print("Aborted.")
        return

    # 2. Setup Tokenizer & Writer
    tokenizer_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    writer = BinaryWriter(output_dir / "train")

    total_samples = 0

    # 3. Process
    for file_path in target_files:
        console.print(t(f"Processing {file_path.name}...", f"{file_path.name} を処理中..."))
        texts = []

        try:
            if file_path.suffix == '.txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    texts = [f.read()]
            elif file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        texts = [str(x) for x in data]
                    else:
                        texts = [str(data)]
            elif file_path.suffix == '.jsonl':
                 with open(file_path, 'r', encoding='utf-8') as f:
                    texts = [line for line in f]
            elif file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
                # Try to find a text column or join all
                text_col = next((c for c in df.columns if 'text' in c.lower() or 'content' in c.lower()), None)
                if text_col:
                    texts = df[text_col].dropna().astype(str).tolist()
                else:
                    texts = df.astype(str).apply(lambda x: ' '.join(x), axis=1).tolist()
            elif file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
                text_col = next((c for c in df.columns if 'text' in c.lower()), None)
                if text_col:
                    texts = df[text_col].dropna().astype(str).tolist()
                else:
                     texts = df.astype(str).apply(lambda x: ' '.join(x), axis=1).tolist()

            # Tokenize and Write
            batch_tokens = []
            for text in track(texts, description="Tokenizing..."):
                if text.strip():
                    ids = tokenizer(text, truncation=False, padding=False)['input_ids']
                    batch_tokens.append(ids)
                    total_samples += 1

            if batch_tokens:
                writer.append(batch_tokens)

        except Exception as e:
            console.print(f"[red]Error processing {file_path.name}: {e}[/red]")

    output_path = writer.close()

    # Save Metadata
    meta = {
        'dataset': 'user_imported',
        'files': [f.name for f in target_files],
        'doc_count': total_samples,
        'path': output_path
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    console.print(t(f"\n[bold green]Import Complete![/bold green] {total_samples} samples imported.", f"\n[bold green]インポート完了！[/bold green] {total_samples} 件のデータを変換しました。"))
    console.print(t("You can now configure the mixing recipe via 'make recipe'.", "'make recipe' コマンドで学習データの配合を設定できます。"))

if __name__ == "__main__":
    main()
