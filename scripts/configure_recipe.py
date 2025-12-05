#!/usr/bin/env python3
"""
MUSE Concierge - Training Wizard (Phase 8 対応版)
===================================================
データセット配合とハードウェア最適化を自動設定します。

Usage:
    make recipe
    python scripts/configure_recipe.py
"""
import os
import sys
import yaml
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, IntPrompt, Confirm
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for better UI: pip install rich")

console = Console() if RICH_AVAILABLE else None

# Language detection
IS_JP = os.getenv("MUSE_LANG", "1") == "2"
if os.path.exists(".muse_config"):
    with open(".muse_config") as f:
        for line in f:
            if "MUSE_LANG" in line:
                IS_JP = "2" in line

def t(en, jp):
    return jp if IS_JP else en


@dataclass
class Phase8Config:
    """Phase 8 学習設定"""
    # Model Architecture
    d_model: int = 4096
    n_layers: int = 48
    n_seq: int = 512
    num_heads: int = 32
    vocab_size: int = 32000  # Japanese tokenizer
    
    # Compression
    low_rank_ffn: bool = True
    low_rank_attention: bool = True
    low_rank_rank: int = 16
    use_bitnet: bool = True
    
    # Training
    batch_size: int = 1
    gradient_accumulation_steps: int = 32
    epochs: int = 1
    learning_rate: float = 0.02
    warmup_steps: int = 2000
    
    # Memory Optimization
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = "bfloat16"
    
    # Triton / Compile
    use_triton_kernel: bool = True
    use_torch_compile: bool = True
    compile_mode: str = "max-autotune"
    
    # Language
    language: str = "japanese"  # japanese or english
    tokenizer_name: str = "rinna/japanese-gpt-neox-3.6b"
    
    # Paths
    save_dir: str = "checkpoints/phase8_10b_japanese"


def detect_gpu():
    """GPU情報を検出"""
    try:
        import torch
        if not torch.cuda.is_available():
            return None, 0
        
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return name, vram
    except:
        return None, 0


def estimate_vram(config: Phase8Config) -> float:
    """VRAM使用量を推定（MB）"""
    # 簡易推定式
    # Base: Embedding + Position
    embed_mem = config.vocab_size * config.d_model * 2 / (1024**2)  # FP16
    
    # Per layer (with compression)
    if config.low_rank_ffn and config.low_rank_attention:
        # Low-rank: ~5% of full
        layer_mem = (config.d_model * config.low_rank_rank * 8) * 2 / (1024**2)
    else:
        layer_mem = (config.d_model ** 2 * 12) * 2 / (1024**2)
    
    total_layers_mem = layer_mem * config.n_layers
    
    # Activations (with gradient checkpointing)
    if config.use_gradient_checkpointing:
        activation_mem = config.batch_size * config.n_seq * config.d_model * 4 / (1024**2)
    else:
        activation_mem = config.batch_size * config.n_seq * config.d_model * config.n_layers * 4 / (1024**2)
    
    # Optimizer states (Muon is lighter than Adam)
    optimizer_mem = (embed_mem + total_layers_mem) * 1.5
    
    total = embed_mem + total_layers_mem + activation_mem + optimizer_mem
    
    # Safety margin
    return total * 1.2


def calculate_dense_params(config: Phase8Config) -> int:
    """Dense換算パラメータ数を計算"""
    # Embedding
    embed_params = config.vocab_size * config.d_model + config.n_seq * config.d_model
    
    # Per layer (Dense)
    attention_params = 4 * config.d_model ** 2  # Q, K, V, O
    ffn_params = 2 * config.d_model * (config.d_model * 4)  # up + down
    layer_params = attention_params + ffn_params
    
    # Total
    total_layers = layer_params * config.n_layers
    lm_head = 0  # Tied with embedding
    
    return embed_params + total_layers + lm_head


def calculate_actual_params(config: Phase8Config) -> int:
    """実際のパラメータ数を計算"""
    # Embedding (not compressed)
    embed_params = config.vocab_size * config.d_model + config.n_seq * config.d_model
    
    # Per layer (with Low-Rank)
    if config.low_rank_ffn and config.low_rank_attention:
        # Low-rank: 2 * (d_model * rank) per projection
        attention_params = 4 * 2 * config.d_model * config.low_rank_rank
        ffn_params = 2 * 2 * config.d_model * config.low_rank_rank
    else:
        attention_params = 4 * config.d_model ** 2
        ffn_params = 2 * config.d_model * (config.d_model * 4)
    
    layer_params = attention_params + ffn_params
    total_layers = layer_params * config.n_layers
    
    return embed_params + total_layers


def scan_datasets() -> Dict[str, Dict]:
    """利用可能なデータセットをスキャン"""
    data_dir = Path("data")
    datasets = {}
    
    # Japanese datasets
    jp_dirs = ["japanese", "wiki_ja", "wikipedia_ja", "cc100_ja", "dolly_ja", "alpaca_ja"]
    # English datasets
    en_dirs = ["cosmopedia", "evol_instruct_code", "openwebtext"]
    
    for name in jp_dirs + en_dirs:
        path = data_dir / name
        if path.exists():
            # Check for data files
            has_data = any(path.glob("*.jsonl")) or any(path.glob("*.json")) or (path / "metadata.json").exists()
            if has_data:
                is_japanese = name in jp_dirs or "ja" in name.lower()
                datasets[name] = {
                    "path": str(path),
                    "language": "japanese" if is_japanese else "english",
                    "type": "instruction" if any(x in name for x in ["dolly", "alpaca", "instruct"]) else "pretrain"
                }
    
    return datasets


def main():
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            t("🧙 MUSE Concierge - Training Wizard (Phase 8)", "🧙 MUSE コンシェルジュ - 学習設定ウィザード (Phase 8)"),
            subtitle="10B Japanese/English LLM",
            style="bold blue"
        ))
    else:
        print("=" * 60)
        print("MUSE Concierge - Training Wizard (Phase 8)")
        print("=" * 60)
    
    # 1. GPU Detection
    gpu_name, vram_gb = detect_gpu()
    if gpu_name:
        print(f"\n🖥️  GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        print("\n⚠️  No GPU detected. Training will be slow on CPU.")
    
    # 2. Language Selection
    print(t("\n🌐 Select model language:", "\n🌐 モデルの言語を選択:"))
    print("  1. 🇯🇵 Japanese (日本語)")
    print("  2. 🇺🇸 English")
    print("  3. 🌍 Multilingual")
    
    lang_choice = input("Choice [1]: ").strip() or "1"
    
    config = Phase8Config()
    
    if lang_choice == "1":
        config.language = "japanese"
        config.tokenizer_name = "rinna/japanese-gpt-neox-3.6b"
        config.vocab_size = 32000
        config.save_dir = "checkpoints/phase8_10b_japanese"
    elif lang_choice == "2":
        config.language = "english"
        config.tokenizer_name = "gpt2"
        config.vocab_size = 50257
        config.save_dir = "checkpoints/phase8_10b_english"
    else:
        config.language = "multilingual"
        config.tokenizer_name = "xlm-roberta-base"
        config.vocab_size = 250002
        config.save_dir = "checkpoints/phase8_10b_multi"
    
    # 3. Training Goal
    print(t("\n🎯 Select training goal:", "\n🎯 学習の目的を選択:"))
    print(t("  1. 🔍 Debug (Quick test)", "  1. 🔍 デバッグ (動作確認)"))
    print(t("  2. 🚀 Production (Full training)", "  2. 🚀 本番学習 (フル学習)"))
    print(t("  3. ⚡ Benchmark (Max performance)", "  3. ⚡ ベンチマーク (限界に挑戦)"))
    
    goal_choice = input("Choice [2]: ").strip() or "2"
    
    if goal_choice == "1":
        config.epochs = 1
        config.n_layers = 12
        config.d_model = 1024
    elif goal_choice == "3":
        config.epochs = 3
        config.gradient_accumulation_steps = 64
    
    # 4. Hardware Optimization
    if vram_gb > 0:
        print(f"\n⚙️  {t('Hardware optimization:', 'ハードウェア最適化:')}")
        
        if vram_gb <= 8:
            print(t("  Detected: Low VRAM (≤8GB) - Using extreme compression", 
                   "  検出: 低VRAM (≤8GB) - 極限圧縮モード"))
            config.batch_size = 1
            config.gradient_accumulation_steps = 32
            config.use_gradient_checkpointing = True
        elif vram_gb <= 12:
            print(t("  Detected: Medium VRAM (8-12GB)", 
                   "  検出: 中VRAM (8-12GB)"))
            config.batch_size = 2
            config.gradient_accumulation_steps = 16
        else:
            print(t("  Detected: High VRAM (>12GB)", 
                   "  検出: 高VRAM (>12GB)"))
            config.batch_size = 4
            config.gradient_accumulation_steps = 8
    
    # 5. Dataset Selection
    print(t("\n📚 Scanning datasets...", "\n📚 データセットをスキャン中..."))
    datasets = scan_datasets()
    
    if datasets:
        print(t(f"  Found {len(datasets)} dataset(s):", f"  {len(datasets)}個のデータセットを発見:"))
        for name, info in datasets.items():
            lang_emoji = "🇯🇵" if info["language"] == "japanese" else "🇺🇸"
            print(f"    {lang_emoji} {name} ({info['type']})")
    else:
        print(t("  No datasets found. Run 'make prepare-japanese-data' first.",
               "  データセットがありません。'make prepare-japanese-data' を実行してください。"))
    
    # 6. Show Configuration Summary
    dense_params = calculate_dense_params(config)
    actual_params = calculate_actual_params(config)
    est_vram = estimate_vram(config)
    compression = (1 - actual_params / dense_params) * 100 if dense_params > 0 else 0
    
    print(t("\n📊 Configuration Summary:", "\n📊 設定サマリー:"))
    print("-" * 50)
    
    if RICH_AVAILABLE:
        table = Table(show_header=True)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Language", config.language.title())
        table.add_row("d_model", str(config.d_model))
        table.add_row("n_layers", str(config.n_layers))
        table.add_row("Dense Params", f"{dense_params / 1e9:.2f}B")
        table.add_row("Actual Params", f"{actual_params / 1e6:.1f}M")
        table.add_row("Compression", f"{compression:.1f}%")
        table.add_row("Est. VRAM", f"{est_vram:.0f} MB")
        table.add_row("Batch Size", f"{config.batch_size} (×{config.gradient_accumulation_steps} accum)")
        table.add_row("Save Dir", config.save_dir)
        
        console.print(table)
    else:
        print(f"  Language: {config.language}")
        print(f"  d_model: {config.d_model}")
        print(f"  n_layers: {config.n_layers}")
        print(f"  Dense Params: {dense_params / 1e9:.2f}B")
        print(f"  Actual Params: {actual_params / 1e6:.1f}M")
        print(f"  Compression: {compression:.1f}%")
        print(f"  Est. VRAM: {est_vram:.0f} MB")
    
    # 7. Confirm and Save
    print()
    confirm = input(t("Save this configuration? [Y/n]: ", "この設定を保存しますか？ [Y/n]: ")).strip().lower()
    
    if confirm in ["", "y", "yes"]:
        # Save config
        config_dir = Path("configs")
        config_dir.mkdir(exist_ok=True)
        
        config_dict = asdict(config)
        
        # Save as YAML
        yaml_path = config_dir / "user_train_config.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False)
        
        print(f"\n✅ {t('Configuration saved to:', '設定を保存しました:')} {yaml_path}")
        
        # Show next steps
        print(t("\n🚀 Next steps:", "\n🚀 次のステップ:"))
        if config.language == "japanese":
            print("  1. make prepare-japanese-data  # Download Japanese data")
            print("  2. make start-japanese         # Start training")
        else:
            print("  1. make data-lite              # Download English data")
            print("  2. make start-10b-local        # Start training")
        
        print(t("\nOr run with custom config:", "\nまたはカスタム設定で実行:"))
        print(f"  python scripts/train_phase8.py --config {yaml_path}")
    else:
        print(t("Configuration not saved.", "設定は保存されませんでした。"))


if __name__ == "__main__":
    main()
