#!/usr/bin/env python3
"""
MUSE Concierge - Training Wizard
Auto-configures training parameters based on hardware calibration and user goals.
"""
import os
import sys
import yaml
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table
from rich.layout import Layout
from rich.live import Live

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from scripts.calibration import MuseCalibrator
except ImportError:
    MuseCalibrator = None

# Language
LANG = "1"
if os.path.exists(".muse_config"):
    with open(".muse_config") as f:
        for line in f:
            if "MUSE_LANG" in line:
                LANG = line.strip().split("=")[1].strip("'\"")
IS_JP = (LANG == "2")

def t(en, jp): return jp if IS_JP else en

console = Console()

def main():
    console.print(Panel.fit(
        t("MUSE Concierge - Training Wizard", "MUSE コンシェルジュ - 学習設定ウィザード"),
        subtitle="Auto-tuning for O(N) Architecture",
        style="bold blue"
    ))

    # 1. Goal Selection
    console.print(t("\nWhat is your goal today?", "\n今日の学習の目的は何ですか？"))
    console.print(t("1. Debug (Quick check)", "1. デバッグ (とりあえず動かす)"))
    console.print(t("2. Benchmark (Push limits)", "2. ベンチマーク (性能の限界に挑戦)"))
    console.print(t("3. Production (Train a good model)", "3. 本番学習 (良いモデルを作る)"))

    goal = IntPrompt.ask("Choice", choices=["1", "2", "3"], default="1")

    # 2. Calibration
    cal = MuseCalibrator()
    if cal and cal.device.type == 'cuda':
        if Confirm.ask(t("Run hardware calibration?", "ハードウェア診断（キャリブレーション）を実行しますか？"), default=True):
            cal.calibrate()
    else:
        console.print(t("[yellow]Skipping calibration (CPU or module missing).[/yellow]", "[yellow]キャリブレーションをスキップします。[/yellow]"))

    # 3. Dataset Recipe
    data_dir = Path("data")
    available_datasets = []
    if data_dir.exists():
        for d in data_dir.iterdir():
            if d.is_dir() and d.name != 'import' and (d / "metadata.json").exists():
                available_datasets.append(d.name)

    ratios = {}
    if available_datasets:
        console.print(t("\n[Dataset Recipe]", "\n[データセット配合]"))
        remaining = 100
        for i, ds in enumerate(available_datasets):
            if i == len(available_datasets) - 1:
                val = remaining
                console.print(f"- {ds}: [bold]{val}%[/bold] (Auto-filled)")
            else:
                val = IntPrompt.ask(f"- {ds} (Remaining: {remaining}%)", default=0)
                val = min(val, remaining)
            ratios[ds] = val / 100.0
            remaining -= val
    else:
        console.print(t("[yellow]No datasets found. Using default logic.[/yellow]", "[yellow]データセットが見つかりません。[/yellow]"))

    # 4. Parameter Proposal
    # Default params
    d_model = 512
    n_layers = 6
    batch_size = 4
    seq_len = 1024
    epochs = 1

    # Logic based on goal & calibration
    if goal == "1": # Debug
        d_model, n_layers = 256, 4
        batch_size, seq_len = 2, 512
        epochs = 1
    elif goal == "2": # Benchmark
        d_model, n_layers = 1024, 12
        batch_size, seq_len = 1, 8192 # Push seq len
        epochs = 1
    elif goal == "3": # Production
        d_model, n_layers = 768, 12
        batch_size, seq_len = 8, 2048
        epochs = 3

    # Apply calibration limits if available
    if cal and cal.memory_coeffs['base'] > 0:
        mem, _ = cal.predict(batch_size, seq_len, d_model, n_layers)
        limit = cal.vram_total * 0.9

        if mem > limit:
            console.print(t(f"[red]Proposal {mem:.0f}MB exceeds VRAM {limit:.0f}MB. Downgrading...[/red]", f"[red]提案設定 ({mem:.0f}MB) がVRAM ({limit:.0f}MB) を超えます。設定を下げます...[/red]"))
            while mem > limit and batch_size > 1:
                batch_size = max(1, batch_size // 2)
                mem, _ = cal.predict(batch_size, seq_len, d_model, n_layers)

            while mem > limit and seq_len > 512:
                seq_len = seq_len // 2
                mem, _ = cal.predict(batch_size, seq_len, d_model, n_layers)

    # Show Proposal
    table = Table(title=t("Recommended Configuration", "推奨設定"))
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("d_model", str(d_model))
    table.add_row("n_layers", str(n_layers))
    table.add_row("Batch Size", str(batch_size))
    table.add_row("Sequence Length", str(seq_len))
    table.add_row("Epochs", str(epochs))

    if cal and cal.memory_coeffs['base'] > 0:
        pred_mem, pred_time = cal.predict(batch_size, seq_len, d_model, n_layers)
        table.add_row("Est. VRAM", f"{pred_mem:.0f} MB / {cal.vram_total:.0f} MB")

    console.print(table)

    if not Confirm.ask(t("Accept this configuration?", "この設定で決定しますか？"), default=True):
        console.print(t("Manual tuning not yet implemented. Using proposed config.", "手動調整は未実装です。提案設定を使用します。"))

    # 5. Save
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)

    # Save Recipe
    with open(config_dir / "dataset_mixing.yaml", 'w') as f:
        yaml.dump({'mixing_ratios': ratios}, f)

    # Save Train Config
    train_config = {
        'd_model': d_model,
        'n_layers': n_layers,
        'batch_size': batch_size,
        'n_seq': seq_len,
        'epochs': epochs,
        'learning_rate': 1e-4 if goal == "3" else 1e-3
    }
    with open(config_dir / "user_train_config.yaml", 'w') as f:
        yaml.dump(train_config, f)

    console.print(t("\n[bold green]Ready to fly! 🚀[/bold green]", "\n[bold green]準備完了！ 🚀[/bold green]"))
    console.print(t("Run 'make train-user' to start.", "'make train-user' で発進してください。"))

if __name__ == "__main__":
    main()
