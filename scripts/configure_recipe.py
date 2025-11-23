#!/usr/bin/env python3
"""
MUSE Recipe Wizard
Configure dataset mixing ratios for training.
MUSE レシピウィザード - 学習データの配合設定
"""
import os
import sys
import yaml
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.prompt import IntPrompt

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
    console.print(t("[bold blue]MUSE Recipe Wizard[/bold blue]", "[bold blue]MUSE レシピウィザード[/bold blue]"))

    data_dir = Path("data")
    if not data_dir.exists():
        console.print(t("[red]Data directory not found. Run 'make setup' first.[/red]", "[red]data ディレクトリが見つかりません。先に 'make setup' を実行してください。[/red]"))
        return

    # 1. Detect Datasets
    # Look for subdirectories containing .bin files or metadata
    available_datasets = []
    for d in data_dir.iterdir():
        if d.is_dir() and d.name != 'import':
            # Check if it looks like a dataset
            if list(d.glob("*.bin")) or list(d.glob("*/*.bin")) or (d / "metadata.json").exists():
                available_datasets.append(d.name)

    if not available_datasets:
        console.print(t("[yellow]No datasets found.[/yellow]", "[yellow]利用可能なデータセットが見つかりません。[/yellow]"))
        return

    console.print(t(f"Available Datasets: {', '.join(available_datasets)}", f"利用可能なデータセット: {', '.join(available_datasets)}"))

    # 2. Input Ratios
    ratios = {}
    remaining = 100

    console.print(t("\nPlease enter the percentage for each dataset (Total must be 100%).", "\n各データセットの比率（%）を入力してください（合計100%になるように）。"))

    for i, ds in enumerate(available_datasets):
        is_last = (i == len(available_datasets) - 1)

        if is_last:
            val = remaining
            console.print(f"- {ds}: [bold]{val}%[/bold] (Auto-filled)")
        else:
            prompt_text = f"- {ds} (Remaining: {remaining}%)"
            val = IntPrompt.ask(prompt_text, default=0)
            if val > remaining:
                console.print(t(f"[red]Value too high. Max is {remaining}.[/red]", f"[red]値が大きすぎます。最大 {remaining} までです。[/red]"))
                val = remaining

        ratios[ds] = val / 100.0
        remaining -= val

    # 3. Save Config
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "dataset_mixing.yaml"

    config_data = {
        'mixing_ratios': ratios
    }

    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)

    console.print(t(f"\n[bold green]Recipe saved to {config_path}![/bold green]", f"\n[bold green]配合レシピを {config_path} に保存しました！[/bold green]"))

    # Show Summary Table
    table = Table(title=t("Current Recipe", "現在の配合レシピ"))
    table.add_column("Dataset", style="cyan")
    table.add_column("Ratio", style="magenta")

    for ds, ratio in ratios.items():
        table.add_row(ds, f"{ratio*100:.0f}%")

    console.print(table)
    console.print(t("Run 'make train-user' to start training with this recipe.", "このレシピで学習を始めるには 'make train-user' を実行してください。"))

if __name__ == "__main__":
    main()
