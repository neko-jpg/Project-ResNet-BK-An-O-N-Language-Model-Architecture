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
        console.print(t("\n[Dataset Recipe Strategy]", "\n[データセット配合戦略]"))
        console.print(t("1. Balanced (Auto)", "1. バランス型 (おまかせ)"))
        console.print(t("2. Japanese Focused (Auto)", "2. 日本語重視 (おまかせ)"))
        console.print(t("3. Code Heavy (Auto)", "3. コード重視 (おまかせ)"))
        console.print(t("4. Manual (Custom)", "4. 手動設定 (カスタム)"))

        strategy = IntPrompt.ask("Choice", choices=["1", "2", "3", "4"], default="1")

        if strategy == "4":
            # Manual Mode
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
            # Auto Modes
            ratios = {ds: 0.0 for ds in available_datasets}

            # Simple keyword matching
            jp_sets = [d for d in available_datasets if 'jp' in d or 'japanese' in d or 'wiki_ja' in d]
            code_sets = [d for d in available_datasets if 'code' in d or 'python' in d or 'evol' in d]
            general_sets = [d for d in available_datasets if d not in jp_sets and d not in code_sets]

            # Ensure no division by zero
            n_jp = len(jp_sets)
            n_code = len(code_sets)
            n_gen = len(general_sets)

            if strategy == "1": # Balanced
                # Distribute evenly across categories, then within categories
                # Target: 33% JP, 33% Code, 33% General
                if n_jp > 0:
                    for d in jp_sets: ratios[d] = 0.33 / n_jp
                if n_code > 0:
                    for d in code_sets: ratios[d] = 0.33 / n_code
                if n_gen > 0:
                    for d in general_sets: ratios[d] = 0.34 / n_gen

                # Normalize if some categories were empty
                total = sum(ratios.values())
                if total > 0:
                    for k in ratios: ratios[k] /= total

            elif strategy == "2": # Japanese Focused
                # 70% JP, 30% others
                target_jp = 0.7 if n_jp > 0 else 0.0
                target_others = 1.0 - target_jp

                if n_jp > 0:
                    for d in jp_sets: ratios[d] = target_jp / n_jp

                n_others = n_code + n_gen
                if n_others > 0:
                    for d in code_sets + general_sets: ratios[d] = target_others / n_others

            elif strategy == "3": # Code Heavy
                # 70% Code, 30% others
                target_code = 0.7 if n_code > 0 else 0.0
                target_others = 1.0 - target_code

                if n_code > 0:
                    for d in code_sets: ratios[d] = target_code / n_code

                n_others = n_jp + n_gen
                if n_others > 0:
                    for d in jp_sets + general_sets: ratios[d] = target_others / n_others

            # Display Proposed Ratio
            console.print(t("\n[Proposed Mix]", "\n[提案された配合]"))
            for ds, r in ratios.items():
                if r > 0.001:
                    console.print(f"- {ds}: [bold]{r*100:.1f}%[/bold]")

            if not Confirm.ask(t("Use this mix?", "この配合でよろしいですか？"), default=True):
                 # Fallback to manual if rejected
                console.print(t("Switching to manual mode...", "手動モードに切り替えます..."))
                ratios = {}
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
        # Use cal.vram_total or fallback
        total_vram = cal.vram_total if cal.vram_total > 0 else 8192 # Default 8GB if unknown
        limit = total_vram * 0.9

        if mem > limit:
            console.print(t(f"[red]Proposal {mem:.0f}MB exceeds VRAM {limit:.0f}MB. Downgrading...[/red]", f"[red]提案設定 ({mem:.0f}MB) がVRAM ({limit:.0f}MB) を超えます。設定を下げます...[/red]"))

            # Simple iterative reduction
            # 1. Reduce Batch Size
            while mem > limit and batch_size > 1:
                batch_size = max(1, batch_size // 2)
                mem, _ = cal.predict(batch_size, seq_len, d_model, n_layers)

            # 2. Reduce Sequence Length
            while mem > limit and seq_len > 512:
                seq_len = seq_len // 2
                mem, _ = cal.predict(batch_size, seq_len, d_model, n_layers)

            # 3. Reduce Layers (Last resort)
            while mem > limit and n_layers > 2:
                n_layers -= 2
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
        total_vram_disp = cal.vram_total if cal.vram_total > 0 else 8192
        table.add_row("Est. VRAM", f"{pred_mem:.0f} MB / {total_vram_disp:.0f} MB")

    console.print(table)

    if not Confirm.ask(t("Accept this configuration?", "この設定で決定しますか？"), default=True):
        console.print(t("Manual tuning not yet implemented. Using proposed config.", "手動調整は未実装です。提案設定を使用します。"))

    # 5. Save
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)

    # Save Recipe
    yaml_path = config_dir / "dataset_mixing.yaml"

    # Try to load existing to preserve paths/metadata
    existing_data = {}
    if yaml_path.exists():
        try:
            with open(yaml_path, 'r') as f:
                existing_data = yaml.safe_load(f) or {}
        except Exception:
            existing_data = {}

    if 'datasets' not in existing_data or not isinstance(existing_data['datasets'], dict):
        # Fallback if file is broken/missing: construct minimal valid config
        console.print(t("[yellow]Warning: Reconstructing dataset config from scratch (metadata lost).[/yellow]",
                        "[yellow]警告: データセット設定を再構築します（メタデータは失われます）。[/yellow]"))
        existing_data['datasets'] = {}
        # If we are reconstructing, we assume we can just use the keys from ratios (which came from data dir)
        for ds in ratios:
             existing_data['datasets'][ds] = {'path': f"./data/{ds}"}

    # Update weights
    for ds, weight in ratios.items():
        if ds in existing_data.get('datasets', {}):
            existing_data['datasets'][ds]['weight'] = float(weight)
        else:
            # New dataset found in data/ but not in yaml? Add it.
            existing_data.setdefault('datasets', {})[ds] = {
                'path': f"./data/{ds}",
                'weight': float(weight)
            }

    # Write back preserving structure
    with open(yaml_path, 'w') as f:
        yaml.dump(existing_data, f, default_flow_style=False, sort_keys=False)

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
