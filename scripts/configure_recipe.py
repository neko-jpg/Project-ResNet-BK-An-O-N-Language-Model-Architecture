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
        def _fallback_uniform():
            return {ds: 1.0 / len(available_datasets) for ds in available_datasets}

        console.print(t("\n[Dataset Recipe Strategy]", "\n[データセット配合戦略]"))
        console.print(t("1. Balanced (Auto)", "1. バランス型 (おまかせ)"))
        console.print(t("2. Japanese Focused (Auto)", "2. 日本語重視 (おまかせ)"))
        console.print(t("3. Code Heavy (Auto)", "3. コード重視 (おまかせ)"))
        console.print(t("4. Manual (Custom)", "4. 手動設定 (カスタム)"))

        strategy = IntPrompt.ask("Choice", choices=["1", "2", "3", "4"], default="1")

        # helper: deterministic ratio assignment
        def assign_ratios(strategy_id):
            jp_sets = [d for d in available_datasets if 'jp' in d.lower() or 'japanese' in d.lower() or 'wiki_ja' in d.lower()]
            code_sets = [d for d in available_datasets if 'code' in d.lower() or 'python' in d.lower() or 'evol' in d.lower()]
            general_sets = [d for d in available_datasets if d not in jp_sets and d not in code_sets]

            n_jp, n_code, n_gen = len(jp_sets), len(code_sets), len(general_sets)
            ratios_local = {ds: 0.0 for ds in available_datasets}

            if strategy_id == "1":  # Balanced: 33/33/34 across categories
                cat_weights = {'jp': 0.33, 'code': 0.33, 'gen': 0.34}
            elif strategy_id == "2":  # JP heavy: 70% JP
                cat_weights = {'jp': 0.70, 'code': 0.15, 'gen': 0.15}
            elif strategy_id == "3":  # Code heavy: 70% Code
                cat_weights = {'jp': 0.15, 'code': 0.70, 'gen': 0.15}
            else:
                cat_weights = {'jp': 1/3, 'code': 1/3, 'gen': 1/3}

            # distribute per category if exists
            if n_jp > 0:
                for d in jp_sets:
                    ratios_local[d] = cat_weights['jp'] / n_jp
            if n_code > 0:
                for d in code_sets:
                    ratios_local[d] = cat_weights['code'] / n_code
            if n_gen > 0:
                for d in general_sets:
                    ratios_local[d] = cat_weights['gen'] / n_gen

            # normalize to 1.0
            total = sum(ratios_local.values())
            if total > 0:
                for k in ratios_local:
                    ratios_local[k] /= total
            return ratios_local

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
            # Auto Modes with explicit ratios
            ratios = assign_ratios(strategy)

            # Display Proposed Ratio
            # Normalize if total is zero (edge cases)
            total_mix = sum(ratios.values())
            if total_mix <= 0 and available_datasets:
                ratios = {ds: 1.0 / len(available_datasets) for ds in available_datasets}

            # Display Proposed Ratio in a table
            if sum(ratios.values()) <= 0:
                ratios = _fallback_uniform()

            mix_table = Table(title=t("Proposed Mix", "提案された配合"))
            mix_table.add_column("Dataset", style="cyan")
            mix_table.add_column("Weight (%)", style="magenta")
            for ds, r in ratios.items():
                mix_table.add_row(ds, f"{r*100:.1f}")
            console.print(mix_table)

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

                # Show manual mix table
                mix_table = Table(title=t("Manual Mix", "手動配合"))
                mix_table.add_column("Dataset", style="cyan")
                mix_table.add_column("Weight (%)", style="magenta")
                for ds, r in ratios.items():
                    mix_table.add_row(ds, f"{r*100:.1f}")
                console.print(mix_table)

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
    def tune_with_cal(bs, slen, dm, nl, cal_obj):
        if cal_obj is None or cal_obj.memory_coeffs['base'] <= 0:
            return bs, slen, dm, nl, None, None
        total_vram = cal_obj.vram_total if cal_obj.vram_total > 0 else 8192  # MB
        limit = total_vram * 0.9
        mem, tcost = cal_obj.predict(bs, slen, dm, nl)
        while mem > limit:
            if bs > 1:
                bs = max(1, bs // 2)
            elif slen > 256:
                slen = max(256, slen // 2)
            elif dm > 256:
                dm = max(256, dm - 128)
            elif nl > 2:
                nl = max(2, nl - 2)
            else:
                break
            mem, tcost = cal_obj.predict(bs, slen, dm, nl)
        return bs, slen, dm, nl, mem, tcost

    tuned_mem = None
    tuned_time = None
    if cal and cal.memory_coeffs['base'] > 0:
        batch_size, seq_len, d_model, n_layers, tuned_mem, tuned_time = tune_with_cal(
            batch_size, seq_len, d_model, n_layers, cal
        )
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
        pred_mem, pred_time = cal.estimate_exact(batch_size, seq_len, d_model, n_layers) or cal.predict(batch_size, seq_len, d_model, n_layers)
        total_vram_disp = cal.vram_total if cal.vram_total > 0 else 8192
        table.add_row("Est. VRAM", f"{pred_mem:.0f} MB / {total_vram_disp:.0f} MB")
        if tuned_time is not None:
            table.add_row("Est. Step Time", f"{pred_time:.3f}s")

    console.print(table)


    if not Confirm.ask(t("Accept this configuration?", "この設定で決定しますか？"), default=True):
        console.print(t("Enter manual overrides. Press Enter to keep current value.", "手動で上書きします。空Enterで現在値を維持します。"))
        try:
            d_model = int(Prompt.ask("d_model", default=str(d_model)))
            n_layers = int(Prompt.ask("n_layers", default=str(n_layers)))
            batch_size = int(Prompt.ask("Batch Size", default=str(batch_size)))
            seq_len = int(Prompt.ask("Sequence Length", default=str(seq_len)))
            epochs = int(Prompt.ask("Epochs", default=str(epochs)))
        except Exception:
            console.print(t("[yellow]Invalid input detected. Keeping previous proposal.[/yellow]", "[yellow]入力が不正です。提案設定をそのまま使用します。[/yellow]"))

        # Re-run calibration check after manual override
        if cal and cal.memory_coeffs['base'] > 0:
            total_vram_disp = cal.vram_total if cal.vram_total > 0 else 8192
            pred_mem, pred_time = cal.estimate_exact(batch_size, seq_len, d_model, n_layers) or cal.predict(batch_size, seq_len, d_model, n_layers)
            if pred_mem > total_vram_disp * 0.9:
                console.print(t(f"[red]Warning: Est. VRAM {pred_mem:.0f} MB exceeds 90% of device ({total_vram_disp:.0f} MB).[/red]",
                                f"[red]警告: 推定VRAM {pred_mem:.0f} MB がデバイスの90% ({total_vram_disp:.0f} MB) を超えます。[/red]"))

    # 5. Save
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)

    # Save Recipe
    if not ratios and available_datasets:
        # If nothing was set (e.g., user skipped), fall back to uniform weights
        ratios = {ds: 1.0 / len(available_datasets) for ds in available_datasets}

    datasets_cfg = {}
    for ds, w in ratios.items():
        datasets_cfg[ds] = {
            'path': f"./data/{ds}",
            'weight': float(w)
        }

    # Final guard: ensure at least one dataset entry exists
    if not datasets_cfg:
        console.print(t("[red]No datasets selected. Creating a placeholder pointing to data/wiki_ja.[/red]",
                        "[red]�f�[�^�Z�b�g���������Ă��܂���Bdata/wiki_ja �ւ̕ύX�̕�����쐬���܂�.[/red]"))
        datasets_cfg = {
            'wiki_ja': {
                'path': "./data/wiki_ja",
                'weight': 1.0
            }
        }

    with open(config_dir / "dataset_mixing.yaml", 'w') as f:
        yaml.dump({'datasets': datasets_cfg}, f)

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
