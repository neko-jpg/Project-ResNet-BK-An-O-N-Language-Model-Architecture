#!/usr/bin/env python3
"""
MUSE Concierge - Training Wizard
Auto-configures training parameters based on hardware calibration and user goals.
"""
import os
import sys
import yaml
import time
import io
import contextlib
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
from rich.spinner import Spinner
from rich.status import Status
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

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

def optimize_parameters(cal, goal, target_vram_ratio):
    """
    Finds the maximum parameters (d_model, n_layers) that fit within the target VRAM.
    Prioritizes Model Size (d_model, n_layers) over Batch Size if constrained.
    """
    total_vram = cal.vram_total if cal.vram_total > 0 else 8192
    limit = total_vram * target_vram_ratio

    # 1. Base constraints based on Goal
    if goal == "2": # Benchmark
        batch_size = 1
        seq_len = 2048
        d_model = 256
        n_layers = 4
    elif goal == "3": # Production
        batch_size = 8
        seq_len = 1024
        d_model = 256
        n_layers = 4
    else: # Debug
        batch_size = 2
        seq_len = 512
        d_model = 128
        n_layers = 2

    final_mem = 0.0
    final_time = 0.0
    final_dm, final_nl = d_model, n_layers

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task_id = progress.add_task(t("Scaling up parameters...", "パラメータを最適化（拡大）中..."), total=None)

        # 2. Scale Up Phase (Using Predict)
        # We increase d_model and n_layers iteratively until we hit the limit
        # Strategy: Increase d_model by 64 and n_layers by 2 in steps
        while True:
            next_dm = final_dm + 64
            next_nl = final_nl + 2

            # Constraint check
            if next_dm > 2048: break # Soft cap

            # Predict
            with contextlib.redirect_stderr(io.StringIO()):
                mem, _ = cal.predict(batch_size, seq_len, next_dm, next_nl)

            if mem > limit:
                break # Stop growing

            final_dm = next_dm
            final_nl = next_nl

            progress.update(task_id, description=t(f"Simulating {final_dm}x{final_nl}...", f"シミュレーション中: {final_dm}x{final_nl}..."))
            time.sleep(0.01) # UI flush

        # 3. Verification Phase (Estimate Exact)
        # Verify the calculated max point. If OOM, reduce.
        attempts = 0
        valid_config = False

        # Start verification from the predicted max
        d_model, n_layers = final_dm, final_nl

        while not valid_config and attempts < 15:
            progress.update(task_id, description=t(f"Verifying {d_model}x{n_layers}...", f"実測検証中: {d_model}x{n_layers}..."))

            # Run exact check
            with contextlib.redirect_stderr(io.StringIO()):
                 real_mem, real_time = cal.estimate_exact(batch_size, seq_len, d_model, n_layers)

            if real_mem is not None and real_mem <= limit:
                # Success
                final_mem = real_mem
                final_time = real_time
                valid_config = True
                progress.update(task_id, description=t("Optimization Successful!", "最適化成功！"))
            else:
                # Failed (OOM or Limit Exceeded)
                progress.update(task_id, description=t(f"Limit exceeded. Reducing...", f"上限超過。縮小中..."))

                # Reduction Strategy: Prioritize keeping d_model if possible, but d_model is heavy.
                # Reduce d_model by 64
                d_model = max(128, d_model - 64)
                # If d_model is already small, reduce layers
                if d_model <= 256:
                     n_layers = max(2, n_layers - 2)

                attempts += 1
                if d_model <= 128 and n_layers <= 2:
                    # Minimum reached, just accept it (or fail later)
                    valid_config = True
                    final_mem = real_mem if real_mem else 9999
                    break

    return batch_size, seq_len, d_model, n_layers, final_mem, final_time

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

    # 2. Calibration & Triton Check
    cal = MuseCalibrator()

    # Check Triton: Strict unless Debug mode
    # If Goal=1 (Debug), strict=False. Else True.
    if cal:
        cal.check_triton(strict=(goal != "1"))

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

        # ... (Existing Recipe Logic - Simplified Copy) ...
        # (We need to reimplement the strategy logic here as we are overwriting)
        def assign_ratios(strategy_id):
            jp_sets = [d for d in available_datasets if 'jp' in d.lower() or 'japanese' in d.lower() or 'wiki_ja' in d.lower()]
            code_sets = [d for d in available_datasets if 'code' in d.lower() or 'python' in d.lower() or 'evol' in d.lower()]
            general_sets = [d for d in available_datasets if d not in jp_sets and d not in code_sets]

            cat_weights = {'jp': 0.33, 'code': 0.33, 'gen': 0.34}
            if strategy_id == "2": cat_weights = {'jp': 0.70, 'code': 0.15, 'gen': 0.15}
            elif strategy_id == "3": cat_weights = {'jp': 0.15, 'code': 0.70, 'gen': 0.15}

            ratios_local = {ds: 0.0 for ds in available_datasets}

            for sets, key in [(jp_sets, 'jp'), (code_sets, 'code'), (general_sets, 'gen')]:
                if sets:
                    for d in sets: ratios_local[d] = cat_weights[key] / len(sets)

            total = sum(ratios_local.values())
            if total > 0:
                for k in ratios_local: ratios_local[k] /= total
            return ratios_local

        if strategy == "4":
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
            ratios = assign_ratios(strategy)
            # Normalize
            if sum(ratios.values()) <= 0:
                 ratios = {ds: 1.0 / len(available_datasets) for ds in available_datasets}

            mix_table = Table(title=t("Proposed Mix", "提案された配合"))
            mix_table.add_column("Dataset", style="cyan")
            mix_table.add_column("Weight (%)", style="magenta")
            for ds, r in ratios.items():
                mix_table.add_row(ds, f"{r*100:.1f}")
            console.print(mix_table)

            if not Confirm.ask(t("Use this mix?", "この配合でよろしいですか？"), default=True):
                 # Fallback to manual
                 ratios = {} # simplified for brevity, assume user wants empty or restart
                 # Actually standard behavior: if no, go to manual.
                 # For now, to keep file size small, I'll assume Yes or simple fallback.
                 # Let's just implement simple manual fallback
                 console.print(t("Switching to manual...", "手動モードに切り替えます..."))
                 remaining = 100
                 for i, ds in enumerate(available_datasets):
                     val = IntPrompt.ask(f"- {ds} (Remaining: {remaining}%)", default=0)
                     val = min(val, remaining)
                     ratios[ds] = val / 100.0
                     remaining -= val

    else:
        console.print(t("[yellow]No datasets found.[/yellow]", "[yellow]データセットが見つかりません。[/yellow]"))

    # 4. Target VRAM & Parameter Optimization
    console.print(t("\n[Hardware Limit Settings]", "\n[ハードウェア制限設定]"))
    target_vram_percent = IntPrompt.ask(
        t("Target VRAM Usage (%)", "目標VRAM使用率 (%)"),
        default="90"
    )
    target_vram_ratio = target_vram_percent / 100.0

    # Default Start Points
    d_model, n_layers, batch_size, seq_len = 512, 6, 4, 1024
    epochs = 1
    if goal == "3": epochs = 3 # Production default

    # Auto-Optimize
    if cal and cal.memory_coeffs['base'] > 0:
        bs, sl, dm, nl, mem, time_sec = optimize_parameters(cal, goal, target_vram_ratio)
        d_model, n_layers, batch_size, seq_len = dm, nl, bs, sl

    # 5. Proposal & Manual Override Loop
    while True:
        table = Table(title=t("Configuration Proposal", "設定プロポーザル"))
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("d_model", str(d_model))
        table.add_row("n_layers", str(n_layers))
        table.add_row("Batch Size", str(batch_size))
        table.add_row("Sequence Length", str(seq_len))
        table.add_row("Epochs", str(epochs))

        # Verify stats
        est_mem = 0
        est_time = 0
        total_vram_disp = cal.vram_total if cal and cal.vram_total > 0 else 8192

        if cal and cal.memory_coeffs['base'] > 0:
            with console.status(t("Verifying config...", "設定を検証中...")):
                # Use predict first for speed, then verify?
                # Actually for final validation, we want estimate_exact if possible
                with contextlib.redirect_stderr(io.StringIO()):
                     r_mem, r_time = cal.estimate_exact(batch_size, seq_len, d_model, n_layers)
                     if r_mem is not None:
                         est_mem = r_mem
                     else:
                         # OOM or failed
                         p_mem, _ = cal.predict(batch_size, seq_len, d_model, n_layers)
                         est_mem = p_mem if p_mem > total_vram_disp else 999999

        usage_pct = (est_mem / total_vram_disp) * 100
        table.add_row("Est. VRAM", f"{est_mem:.0f} MB ({usage_pct:.1f}%)")
        console.print(table)

        # Check Limits
        if usage_pct > 100:
             console.print(t("[bold red]⛔ LIMIT EXCEEDED: VRAM usage > 100%[/bold red]", "[bold red]⛔ 上限超過: VRAM使用率が100%を超えています。再設定してください。[/bold red]"))
             # Force Override (No break)
        elif usage_pct > target_vram_percent:
             console.print(t(f"[yellow]⚠ Warning: Usage {usage_pct:.1f}% exceeds target {target_vram_percent}%[/yellow]",
                             f"[yellow]⚠ 警告: 使用率 {usage_pct:.1f}% が目標 {target_vram_percent}% を超えています[/yellow]"))
             if Confirm.ask(t("Accept anyway?", "それでも続行しますか？"), default=False):
                 break
        else:
             if Confirm.ask(t("Accept this configuration?", "この設定で決定しますか？"), default=True):
                 break

        # Manual Entry
        console.print(t("Enter manual overrides (Empty to keep):", "手動で上書きします（空Enterで維持）:"))
        try:
            d_model = int(Prompt.ask("d_model", default=str(d_model)))
            n_layers = int(Prompt.ask("n_layers", default=str(n_layers)))
            batch_size = int(Prompt.ask("Batch Size", default=str(batch_size)))
            seq_len = int(Prompt.ask("Sequence Length", default=str(seq_len)))
            epochs = int(Prompt.ask("Epochs", default=str(epochs)))
        except ValueError:
             console.print("[red]Invalid input[/red]")

    # 6. Save Config
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)

    if not ratios and available_datasets:
        ratios = {ds: 1.0 / len(available_datasets) for ds in available_datasets}

    datasets_cfg = {}
    for ds, w in ratios.items():
        datasets_cfg[ds] = {'path': f"./data/{ds}", 'weight': float(w)}

    if not datasets_cfg:
        datasets_cfg = {'wiki_ja': {'path': "./data/wiki_ja", 'weight': 1.0}}

    with open(config_dir / "dataset_mixing.yaml", 'w') as f:
        yaml.dump({'datasets': datasets_cfg}, f)

    train_config = {
        'd_model': d_model, 'n_layers': n_layers, 'batch_size': batch_size,
        'n_seq': seq_len, 'epochs': epochs,
        'learning_rate': 1e-4 if goal == "3" else 1e-3
    }
    with open(config_dir / "user_train_config.yaml", 'w') as f:
        yaml.dump(train_config, f)

    console.print(t("\n[bold green]Ready to fly! 🚀[/bold green]", "\n[bold green]準備完了！ 🚀[/bold green]"))
    console.print(t("Run 'make train-user' to start.", "'make train-user' で発進してください。"))

if __name__ == "__main__":
    main()
