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

class AutoTuner:
    """
    Automated parameter tuning logic.
    Optimizes configuration to fit within VRAM target using bidirectional scaling.
    """
    def __init__(self, calibrator, goal):
        self.cal = calibrator
        self.goal = goal

        # Priority: d_model (Most impactful) -> n_layers -> batch_size -> seq_len
        self.priority = ['d_model', 'n_layers', 'batch_size', 'n_seq']

        # Constraints
        self.limits = {
            'd_model': {'min': 128, 'max': 4096, 'step': 64},
            'n_layers': {'min': 2, 'max': 256, 'step': 2},
            'batch_size': {'min': 1, 'max': 128, 'step': 1},
            'n_seq': {'min': 128, 'max': 4096, 'step': 128}
        }

        # Adjust limits based on goal? (Optional, but kept simple for now)
        if goal == "1": # Debug
            self.limits['d_model']['max'] = 1024
            self.limits['n_layers']['max'] = 16

    def tune(self, config, locked_params, target_vram_ratio, **kwargs):
        """
        Adjusts config to meet target_vram_ratio.
        Respects locked_params.
        Returns: (new_config, status_dict)
        """
        total_vram = self.cal.vram_total if self.cal.vram_total > 0 else 8192
        target_vram = total_vram * target_vram_ratio

        current_cfg = config.copy()
        iterations = 0
        max_iterations = 100
        direction = "stable"

        while iterations < max_iterations:
            # Predict current usage
            with contextlib.redirect_stderr(io.StringIO()):
                est_mem, _ = self.cal.predict(
                    current_cfg['batch_size'],
                    current_cfg['n_seq'],
                    current_cfg['d_model'],
                    current_cfg['n_layers'],
                    **kwargs
                )

            # Check convergence (within small margin or safe side)
            # Actually we want to be <= target.
            # If we are > target, we MUST reduce.
            # If we are < target, we CAN expand (up to a point).

            # Let's define "Convergence" as:
            # 1. Under limit: est_mem <= target_vram
            # 2. Close enough: est_mem >= target_vram * 0.95 (If we are expanding)

            if est_mem > target_vram:
                # REDUCTION PHASE
                direction = "reduce"
                changed = False
                for param in self.priority: # Reduce d_model first
                    if param in locked_params: continue

                    val = current_cfg[param]
                    lim = self.limits[param]

                    if val > lim['min']:
                        # Reduce
                        step = lim['step']
                        new_val = max(lim['min'], val - step)
                        current_cfg[param] = new_val
                        changed = True
                        break # One change per iteration to re-evaluate VRAM

                if not changed:
                    # Cannot reduce further (everything min or locked)
                    break

            elif est_mem < target_vram * 0.95:
                # EXPANSION PHASE
                # Only expand if we are significantly below (e.g., < 95%)
                direction = "expand"
                changed = False
                for param in self.priority: # Expand d_model first
                    if param in locked_params: continue

                    val = current_cfg[param]
                    lim = self.limits[param]

                    if val < lim['max']:
                        # Check if next step would blow limit?
                        # No, just expand, loop will catch it next time and reduce if needed.
                        # But to avoid oscillation, we should be careful.
                        # Simple approach: Expand, let next iteration reduce if overshot.
                        # To prevent infinite toggle, maybe check prediction here?

                        step = lim['step']
                        next_val = val + step

                        # Tentative check
                        temp_cfg = current_cfg.copy()
                        temp_cfg[param] = next_val
                        with contextlib.redirect_stderr(io.StringIO()):
                            temp_mem, _ = self.cal.predict(
                                temp_cfg['batch_size'], temp_cfg['n_seq'],
                                temp_cfg['d_model'], temp_cfg['n_layers'],
                                **kwargs
                            )

                        if temp_mem <= target_vram:
                            current_cfg[param] = next_val
                            changed = True
                            break # One change per iteration

                if not changed:
                    break
            else:
                # Converged (Between 95% and 100%)
                direction = "converged"
                break

            iterations += 1

        return current_cfg, {
            "iterations": iterations,
            "final_mem": est_mem,
            "direction": direction
        }

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

    goal_choice = IntPrompt.ask("Choice", choices=["1", "2", "3"], default="1")

    # 1.5 Model Architecture
    console.print(t("\nWhich model architecture to use?", "\n使用するモデルアーキテクチャを選択してください。"))
    console.print(t("1. Phase 3 (Standard ResNet-BK)", "1. Phase 3 (標準 ResNet-BK)"))
    console.print(t("2. Phase 7 (Hybrid Hyperbolic Attention)", "2. Phase 7 (ハイブリッド双曲アテンション)"))
    arch_choice = IntPrompt.ask("Choice", choices=["1", "2"], default="2")
    model_type = "phase3" if arch_choice == "1" else "phase7"
    predict_kwargs = {}
    if model_type == 'phase7':
        predict_kwargs['use_hybrid_attention'] = True
    else:
        # Default for Phase 3/4 MoE
        predict_kwargs['num_experts'] = 4


    # 2. Calibration
    cal = MuseCalibrator()
    if cal:
        cal.check_triton(strict=(goal_choice != "1"))

    if cal and cal.device.type == 'cuda':
        if Confirm.ask(t("Run hardware calibration?", "ハードウェア診断（キャリブレーション）を実行しますか？"), default=True):
            cal.calibrate()
    else:
        console.print(t("[yellow]Skipping calibration (CPU or module missing).[/yellow]", "[yellow]キャリブレーションをスキップします。[/yellow]"))

    # 3. Dataset Recipe (Simplified for this file update, keeping logic)
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

        # Quick re-impl of ratio logic
        jp_sets = [d for d in available_datasets if 'jp' in d.lower() or 'japanese' in d.lower() or 'wiki_ja' in d.lower()]
        code_sets = [d for d in available_datasets if 'code' in d.lower() or 'python' in d.lower() or 'evol' in d.lower()]
        general_sets = [d for d in available_datasets if d not in jp_sets and d not in code_sets]

        def get_ratios(strat):
            if strat == "4": return {}
            w = {'jp': 0.33, 'code': 0.33, 'gen': 0.34}
            if strat == "2": w = {'jp': 0.70, 'code': 0.15, 'gen': 0.15}
            elif strat == "3": w = {'jp': 0.15, 'code': 0.70, 'gen': 0.15}

            res = {}
            for s, k in [(jp_sets, 'jp'), (code_sets, 'code'), (general_sets, 'gen')]:
                if s:
                    for d in s: res[d] = w[k] / len(s)

            tot = sum(res.values())
            if tot > 0:
                for k in res: res[k] /= tot
            return res

        ratios = get_ratios(strategy)

        if strategy == "4" or not Confirm.ask(t("Use this mix?", "この配合でよろしいですか？"), default=True):
            console.print(t("Switching to manual...", "手動モードに切り替えます..."))
            rem = 100
            ratios = {}
            for i, ds in enumerate(available_datasets):
                val = IntPrompt.ask(f"- {ds} (Remaining: {rem}%)", default=0)
                val = min(val, rem)
                ratios[ds] = val / 100.0
                rem -= val

    # 4. Auto-Tuner Setup
    console.print(t("\n[Hardware Limit Settings]", "\n[ハードウェア制限設定]"))
    target_vram_percent = IntPrompt.ask(t("Target VRAM Usage (%)", "目標VRAM使用率 (%)"), default="90")
    target_vram_ratio = target_vram_percent / 100.0

    tuner = AutoTuner(cal, goal_choice)

    # Initial defaults
    config = {
        'd_model': 512, 'n_layers': 6, 'batch_size': 4, 'n_seq': 1024, 'epochs': 1
    }
    if goal_choice == "3": config['epochs'] = 3

    locked_params = {}

    # Initial Auto-Tune
    if cal and cal.memory_coeffs['base'] > 0:
        with console.status(t("Auto-tuning...", "自動最適化中...")):
            config, _ = tuner.tune(config, locked_params, target_vram_ratio, **predict_kwargs)

    # 5. Cascading Manual Loop
    while True:
        # Estimate
        est_mem = 0
        est_time = 0
        if cal and cal.memory_coeffs['base'] > 0:
            with contextlib.redirect_stderr(io.StringIO()):
                est_mem, est_time = cal.predict(
                    config['batch_size'], config['n_seq'], config['d_model'], config['n_layers'], **predict_kwargs
                )

        usage_pct = (est_mem / (cal.vram_total if cal.vram_total > 0 else 8192)) * 100

        # Display Status
        console.clear() # Optional: Clear screen for cleaner UI? maybe just print new table
        # Actually clearing might be too aggressive if user wants to see history. Let's just print.

        table = Table(title=t("Configuration Proposal", "設定プロポーザル"))
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Status", style="yellow")

        for k in ['d_model', 'n_layers', 'batch_size', 'n_seq', 'epochs']:
            val = config.get(k, 0)
            lock_status = "🔒 Locked" if k in locked_params else "Auto"
            if k == 'epochs': lock_status = "Manual" # Epochs not tuned by VRAM usually
            table.add_row(k, str(val), lock_status)

        table.add_row("Est. VRAM", f"{est_mem:.0f} MB ({usage_pct:.1f}%)", "")
        console.print(table)

        if usage_pct > 100:
             console.print(t("[bold red]⛔ LIMIT EXCEEDED[/bold red]", "[bold red]⛔ 上限超過[/bold red]"))
        elif usage_pct > target_vram_percent:
             console.print(t(f"[yellow]⚠ Usage {usage_pct:.1f}% > Target {target_vram_percent}%[/yellow]", f"[yellow]⚠ 目標超過: {usage_pct:.1f}% > {target_vram_percent}%[/yellow]"))

        # Interaction
        console.print(t("\nOptions:", "\n操作オプション:"))
        console.print(t(" [Enter] Accept & Start (Auto-fix if invalid)", " [Enter] 決定して開始 (超過時は自動修正)"))
        console.print(t(" [key=val] Set value (e.g. d_model=1024)", " [key=val] 値を指定 (例: d_model=1024)"))
        console.print(t(" [r] Reset locks", " [r] ロック解除"))

        user_input = Prompt.ask("Command")

        if not user_input:
            # Empty Enter
            if usage_pct > target_vram_percent or usage_pct < target_vram_percent * 0.9:
                # If not optimal, tune one last time and confirm
                if usage_pct > 100:
                     console.print(t("Fixing configuration to fit VRAM...", "VRAMに収まるよう自動修正します..."))
                else:
                     console.print(t("Optimizing usage...", "使用率を最適化します..."))

                config, _ = tuner.tune(config, locked_params, target_vram_ratio, **predict_kwargs)
                continue # Re-show table
            else:
                break # Go to save

        elif user_input.lower() == 'r':
            locked_params = {}
            console.print("Locks reset.")
            # Re-tune from scratch?
            continue

        elif "=" in user_input:
            try:
                k, v = user_input.split("=")
                k = k.strip()
                v = int(v.strip())

                # Map short names if needed
                key_map = {'seq': 'n_seq', 'seq_len': 'n_seq', 'bs': 'batch_size', 'batch': 'batch_size', 'layers': 'n_layers', 'dim': 'd_model'}
                k = key_map.get(k, k)

                if k in config:
                    config[k] = v
                    if k != 'epochs': # Don't lock epochs for tuner
                        locked_params[k] = True

                    # Trigger Auto-Tune for others
                    with console.status(t("Re-calculating...", "再計算中...")):
                        config, _ = tuner.tune(config, locked_params, target_vram_ratio, **predict_kwargs)
                else:
                    console.print(f"[red]Unknown parameter: {k}[/red]")
            except ValueError:
                console.print("[red]Invalid format. Use key=value[/red]")
        else:
            console.print("[red]Unknown command[/red]")

    # 6. Save (Same as before)
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)

    datasets_cfg = {}
    for ds, w in ratios.items():
        datasets_cfg[ds] = {'path': f"./data/{ds}", 'weight': float(w)}

    if not datasets_cfg:
        datasets_cfg = {'wiki_ja': {'path': "./data/wiki_ja", 'weight': 1.0}}

    with open(config_dir / "dataset_mixing.yaml", 'w') as f:
        yaml.dump({'datasets': datasets_cfg}, f)

    train_config = {
        'model_type': model_type,
        'd_model': config['d_model'], 'n_layers': config['n_layers'],
        'batch_size': config['batch_size'],
        'n_seq': config['n_seq'], 'epochs': config.get('epochs', 1),
        'learning_rate': 1e-4 if goal_choice == "3" else 1e-3
    }

    if model_type == 'phase7':
        # Add Phase 7 specific parameters
        # Sensible defaults, can be exposed to user later if needed
        train_config['num_heads'] = max(1, config['d_model'] // 128) # Keep head dim reasonable
        train_config['local_window_size'] = 128

    with open(config_dir / "user_train_config.yaml", 'w') as f:
        yaml.dump(train_config, f)

    console.print(t("\n[bold green]Ready to fly! 🚀[/bold green]", "\n[bold green]準備完了！ 🚀[/bold green]"))
    console.print(t("Run 'make train-user' to start.", "'make train-user' で発進してください。"))

if __name__ == "__main__":
    main()
