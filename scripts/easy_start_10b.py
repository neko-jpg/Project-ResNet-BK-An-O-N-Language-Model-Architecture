#!/usr/bin/env python3
"""
Easy Start 10B (Local Training)
===============================
Automates the entire process of setting up and training a 10B parameter Phase 8 model
on a local machine with a consumer GPU (e.g., RTX 3080/4090).

Steps:
1. Environment Check (CUDA/Triton)
2. Dependency Check
3. Dataset Recipe Generation (if missing)
4. Data Preparation (if missing)
5. Model Compression (10B Initialization)
6. Training Launch
"""

import os
import sys
import subprocess
import time
import yaml
from pathlib import Path
import importlib.util

# Ensure we can import rich
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.theme import Theme
    from rich.markdown import Markdown
except ImportError:
    print("Installing 'rich' library for better UI...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.theme import Theme
    from rich.markdown import Markdown

# Setup Rich Console
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "step": "bold magenta"
})
console = Console(theme=custom_theme)

# Global VRAM detection
DETECTED_VRAM_GB = 8.0  # Default to 8GB mode

def run_command(command, description, exit_on_error=True):
    """Runs a shell command with a spinner."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task(description, total=None)
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            if exit_on_error:
                progress.stop()
                console.print(f"[error]Failed to execute: {command}[/error]")
                console.print(Panel(e.stderr, title="Error Output", border_style="red"))
                sys.exit(1)
            return False, e.stderr

def check_gpu():
    """Checks for NVIDIA GPU and Triton compatibility."""
    console.print("[step]Step 1: Checking Environment (GPU & Triton)...[/step]")

    # Check PyTorch CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            console.print("[error]Error: No NVIDIA GPU detected![/error]")
            console.print("This training workflow requires a CUDA-enabled GPU (RTX 3080 or better recommended).")
            sys.exit(1)

        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        console.print(f"[success]✔ GPU Detected: {gpu_name} ({vram_gb:.1f} GB VRAM)[/success]")
        
        # Store VRAM for config selection
        global DETECTED_VRAM_GB
        DETECTED_VRAM_GB = vram_gb
        
        # Auto-select config based on VRAM
        if vram_gb <= 10:
            console.print("[info]Using RTX 3080 optimized config (8GB mode)[/info]")
        else:
            console.print("[info]Using standard 10B config[/info]")

        # Check Triton
        if importlib.util.find_spec("triton") is None:
            console.print("[warning]⚠ Triton not found. Attempting to install...[/warning]")
            run_command(f"{sys.executable} -m pip install triton", "Installing Triton...")

        import triton
        console.print(f"[success]✔ Triton Detected: v{triton.__version__}[/success]")
        console.print("[success]✔ Safe-Log Triton kernels available[/success]")

    except ImportError:
        console.print("[error]Error: PyTorch not installed correctly.[/error]")
        sys.exit(1)

def setup_dependencies():
    """Ensures dependencies are installed."""
    console.print("\n[step]Step 2: verifying Dependencies...[/step]")
    # We assume 'make setup' or 'pip install -r requirements.txt' might have been run,
    # but let's double check critical ones without full reinstall to save time.

    run_command(f"{sys.executable} -m pip install -r requirements.txt", "Syncing Dependencies...")
    console.print("[success]✔ Dependencies Synced[/success]")

def check_dataset_recipe():
    """Checks for dataset config, generates default if missing."""
    console.print("\n[step]Step 3: Checking Dataset Recipe...[/step]")
    config_path = Path("configs/dataset_mixing.yaml")

    if config_path.exists():
        console.print(f"[info]✔ Found existing recipe: {config_path}[/info]")
    else:
        console.print("[warning]⚠ No recipe found. Generating default 'High-Density' recipe...[/warning]")

        # Default Recipe: Cosmopedia + Code (High quality synthetic data)
        default_recipe = {
            "datasets": {
                "cosmopedia": {
                    "path": "./data/cosmopedia",
                    "weight": 0.5
                },
                "evol_instruct_code": {
                    "path": "./data/evol_instruct_code",
                    "weight": 0.5
                }
            }
        }

        with open(config_path, "w") as f:
            yaml.dump(default_recipe, f)
        console.print(f"[success]✔ Generated default recipe at {config_path}[/success]")

def check_and_prepare_data():
    """Checks if data exists, otherwise runs preparation."""
    console.print("\n[step]Step 4: Checking Data Availability...[/step]")

    # Check for the two default datasets we use
    data_dir = Path("./data")
    cosmopedia_exists = (data_dir / "cosmopedia").exists()
    code_exists = (data_dir / "evol_instruct_code").exists()

    if cosmopedia_exists and code_exists:
        console.print("[info]✔ Data directories found.[/info]")
    else:
        console.print("[warning]⚠ Data missing. Downloading/Preparing (Limit: 20k samples)...[/warning]")

        datasets_to_prep = []
        if not cosmopedia_exists: datasets_to_prep.append("cosmopedia")
        if not code_exists: datasets_to_prep.append("evol_instruct_code")

        cmd = f"{sys.executable} scripts/prepare_datasets.py --datasets {' '.join(datasets_to_prep)} --max_samples 20000"
        run_command(cmd, f"Preparing Datasets ({', '.join(datasets_to_prep)})...")
        console.print("[success]✔ Data Prepared[/success]")

def prepare_compression():
    """Runs the 10B compression/initialization step."""
    console.print("\n[step]Step 5: Preparing 10B Compressed Model...[/step]")

    checkpoint_dir = Path("checkpoints/compressed_10b_start")
    checkpoint_file = checkpoint_dir / "compressed_model.pt"

    if checkpoint_file.exists():
        console.print(f"[info]✔ Found existing 10B checkpoint at {checkpoint_dir}[/info]")
        return

    console.print("[info]Initializing and Compressing 10B Model (this may take a minute)...[/info]")

    # Auto-select architecture based on VRAM
    global DETECTED_VRAM_GB
    if DETECTED_VRAM_GB <= 10:
        # RTX 3080 8GB optimized
        d_model = 4096
        n_layers = 48
        console.print(f"[info]Using RTX 3080 config: d_model={d_model}, n_layers={n_layers}[/info]")
    else:
        # Standard 10B config
        d_model = 5120
        n_layers = 31
        console.print(f"[info]Using standard config: d_model={d_model}, n_layers={n_layers}[/info]")

    cmd = f"{sys.executable} scripts/compress_model.py --output_dir {checkpoint_dir} --d_model {d_model} --n_layers {n_layers}"
    run_command(cmd, "Compressing Model...")

    if checkpoint_file.exists():
         console.print(f"[success]✔ Model Compressed & Initialized: {checkpoint_file}[/success]")
    else:
         console.print("[error]❌ Compression failed to generate checkpoint file.[/error]")
         sys.exit(1)

def start_training():
    """Launches the training process."""
    console.print("\n[step]Step 6: Launching Training (Phase 8 10B)...[/step]")
    console.print(Panel(
        "Starting local training loop.\n"
        "Monitor the loss. Use 'Ctrl+C' to stop.\n"
        "Logs will be saved to 'logs/'",
        title="MUSE Phase 8",
        border_style="green"
    ))

    # Auto-select config based on VRAM
    global DETECTED_VRAM_GB
    if DETECTED_VRAM_GB <= 10:
        config_file = "configs/phase8_10b_rtx3080.yaml"
    else:
        config_file = "configs/phase8_10b.yaml"
    
    # Construct command
    cmd = [
        sys.executable, "scripts/train_phase8.py",
        "--config", config_file,
        "--resume-from", "checkpoints/compressed_10b_start/compressed_model.pt",
        "--dataset", "configs/dataset_mixing.yaml",
        "--optimizer", "muon",  # Use Muon optimizer by default
        "--compile"  # Enable torch.compile for speed
    ]
    
    console.print(f"[info]Config: {config_file}[/info]")

    # We use os.execv to replace the current process, so the training script takes over the terminal
    # allowing user to see live progress bars from the training script itself.
    os.execv(sys.executable, cmd)

def main():
    console.print(Panel.fit(
        "[bold cyan]MUSE Phase 8: 10B Local Setup & Train[/bold cyan]\n"
        "Automated Environment Construction & Launch",
        border_style="cyan"
    ))

    setup_dependencies()
    check_gpu()
    check_dataset_recipe()
    check_and_prepare_data()
    prepare_compression()
    start_training()

if __name__ == "__main__":
    main()
