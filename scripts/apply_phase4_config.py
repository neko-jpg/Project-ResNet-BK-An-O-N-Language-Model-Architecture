import yaml
import sys
import os
from pathlib import Path
from rich.console import Console

# Import AutoTuner and Calibrator
# We need to add current dir to path to import scripts if we are running from root
sys.path.append(os.getcwd())
try:
    from scripts.configure_recipe import AutoTuner
    from scripts.calibration import MuseCalibrator
except ImportError:
    print("Error: Could not import AutoTuner or MuseCalibrator.")
    sys.path.append(os.path.join(os.getcwd(), 'scripts')) # Try adding scripts dir directly
    try:
        from configure_recipe import AutoTuner
        from calibration import MuseCalibrator
    except ImportError as e:
        print(f"Error: Failed to import dependencies: {e}")
        sys.exit(1)

console = Console()

def apply_phase4_settings():
    user_config_path = Path("configs/user_train_config.yaml")
    phase4_config_path = Path("configs/phase4_strongest.yaml")

    if not user_config_path.exists():
        console.print(f"[red]Error: {user_config_path} not found. Run 'make recipe' first.[/red]")
        sys.exit(1)

    # Load Phase 4 Defaults
    if not phase4_config_path.exists():
        console.print(f"[yellow]Warning: {phase4_config_path} not found. Using defaults.[/yellow]")
        phase4_config = {
            "use_bitnet": True,
            "use_symplectic": True,
            "symplectic_dt": 0.1,
            "use_non_hermitian": True,
            "use_birman_schwinger": True,
            "epsilon": 1.0,
            "use_mourre": True,
            "use_lap": True,
            # Memory saving defaults for Phase 4 on 8-10GB GPUs
            "use_mixed_precision": True,
            "use_gradient_checkpointing": True,
            "use_custom_kernels": True,
        }
    else:
        with open(phase4_config_path, 'r') as f:
            phase4_config = yaml.safe_load(f)

    # Load User Config
    with open(user_config_path, 'r') as f:
        user_config = yaml.safe_load(f)

    console.print(f"[bold blue]MUSE Phase 4 Upgrade Protocol[/bold blue]")
    console.print("Merging Phase 4 features (Symplectic, BitNet, etc.)...")

    # Inject Phase 4 settings
    for key, value in phase4_config.items():
        if key in user_config:
            # Overwrite even if exists, because Phase 4 enforces these flags
            # UNLESS it's a hardware param we want to tune.
            # But the phase4 config usually contains flags, not dims.
            # If phase4 config has d_model=128 (default), we should IGNORE it and let tuner decide.
            if key in ["d_model", "n_layers", "n_seq", "batch_size"]:
                continue

        user_config[key] = value
        console.print(f"  Injecting '{key}': {value}")

    # Initialize Calibration for Re-tuning
    console.print("\n[bold yellow]Re-calibrating hardware for Phase 4 overhead...[/bold yellow]")
    cal = MuseCalibrator()

    # We use a mocked calibration if on CPU or if we want to trust previous calibration?
    # Actually, we should try to load previous calibration cache if possible, but MuseCalibrator doesn't persist it well.
    # Ideally, we assume the user just ran 'make recipe' so the machine is the same.
    # If we are on GPU, we can calibrate quickly? Or just rely on the fallback coefficients?
    # MuseCalibrator will use fallback or zero if not run.
    # Let's try to run a quick calibration if coefficients are empty?
    # Or better: check if coefficients are zero.

    if cal.memory_coeffs['base'] == 0:
        if cal.device.type == 'cuda':
             console.print("  Hardware profile not found. Running quick calibration...")
             cal.calibrate()
        else:
             console.print("  Using CPU/Fallback profile.")
             # Set fallback defaults for CPU/safe mode so tuner works
             cal.memory_coeffs['per_complex'] = 5.0e-5
             cal.memory_coeffs['base'] = 800

    # Initialize AutoTuner
    # We don't know the goal ("Debug", "Benchmark", "Production") from the config easily.
    # But usually Phase 4 is for "Production" or "Benchmark".
    # We can infer from epochs?
    goal = "3" # Default to Production
    if user_config.get('epochs', 1) == 1 and user_config.get('n_layers', 0) < 4:
        goal = "1" # Debug-ish?

    tuner = AutoTuner(cal, goal)

    # Define locked params.
    # The user might have locked params in 'make recipe', but we don't have that state here.
    # However, Phase 4 MUST respect VRAM.
    # Strategy: Unlock d_model and n_layers to allow reduction.
    # Lock batch_size? No, let that float too.
    # We basically want to find the MAX d_model/n_layers that fits with Symplectic=True.

    # We restart tuning from the CURRENT user config as a starting point.
    locked_params = {}

    # Read target VRAM from somewhere? Or default to 90%?
    # The user instruction said "VRAM 90% target".
    target_vram_ratio = 0.80

    console.print(f"  Target VRAM: {target_vram_ratio*100:.0f}%")
    console.print(f"  Current Config: d={user_config.get('d_model')}, L={user_config.get('n_layers')}")

    # Run Tuning with Phase 4 flags
    phase4_flags = {
        'use_symplectic': user_config.get('use_symplectic', False),
        'use_bitnet': user_config.get('use_bitnet', False)
    }

    new_config, status = tuner.tune(user_config, locked_params, target_vram_ratio, **phase4_flags)

    # Symplectic requires even d_model
    if new_config.get('use_symplectic', False) and new_config.get('d_model', 0) % 2 != 0:
        new_config['d_model'] = max(128, new_config['d_model'] - 1)

    # Report changes
    if new_config['d_model'] < user_config['d_model']:
        console.print(f"  [red]Downscaled d_model: {user_config['d_model']} -> {new_config['d_model']}[/red] (to fit Symplectic state)")
    elif new_config['d_model'] > user_config['d_model']:
        console.print(f"  [green]Upscaled d_model: {user_config['d_model']} -> {new_config['d_model']}[/green]")
    else:
        console.print(f"  d_model retained: {new_config['d_model']}")

    if new_config['n_layers'] != user_config['n_layers']:
        console.print(f"  Adjusted n_layers: {user_config['n_layers']} -> {new_config['n_layers']}")

    # Final safety check with conservative estimator
    target_limit = cal.vram_total * target_vram_ratio if cal.vram_total > 0 else float('inf')

    def estimate_mem(cfg):
        return cal.predict(
            cfg['batch_size'],
            cfg['n_seq'],
            cfg['d_model'],
            cfg['n_layers'],
            use_symplectic=cfg.get('use_symplectic', False),
            use_gradient_checkpointing=cfg.get('use_gradient_checkpointing', False),
            use_bitnet=cfg.get('use_bitnet', False),
        )[0]

    est_mem = estimate_mem(new_config)
    if est_mem > target_limit:
        target_msg = f"{target_limit:.0f}MB" if target_limit != float('inf') else "target"
        console.print(f"  [yellow]Estimated usage {est_mem:.0f}MB exceeds target ({target_msg}). Downshifting for safety...[/yellow]")
        # Try reducing batch size first
        while est_mem > target_limit and new_config['batch_size'] > tuner.limits['batch_size']['min']:
            new_config['batch_size'] = max(tuner.limits['batch_size']['min'], new_config['batch_size'] // 2)
            est_mem = estimate_mem(new_config)
        # Then reduce d_model
        while est_mem > target_limit and new_config['d_model'] > tuner.limits['d_model']['min']:
            new_config['d_model'] = max(tuner.limits['d_model']['min'], new_config['d_model'] - tuner.limits['d_model']['step'])
            # Keep even dimension for symplectic
            if new_config.get('use_symplectic', False) and new_config['d_model'] % 2 != 0:
                new_config['d_model'] -= 1
            est_mem = estimate_mem(new_config)
        # Reduce depth if still heavy
        while est_mem > target_limit and new_config['n_layers'] > tuner.limits['n_layers']['min']:
            new_config['n_layers'] = max(tuner.limits['n_layers']['min'], new_config['n_layers'] - tuner.limits['n_layers']['step'])
            est_mem = estimate_mem(new_config)
        # Finally, shorten sequence length if still above target
        while est_mem > target_limit and new_config['n_seq'] > tuner.limits['n_seq']['min']:
            new_config['n_seq'] = max(tuner.limits['n_seq']['min'], new_config['n_seq'] - tuner.limits['n_seq']['step'])
            est_mem = estimate_mem(new_config)
        console.print(f"  [green]Memory guard applied: B={new_config['batch_size']}, d_model={new_config['d_model']}, n_seq={new_config['n_seq']} (est. {est_mem:.0f}MB)[/green]")

    # Save back
    with open(user_config_path, 'w') as f:
        yaml.dump(new_config, f, sort_keys=False)

    console.print(f"\n[bold green]Success! {user_config_path} updated.[/bold green]")
    console.print("Phase 4 active. Parameters optimized for VRAM limit.")

if __name__ == "__main__":
    apply_phase4_settings()
