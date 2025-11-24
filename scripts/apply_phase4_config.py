import yaml
import sys
from pathlib import Path

def apply_phase4_settings():
    user_config_path = Path("configs/user_train_config.yaml")
    phase4_config_path = Path("configs/phase4_strongest.yaml")

    if not user_config_path.exists():
        print(f"Error: {user_config_path} not found. Run 'make recipe' first.")
        sys.exit(1)

    if not phase4_config_path.exists():
        print(f"Error: {phase4_config_path} not found. Using defaults.")
        phase4_config = {
            "use_bitnet": True,
            "use_symplectic": True,
            "symplectic_dt": 0.1,
            "use_non_hermitian": True,
            "use_birman_schwinger": True,
            "epsilon": 1.0,
            "use_mourre": True,
            "use_lap": True
        }
    else:
        with open(phase4_config_path, 'r') as f:
            phase4_config = yaml.safe_load(f)

    with open(user_config_path, 'r') as f:
        user_config = yaml.safe_load(f)

    print("Merging Phase 4 Strongest settings into User Config...")

    # Keys to preserve from User Config (Hardware Auto-Tuned)
    preserved_keys = ["d_model", "n_layers", "n_seq", "batch_size", "epochs", "learning_rate"]

    # Apply Phase 4 settings, but do NOT overwrite preserved keys if they exist in user config
    for key, value in phase4_config.items():
        if key in preserved_keys and key in user_config:
            print(f"  Keeping tuned value for '{key}': {user_config[key]} (Phase 4 default: {value})")
            continue

        # Inject new setting
        user_config[key] = value
        print(f"  Injecting '{key}': {value}")

    # Save back
    with open(user_config_path, 'w') as f:
        yaml.dump(user_config, f, sort_keys=False)

    print(f"\nSuccess! {user_config_path} updated.")
    print("Your auto-tuned hardware settings (d_model, etc.) were preserved.")
    print("BitNet, Symplectic, and Non-Hermitian flags are now enabled.")

if __name__ == "__main__":
    apply_phase4_settings()
