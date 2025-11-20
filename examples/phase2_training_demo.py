"""
Phase 2 Training Demo

このスクリプトは、Phase 2モデルの訓練方法を示すデモです。

Usage:
    python examples/phase2_training_demo.py

Author: Project MUSE Team
"""

import torch
from pathlib import Path

# Phase 2コンポーネント
from src.models.phase2 import create_phase2_model


def demo_training_setup():
    """訓練セットアップのデモ"""
    print("\n" + "="*80)
    print("Phase 2 Training Demo")
    print("="*80)
    
    # 1. モデルの作成
    print("\n1. Creating Phase 2 model...")
    model = create_phase2_model(
        preset='small',
        vocab_size=10000,
    )
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 2. 訓練スクリプトの使用方法
    print("\n2. Training Script Usage:")
    print("   " + "-"*76)
    
    print("\n   Basic training:")
    print("   $ python scripts/train_phase2.py --preset small --num-epochs 10")
    
    print("\n   With WandB logging:")
    print("   $ python scripts/train_phase2.py \\")
    print("       --preset base \\")
    print("       --use-wandb \\")
    print("       --wandb-project phase2-training \\")
    print("       --wandb-name my-experiment")
    
    print("\n   Custom configuration:")
    print("   $ python scripts/train_phase2.py \\")
    print("       --d-model 512 \\")
    print("       --n-layers 6 \\")
    print("       --batch-size 4 \\")
    print("       --learning-rate 1e-4 \\")
    print("       --use-triton \\")
    print("       --base-decay 0.01 \\")
    print("       --hebbian-eta 0.1 \\")
    print("       --snr-threshold 2.0")
    
    # 3. 診断情報の確認
    print("\n3. Diagnostic Information:")
    print("   " + "-"*76)
    print("   The training script logs the following metrics:")
    print("   - Γ (Forgetting Rate): mean, std, min, max")
    print("   - SNR (Signal-to-Noise Ratio): mean, low_ratio")
    print("   - Memory Resonance: num_resonant_modes, total_energy")
    print("   - Lyapunov Stability: stable_ratio, mean_energy")
    print("   - VRAM Usage: peak_vram_mb, current_vram_mb")
    
    # 4. チェックポイントの確認
    print("\n4. Checkpoint Management:")
    print("   " + "-"*76)
    print("   Checkpoints are saved to: checkpoints/phase2/")
    print("   - best_model.pt: Best model based on validation loss")
    print("   - checkpoint_epoch{N}.pt: Periodic checkpoints (every 5 epochs)")
    print("   - training_history.json: Complete training history")
    
    # 5. WandB可視化
    print("\n5. WandB Visualization:")
    print("   " + "-"*76)
    print("   Real-time metrics logged to WandB:")
    print("   - batch/gamma_mean, batch/gamma_std")
    print("   - batch/snr_mean, batch/snr_low_ratio")
    print("   - batch/resonant_modes, batch/resonance_energy")
    print("   - batch/stability_ratio, batch/fast_weight_energy")
    print("   - train/loss, val/loss, train/perplexity, val/perplexity")
    
    # 6. 訓練履歴の読み込み
    print("\n6. Loading Training History:")
    print("   " + "-"*76)
    print("   import json")
    print("   with open('checkpoints/phase2/training_history.json') as f:")
    print("       history = json.load(f)")
    print("   print(f\"Best val loss: {history['best_val_loss']}\")")
    print("   print(f\"Gamma history: {history['gamma_history']}\")")
    
    print("\n" + "="*80)
    print("Demo completed!")
    print("="*80 + "\n")


def demo_quick_test():
    """クイックテストのデモ"""
    print("\n" + "="*80)
    print("Quick Test Demo")
    print("="*80)
    
    print("\nTo run a quick test (3 epochs, small model):")
    print("$ python scripts/train_phase2.py \\")
    print("    --preset small \\")
    print("    --num-epochs 3 \\")
    print("    --batch-size 2 \\")
    print("    --num-train-batches 20 \\")
    print("    --num-val-batches 5")
    
    print("\nExpected output:")
    print("  - Training completes in ~2-3 minutes")
    print("  - Loss should decrease from ~8.0 to ~6.0")
    print("  - Γ values should be around 0.01-0.02")
    print("  - SNR should improve from ~1.5 to ~2.0")
    print("  - Checkpoints saved to checkpoints/phase2/")
    
    print("\n" + "="*80 + "\n")


def demo_wandb_setup():
    """WandBセットアップのデモ"""
    print("\n" + "="*80)
    print("WandB Setup Demo")
    print("="*80)
    
    print("\n1. Install WandB:")
    print("   $ pip install wandb")
    
    print("\n2. Login to WandB:")
    print("   $ wandb login")
    
    print("\n3. Run training with WandB:")
    print("   $ python scripts/train_phase2.py \\")
    print("       --preset base \\")
    print("       --use-wandb \\")
    print("       --wandb-project phase2-breath-of-life \\")
    print("       --wandb-name experiment-001")
    
    print("\n4. View results:")
    print("   Open your browser and go to:")
    print("   https://wandb.ai/your-username/phase2-breath-of-life")
    
    print("\n5. Key visualizations to check:")
    print("   - batch/gamma_mean: Should show Γ evolution over time")
    print("   - batch/snr_mean: Should show SNR improvement")
    print("   - batch/resonant_modes: Should show memory resonance")
    print("   - batch/stability_ratio: Should stay close to 1.0 (100%)")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Run all demos
    demo_training_setup()
    demo_quick_test()
    demo_wandb_setup()
    
    print("\n" + "="*80)
    print("All demos completed!")
    print("="*80)
    print("\nNext steps:")
    print("1. Run a quick test: python scripts/train_phase2.py --preset small --num-epochs 3")
    print("2. Check the output: checkpoints/phase2/training_history.json")
    print("3. Try with WandB: python scripts/train_phase2.py --use-wandb")
    print("="*80 + "\n")
