"""
Phase 2 Factory Demo

このスクリプトは、Phase 2モデルファクトリーの使用例を示します。

使用例:
    python examples/phase2_factory_demo.py
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.phase2 import (
    create_phase2_model,
    convert_phase1_to_phase2,
    get_phase2_preset,
    Phase2Config,
)


def demo_create_with_preset():
    """プリセットを使用したモデル作成のデモ"""
    print("=" * 60)
    print("Demo 1: Create Phase 2 Model with Preset")
    print("=" * 60)
    
    # Small preset
    print("\n1. Creating 'small' preset model...")
    model_small = create_phase2_model(preset="small")
    
    # Base preset
    print("\n2. Creating 'base' preset model...")
    model_base = create_phase2_model(preset="base")
    
    # Large preset
    print("\n3. Creating 'large' preset model...")
    model_large = create_phase2_model(preset="large")
    
    print("\n✓ All preset models created successfully!")


def demo_create_with_custom_config():
    """カスタム設定を使用したモデル作成のデモ"""
    print("\n" + "=" * 60)
    print("Demo 2: Create Phase 2 Model with Custom Config")
    print("=" * 60)
    
    # カスタム設定
    config = Phase2Config(
        vocab_size=30000,
        d_model=768,
        n_layers=8,
        n_seq=1024,
        num_heads=12,
        head_dim=64,
        base_decay=0.015,
        hebbian_eta=0.12,
        snr_threshold=2.5,
        resonance_enabled=True,
        use_zeta_init=True,
    )
    
    print("\nCustom configuration:")
    print(f"  - vocab_size: {config.vocab_size}")
    print(f"  - d_model: {config.d_model}")
    print(f"  - n_layers: {config.n_layers}")
    print(f"  - base_decay: {config.base_decay}")
    print(f"  - hebbian_eta: {config.hebbian_eta}")
    
    print("\nCreating model with custom config...")
    model = create_phase2_model(config=config)
    
    print("\n✓ Custom model created successfully!")


def demo_create_with_direct_params():
    """パラメータ直接指定によるモデル作成のデモ"""
    print("\n" + "=" * 60)
    print("Demo 3: Create Phase 2 Model with Direct Parameters")
    print("=" * 60)
    
    print("\nCreating model with direct parameters...")
    model = create_phase2_model(
        vocab_size=25000,
        d_model=512,
        n_layers=6,
        n_seq=2048,
    )
    
    print("\n✓ Model created with direct parameters!")


def demo_forward_pass():
    """Forward passのデモ"""
    print("\n" + "=" * 60)
    print("Demo 4: Forward Pass")
    print("=" * 60)
    
    # モデル作成（CPUで実行）
    print("\nCreating model on CPU...")
    model = create_phase2_model(preset="small", device=torch.device("cpu"))
    model.eval()
    
    # ダミー入力（CPUで作成）
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, 50257, (batch_size, seq_len))
    
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Input device: {input_ids.device}")
    print(f"Model device: {next(model.parameters()).device}")
    
    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        output = model(input_ids)
    
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    
    # 診断情報付きforward pass
    print("\nRunning forward pass with diagnostics...")
    with torch.no_grad():
        output, diagnostics = model(input_ids, return_diagnostics=True)
    
    print("\nDiagnostics keys:")
    for key in diagnostics.keys():
        print(f"  - {key}")
    
    print("\n✓ Forward pass completed successfully!")


def demo_phase1_to_phase2_conversion():
    """Phase 1からPhase 2への変換デモ"""
    print("\n" + "=" * 60)
    print("Demo 5: Phase 1 to Phase 2 Conversion")
    print("=" * 60)
    
    # Phase 1モデルのダミー（実際にはPhase1IntegratedModelを使用）
    print("\nCreating dummy Phase 1 model...")
    
    # 簡易的なダミーモデル
    class DummyPhase1Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(50257, 512)
            self.linear1 = torch.nn.Linear(512, 2048)
            self.linear2 = torch.nn.Linear(2048, 512)
            self.lm_head = torch.nn.Linear(512, 50257)
        
        def forward(self, x):
            x = self.embedding(x)
            x = self.linear1(x)
            x = torch.nn.functional.gelu(x)
            x = self.linear2(x)
            return self.lm_head(x)
    
    phase1_model = DummyPhase1Model()
    
    print(f"Phase 1 model parameters: {sum(p.numel() for p in phase1_model.parameters()):,}")
    
    # Phase 2に変換
    print("\nConverting to Phase 2...")
    phase2_config = Phase2Config(
        vocab_size=50257,
        d_model=512,
        n_layers=4,
        n_seq=1024,
    )
    
    phase2_model = convert_phase1_to_phase2(
        phase1_model,
        phase2_config=phase2_config,
        copy_compatible_weights=True,
        freeze_phase1_weights=False,
    )
    
    print(f"\nPhase 2 model parameters: {sum(p.numel() for p in phase2_model.parameters()):,}")
    
    print("\n✓ Conversion completed successfully!")


def demo_config_validation():
    """設定検証のデモ"""
    print("\n" + "=" * 60)
    print("Demo 6: Configuration Validation")
    print("=" * 60)
    
    # 正常な設定
    print("\n1. Valid configuration:")
    config_valid = Phase2Config(
        vocab_size=50000,
        d_model=512,
        n_layers=6,
    )
    try:
        config_valid.validate()
        print("   ✓ Configuration is valid")
    except ValueError as e:
        print(f"   ✗ Validation failed: {e}")
    
    # 無効な設定（負の値）
    print("\n2. Invalid configuration (negative base_decay):")
    config_invalid = Phase2Config(
        vocab_size=50000,
        d_model=512,
        n_layers=6,
        base_decay=-0.01,  # 無効
    )
    try:
        config_invalid.validate()
        print("   ✓ Configuration is valid")
    except ValueError as e:
        print(f"   ✗ Validation failed (expected): {str(e)[:100]}...")
    
    print("\n✓ Validation demo completed!")


def demo_preset_comparison():
    """プリセット比較のデモ"""
    print("\n" + "=" * 60)
    print("Demo 7: Preset Comparison")
    print("=" * 60)
    
    presets = ["small", "base", "large"]
    
    print("\nPreset configurations:")
    print(f"{'Preset':<10} {'Params':<15} {'d_model':<10} {'n_layers':<10} {'VRAM Target':<12}")
    print("-" * 60)
    
    for preset_name in presets:
        config = get_phase2_preset(preset_name)
        
        # パラメータ数を概算（簡易版）
        # Embedding: vocab_size * d_model
        # Blocks: n_layers * (複数のLinear層)
        # 概算: vocab_size * d_model + n_layers * (d_model^2 * 10)
        approx_params = (
            config.vocab_size * config.d_model +
            config.n_layers * (config.d_model ** 2 * 10)
        )
        
        print(f"{preset_name:<10} {approx_params:>13,}  {config.d_model:<10} "
              f"{config.n_layers:<10} {config.target_vram_gb:<12.1f} GB")
    
    print("\n✓ Preset comparison completed!")


def main():
    """メイン関数"""
    print("\n" + "=" * 60)
    print("Phase 2 Factory Demo")
    print("=" * 60)
    
    try:
        # Demo 1: プリセット
        demo_create_with_preset()
        
        # Demo 2: カスタム設定
        demo_create_with_custom_config()
        
        # Demo 3: 直接パラメータ指定
        demo_create_with_direct_params()
        
        # Demo 4: Forward pass
        demo_forward_pass()
        
        # Demo 5: Phase 1 → Phase 2変換
        demo_phase1_to_phase2_conversion()
        
        # Demo 6: 設定検証
        demo_config_validation()
        
        # Demo 7: プリセット比較
        demo_preset_comparison()
        
        print("\n" + "=" * 60)
        print("All demos completed successfully! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
