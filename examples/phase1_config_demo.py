"""
Phase 1 Configuration System Demo

このスクリプトは、Phase 1 Efficiency Engineの設定システムの使用方法を示します。

Requirements:
    - 4.4: 使用例スクリプトの提供
    - 12.2: チュートリアルの作成

Usage:
    python examples/phase1_config_demo.py
"""

from src.models.phase1 import (
    Phase1Config,
    Phase1Diagnostics,
    Phase1TrainingState,
)


def demo_basic_config():
    """基本的な設定の作成と検証"""
    print("=" * 60)
    print("1. Basic Configuration")
    print("=" * 60)
    
    # デフォルト設定の作成
    config = Phase1Config()
    print(f"Default config created:")
    print(f"  AR-SSM max rank: {config.ar_ssm_max_rank}")
    print(f"  HTT rank: {config.htt_rank}")
    print(f"  Target VRAM: {config.target_vram_gb} GB")
    
    # 設定の検証
    try:
        config.validate()
        print("✓ Configuration is valid")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
    
    print()


def demo_hardware_presets():
    """ハードウェア別プリセット設定"""
    print("=" * 60)
    print("2. Hardware-Specific Presets")
    print("=" * 60)
    
    # 8GB VRAM用（最大圧縮）
    config_8gb = Phase1Config.for_hardware(vram_gb=8.0)
    print(f"8GB VRAM preset (Maximum Compression):")
    print(f"  AR-SSM max rank: {config_8gb.ar_ssm_max_rank}")
    print(f"  HTT compression: {config_8gb.htt_compression_target:.1%}")
    print(f"  Gradient checkpointing: {config_8gb.use_gradient_checkpointing}")
    
    # 10GB VRAM用（バランス型）
    config_10gb = Phase1Config.for_hardware(vram_gb=10.0)
    print(f"\n10GB VRAM preset (Balanced):")
    print(f"  AR-SSM max rank: {config_10gb.ar_ssm_max_rank}")
    print(f"  HTT compression: {config_10gb.htt_compression_target:.1%}")
    
    # 24GB VRAM用（品質優先）
    config_24gb = Phase1Config.for_hardware(vram_gb=24.0)
    print(f"\n24GB VRAM preset (Quality-Focused):")
    print(f"  AR-SSM max rank: {config_24gb.ar_ssm_max_rank}")
    print(f"  HTT compression: {config_24gb.htt_compression_target:.1%}")
    print(f"  Gradient checkpointing: {config_24gb.use_gradient_checkpointing}")
    
    print()


def demo_specialized_presets():
    """特殊用途プリセット設定"""
    print("=" * 60)
    print("3. Specialized Presets")
    print("=" * 60)
    
    # 推論専用設定
    config_inference = Phase1Config.for_inference()
    print(f"Inference-only preset:")
    print(f"  LNS enabled: {config_inference.lns_enabled}")
    print(f"  Gradient checkpointing: {config_inference.use_gradient_checkpointing}")
    
    # 品質最優先設定
    config_quality = Phase1Config.for_maximum_quality()
    print(f"\nMaximum Quality preset:")
    print(f"  AR-SSM max rank: {config_quality.ar_ssm_max_rank}")
    print(f"  HTT rank: {config_quality.htt_rank}")
    print(f"  Target PPL degradation: {config_quality.target_ppl_degradation:.1%}")
    
    # 効率最優先設定
    config_efficiency = Phase1Config.for_maximum_efficiency()
    print(f"\nMaximum Efficiency preset:")
    print(f"  AR-SSM max rank: {config_efficiency.ar_ssm_max_rank}")
    print(f"  HTT compression: {config_efficiency.htt_compression_target:.1%}")
    print(f"  Target VRAM: {config_efficiency.target_vram_gb} GB")
    
    print()


def demo_diagnostics():
    """診断情報の使用例"""
    print("=" * 60)
    print("4. Diagnostics System")
    print("=" * 60)
    
    # 診断情報の作成
    diag = Phase1Diagnostics(
        ar_ssm_effective_rank=16.5,
        ar_ssm_gate_sparsity=0.35,
        ar_ssm_memory_saved_mb=1200.0,
        htt_compression_ratio=0.095,
        htt_reconstruction_error=0.0012,
        bk_det_condition=1.5e-3,
        bk_schatten_s1=45.2,
        bk_schatten_s2=23.8,
        bk_min_eigenvalue=2.1e-4,
        forward_time_ms=12.5,
        backward_time_ms=18.3,
        peak_vram_mb=7650.0,
        throughput_tokens_per_sec=1250.0,
    )
    
    # 健全性チェック
    config = Phase1Config.for_hardware(vram_gb=8.0)
    is_healthy = diag.is_healthy(config)
    print(f"System health check: {'✓ HEALTHY' if is_healthy else '✗ UNHEALTHY'}")
    
    # サマリーの表示
    print("\n" + diag.get_summary())
    
    print()


def demo_training_state():
    """訓練状態の管理例"""
    print("=" * 60)
    print("5. Training State Management")
    print("=" * 60)
    
    # 訓練状態の初期化
    config = Phase1Config(ar_ssm_min_rank=4, ar_ssm_max_rank=32)
    state = Phase1TrainingState(rank_warmup_steps=1000)
    
    print(f"Initial state:")
    print(f"  Current max rank: {state.current_max_rank}")
    print(f"  Schedule step: {state.rank_schedule_step}")
    
    # ランクスケジュールの更新（カリキュラム学習）
    print(f"\nRank schedule progression:")
    for step in [0, 250, 500, 750, 1000]:
        state.rank_schedule_step = step
        state.update_rank_schedule(config)
        progress = step / state.rank_warmup_steps * 100
        print(f"  Step {step:4d} ({progress:5.1f}%): rank = {state.current_max_rank}")
    
    # メトリクスの更新
    print(f"\nMetrics tracking:")
    updated = state.update_best_metrics(ppl=12.5, vram_mb=7500.0)
    print(f"  Updated PPL: {updated['ppl']}, Updated VRAM: {updated['vram']}")
    print(f"  Best PPL: {state.best_ppl:.2f}")
    print(f"  Best VRAM: {state.best_vram_mb:.1f} MB")
    
    # より良いメトリクスで更新
    updated = state.update_best_metrics(ppl=11.2, vram_mb=7200.0)
    print(f"\n  Updated PPL: {updated['ppl']}, Updated VRAM: {updated['vram']}")
    print(f"  Best PPL: {state.best_ppl:.2f}")
    print(f"  Best VRAM: {state.best_vram_mb:.1f} MB")
    
    print()


def demo_serialization():
    """設定のシリアライゼーション例"""
    print("=" * 60)
    print("6. Configuration Serialization")
    print("=" * 60)
    
    # 設定の作成
    original_config = Phase1Config(
        ar_ssm_max_rank=64,
        htt_rank=32,
        target_vram_gb=10.0
    )
    
    # 辞書に変換
    config_dict = original_config.to_dict()
    print(f"Config serialized to dict:")
    print(f"  Keys: {len(config_dict)}")
    print(f"  Sample: ar_ssm_max_rank = {config_dict['ar_ssm_max_rank']}")
    
    # 辞書から復元
    restored_config = Phase1Config.from_dict(config_dict)
    print(f"\nConfig restored from dict:")
    print(f"  ar_ssm_max_rank: {restored_config.ar_ssm_max_rank}")
    print(f"  htt_rank: {restored_config.htt_rank}")
    print(f"  target_vram_gb: {restored_config.target_vram_gb}")
    
    # 訓練状態も同様
    state = Phase1TrainingState(current_max_rank=16, best_ppl=10.5)
    state_dict = state.to_dict()
    restored_state = Phase1TrainingState.from_dict(state_dict)
    print(f"\nTraining state serialization:")
    print(f"  Original best_ppl: {state.best_ppl}")
    print(f"  Restored best_ppl: {restored_state.best_ppl}")
    
    print()


def main():
    """メインデモ実行"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "Phase 1 Configuration System Demo" + " " * 14 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    demo_basic_config()
    demo_hardware_presets()
    demo_specialized_presets()
    demo_diagnostics()
    demo_training_state()
    demo_serialization()
    
    print("=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Implement AR-SSM Layer (Task 3)")
    print("  2. Implement HTT Embedding (Task 4)")
    print("  3. Implement Fused Scan Kernel (Task 5)")
    print()


if __name__ == "__main__":
    main()
