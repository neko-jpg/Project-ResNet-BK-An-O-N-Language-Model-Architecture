"""
Phase 1 Error Recovery Demo

このスクリプトは、Phase 1のエラーハンドリングと自動回復機能のデモンストレーションです。

Requirements: 5.3, 10.4, 10.5
"""

import torch
import torch.nn as nn
from src.models.phase1 import (
    Phase1Config,
    Phase1ErrorRecovery,
    VRAMExhaustedError,
    NumericalInstabilityError,
    create_recovery_context_manager,
)


def demo_vram_recovery():
    """VRAM不足からの自動回復をデモンストレーション。"""
    print("=" * 60)
    print("Demo 1: VRAM Exhaustion Recovery")
    print("=" * 60)
    
    # 回復システムを初期化
    recovery = Phase1ErrorRecovery(max_recovery_attempts=3)
    config = Phase1Config()
    
    print(f"\n初期設定:")
    print(f"  - Gradient Checkpointing: {config.use_gradient_checkpointing}")
    print(f"  - AR-SSM Max Rank: {config.ar_ssm_max_rank}")
    print(f"  - LNS Enabled: {config.lns_enabled}")
    
    # VRAMエラーをシミュレート
    error = VRAMExhaustedError(
        current_mb=9000.0,
        limit_mb=8000.0,
        suggestions=["Enable checkpointing", "Reduce rank"]
    )
    
    print(f"\nVRAMエラー発生: {error.current_mb}MB / {error.limit_mb}MB")
    
    # 回復を試行
    success = recovery.handle_vram_exhausted(
        error=error,
        config=config
    )
    
    if success:
        print(f"\n✓ 回復成功!")
        print(f"  - Gradient Checkpointing: {config.use_gradient_checkpointing}")
        print(f"  - AR-SSM Max Rank: {config.ar_ssm_max_rank}")
        print(f"  - LNS Enabled: {config.lns_enabled}")
    else:
        print(f"\n✗ 回復失敗")
    
    # 回復履歴を表示
    summary = recovery.get_recovery_summary()
    print(f"\n回復サマリー:")
    print(f"  - 総試行回数: {summary['total_recovery_attempts']}")
    print(f"  - 成功: {summary['successful_recoveries']}")
    print(f"  - 失敗: {summary['failed_recoveries']}")


def demo_stability_recovery():
    """数値的不安定性からの自動回復をデモンストレーション。"""
    print("\n" + "=" * 60)
    print("Demo 2: Numerical Instability Recovery")
    print("=" * 60)
    
    # 回復システムを初期化
    recovery = Phase1ErrorRecovery(max_recovery_attempts=3)
    config = Phase1Config()
    
    # ダミーモデルとオプティマイザを作成
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n初期設定:")
    print(f"  - Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"  - Stability Threshold: {config.stability_threshold:.2e}")
    
    # 数値的不安定性エラーをシミュレート
    error = NumericalInstabilityError(
        component="AR-SSM",
        diagnostics={
            "has_nan": True,
            "max_value": 1e20,
            "gradient_norm": 100.0,
        }
    )
    
    print(f"\n数値的不安定性検出:")
    print(f"  - Component: {error.component}")
    print(f"  - Has NaN: {error.diagnostics['has_nan']}")
    print(f"  - Gradient Norm: {error.diagnostics['gradient_norm']}")
    
    # 回復を試行
    success = recovery.handle_numerical_instability(
        error=error,
        optimizer=optimizer,
        config=config
    )
    
    if success:
        print(f"\n✓ 回復成功!")
        print(f"  - Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  - Stability Threshold: {config.stability_threshold:.2e}")
    else:
        print(f"\n✗ 回復失敗")
    
    # 回復履歴を表示
    summary = recovery.get_recovery_summary()
    print(f"\n回復サマリー:")
    print(f"  - 総試行回数: {summary['total_recovery_attempts']}")
    print(f"  - 成功: {summary['successful_recoveries']}")


def demo_context_manager():
    """コンテキストマネージャを使った自動回復をデモンストレーション。"""
    print("\n" + "=" * 60)
    print("Demo 3: Context Manager Auto-Recovery")
    print("=" * 60)
    
    # 回復システムを初期化
    recovery = Phase1ErrorRecovery()
    config = Phase1Config()
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n初期設定:")
    print(f"  - Gradient Checkpointing: {config.use_gradient_checkpointing}")
    print(f"  - Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    # コンテキストマネージャを使用
    print(f"\nコンテキストマネージャ内でエラーをシミュレート...")
    
    try:
        with create_recovery_context_manager(
            recovery,
            model=model,
            optimizer=optimizer,
            config=config
        ):
            # VRAMエラーをシミュレート
            raise VRAMExhaustedError(9000, 8000, [])
    except VRAMExhaustedError:
        print(f"✗ 回復失敗、エラーが再発生しました")
    else:
        print(f"✓ エラーから自動回復しました")
    
    print(f"\n回復後の設定:")
    print(f"  - Gradient Checkpointing: {config.use_gradient_checkpointing}")
    print(f"  - Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    # 回復履歴を表示
    summary = recovery.get_recovery_summary()
    print(f"\n回復サマリー:")
    print(f"  - 総試行回数: {summary['total_recovery_attempts']}")
    print(f"  - 成功: {summary['successful_recoveries']}")
    print(f"  - 失敗: {summary['failed_recoveries']}")


def demo_multiple_recoveries():
    """複数の回復試行をデモンストレーション。"""
    print("\n" + "=" * 60)
    print("Demo 4: Multiple Recovery Attempts")
    print("=" * 60)
    
    # 回復システムを初期化
    recovery = Phase1ErrorRecovery(max_recovery_attempts=5)
    config = Phase1Config()
    
    print(f"\n初期設定:")
    print(f"  - Gradient Checkpointing: {config.use_gradient_checkpointing}")
    print(f"  - AR-SSM Max Rank: {config.ar_ssm_max_rank}")
    print(f"  - LNS Enabled: {config.lns_enabled}")
    
    # 複数回のVRAMエラーをシミュレート
    for i in range(3):
        print(f"\n--- 回復試行 {i+1} ---")
        error = VRAMExhaustedError(9000, 8000, [])
        success = recovery.handle_vram_exhausted(error=error, config=config)
        
        if success:
            print(f"✓ 回復成功")
            print(f"  - Gradient Checkpointing: {config.use_gradient_checkpointing}")
            print(f"  - AR-SSM Max Rank: {config.ar_ssm_max_rank}")
            print(f"  - LNS Enabled: {config.lns_enabled}")
        else:
            print(f"✗ 回復失敗（すべての戦略が使い果たされました）")
            break
    
    # 最終的な回復サマリー
    summary = recovery.get_recovery_summary()
    print(f"\n最終回復サマリー:")
    print(f"  - 総試行回数: {summary['total_recovery_attempts']}")
    print(f"  - 成功: {summary['successful_recoveries']}")
    print(f"  - 失敗: {summary['failed_recoveries']}")
    print(f"\n実行されたアクション:")
    for action_type, count in summary['action_counts'].items():
        print(f"  - {action_type}: {count}回")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Phase 1 Error Recovery Demonstration")
    print("=" * 60)
    print("\nこのデモは、Phase 1のエラーハンドリングと自動回復機能を示します。")
    print("実際の学習ループでは、これらの機能が自動的にエラーから回復し、")
    print("学習の継続性を保証します。")
    
    # デモを実行
    demo_vram_recovery()
    demo_stability_recovery()
    demo_context_manager()
    demo_multiple_recoveries()
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\n詳細については、以下のドキュメントを参照してください:")
    print("  - docs/phase1_implementation_guide.md")
    print("  - src/models/phase1/errors.py")
    print("  - src/models/phase1/recovery.py")
