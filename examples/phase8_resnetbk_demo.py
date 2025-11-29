#!/usr/bin/env python3
"""
Phase8 ResNetBKベース使用例

Phase7（ResNetBK）を継承したPhase8の基本的な使用方法を示します。
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.models.phase8.integrated_model import Phase8IntegratedModel
from src.models.phase8.config import Phase8Config


def basic_usage():
    """基本的な使用例"""
    print("=== Phase8 基本的な使用例 ===\n")
    
    # Phase8設定（Phase7を継承）
    config = Phase8Config(
        d_model=512,
        num_heads=8,
        num_layers=6,
        vocab_size=32000,
        max_seq_len=2048,
        # Phase7機能（継承）
        use_bk_core=True,
        use_hybrid_attention=True,
        use_ar_ssm=True,
        # Phase8機能（新規）
        use_bk_hyperbolic=True,
        use_ar_ssm_fusion=True,
        enable_entailment_cones=False,
        enable_persistent_homology=False,
        enable_sheaf_attention=False
    )
    
    # モデル構築
    model = Phase8IntegratedModel(config)
    if torch.cuda.is_available():
        model = model.cuda()
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Device: {device}")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # ダミー入力
    batch_size = 2
    seq_len = 1024
    x = torch.randn(batch_size, seq_len, config.d_model, device=device)
    
    # Forward pass
    print("Forward pass実行中...")
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    # 出力確認
    if isinstance(output, dict):
        print(f"Output keys: {list(output.keys())}")
        if "output" in output:
            print(f"Output shape: {output['output'].shape}")
        if "diagnostics" in output:
            print("\n診断情報:")
            diagnostics = output["diagnostics"]
            
            # BK-Core診断
            if "bk_core" in diagnostics:
                bk_diag = diagnostics["bk_core"]
                print(f"  BK-Core:")
                print(f"    G_ii Real Mean: {bk_diag.get('g_ii_real_mean', 'N/A')}")
                print(f"    Resonance Detected: {bk_diag.get('resonance_detected', 'N/A')}")
            
            # BK-Hyperbolic診断
            if "bk_hyperbolic" in diagnostics:
                bk_hyp_diag = diagnostics["bk_hyperbolic"]
                print(f"  BK-Hyperbolic:")
                print(f"    Gate Mean: {bk_hyp_diag.get('gate_mean', 'N/A')}")
                print(f"    Curvature Adjusted: {bk_hyp_diag.get('curvature_adjusted', 'N/A')}")
            
            # AR-SSM Fusion診断
            if "ar_ssm_fusion" in diagnostics:
                ar_ssm_diag = diagnostics["ar_ssm_fusion"]
                print(f"  AR-SSM Fusion:")
                print(f"    Rank Mean: {ar_ssm_diag.get('rank_mean', 'N/A')}")
                print(f"    High Rank Ratio: {ar_ssm_diag.get('high_rank_ratio', 'N/A')}")
    else:
        print(f"Output shape: {output.shape}")
    
    print("\n✅ 基本的な使用例が完了しました。")


def phase7_migration():
    """Phase7からの移行例"""
    print("\n=== Phase7からの移行例 ===\n")
    
    # Phase7相当の設定
    phase7_config = {
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 6,
        "vocab_size": 32000,
        "max_seq_len": 2048,
        "use_bk_core": True,
        "use_hybrid_attention": True,
        "use_ar_ssm": True
    }
    
    # Phase8設定（Phase7を継承）
    phase8_config = Phase8Config(
        **phase7_config,
        # Phase8固有の機能を追加
        use_bk_hyperbolic=True,
        use_ar_ssm_fusion=True
    )
    
    print("Phase7設定:")
    for key, value in phase7_config.items():
        print(f"  {key}: {value}")
    
    print("\nPhase8設定（Phase7 + 拡張）:")
    print(f"  use_bk_hyperbolic: {phase8_config.use_bk_hyperbolic}")
    print(f"  use_ar_ssm_fusion: {phase8_config.use_ar_ssm_fusion}")
    
    # モデル構築
    model = Phase8IntegratedModel(phase8_config)
    print(f"\nPhase8モデルが構築されました。")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n✅ Phase7からの移行が完了しました。")


def optional_features():
    """オプション機能の使用例"""
    print("\n=== オプション機能の使用例 ===\n")
    
    # 全機能を有効化
    config = Phase8Config(
        d_model=512,
        num_heads=8,
        num_layers=6,
        vocab_size=32000,
        max_seq_len=2048,
        # Phase7機能
        use_bk_core=True,
        use_hybrid_attention=True,
        use_ar_ssm=True,
        # Phase8コア機能
        use_bk_hyperbolic=True,
        use_ar_ssm_fusion=True,
        # Phase8オプション機能
        enable_entailment_cones=True,
        enable_persistent_homology=True,
        enable_sheaf_attention=True
    )
    
    model = Phase8IntegratedModel(config)
    if torch.cuda.is_available():
        model = model.cuda()
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Device: {device}")
    print("有効化された機能:")
    print(f"  BK-Hyperbolic: {config.use_bk_hyperbolic}")
    print(f"  AR-SSM Fusion: {config.use_ar_ssm_fusion}")
    print(f"  Entailment Cones: {config.use_entailment_cones}")
    print(f"  Persistent Homology: {config.use_persistent_homology}")
    print(f"  Sheaf Attention: {config.use_sheaf_attention}")
    print()
    
    # Forward pass
    x = torch.randn(2, 512, config.d_model, device=device)
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    # 診断情報
    if isinstance(output, dict) and "diagnostics" in output:
        diagnostics = output["diagnostics"]
        print("診断情報:")
        
        if "entailment_cones" in diagnostics:
            print(f"  Entailment Cones: {diagnostics['entailment_cones']}")
        
        if "persistent_homology" in diagnostics:
            print(f"  Persistent Homology: {diagnostics['persistent_homology']}")
        
        if "sheaf_attention" in diagnostics:
            print(f"  Sheaf Attention: {diagnostics['sheaf_attention']}")
    
    print("\n✅ オプション機能の使用例が完了しました。")


def performance_comparison():
    """Phase7 vs Phase8性能比較"""
    print("\n=== Phase7 vs Phase8 性能比較 ===\n")
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping performance comparison.")
        return
    
    import time
    
    # 共通設定
    common_config = {
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 6,
        "vocab_size": 32000,
        "max_seq_len": 2048,
        "use_bk_core": True,
        "use_hybrid_attention": True,
        "use_ar_ssm": True
    }
    
    # Phase8設定（Phase7 + 拡張）
    phase8_config = Phase8Config(
        **common_config,
        use_bk_hyperbolic=True,
        use_ar_ssm_fusion=True
    )
    
    # Phase8モデル
    model = Phase8IntegratedModel(phase8_config).cuda()
    
    # ベンチマーク
    batch_size = 2
    seq_len = 1024
    x = torch.randn(batch_size, seq_len, phase8_config.d_model, device="cuda")
    
    # ウォームアップ
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
    
    # 測定
    torch.cuda.synchronize()
    start_time = time.time()
    
    num_iterations = 10
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(x)
    
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    
    throughput = (batch_size * seq_len * num_iterations) / elapsed_time
    
    print(f"Throughput: {throughput:.2f} tokens/sec")
    print(f"Average Time: {elapsed_time / num_iterations:.4f} sec/iteration")
    
    # メモリ使用量
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"Peak Memory: {peak_memory:.3f} GB")
    
    print("\n✅ 性能比較が完了しました。")


def main():
    """メイン実行"""
    print("Phase8 ResNetBKベース使用例\n")
    print("=" * 60)
    
    # 基本的な使用例
    basic_usage()
    
    # Phase7からの移行例
    phase7_migration()
    
    # オプション機能の使用例
    optional_features()
    
    # 性能比較
    performance_comparison()
    
    print("\n" + "=" * 60)
    print("全ての使用例が完了しました。")


if __name__ == "__main__":
    main()
