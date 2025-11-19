"""
Phase 1 Configuration Presets

このモジュールは、異なるハードウェア構成やユースケースに最適化された
Phase1Configのプリセットを提供します。

Requirements:
    - 12.3: ハイパーパラメータドキュメント

Author: Project MUSE Team
"""

from typing import Dict, Any
from .config import Phase1Config


# ============================================================================
# Hardware-Specific Presets
# ============================================================================

def get_preset_8gb() -> Phase1Config:
    """
    8GB VRAM用のプリセット（RTX 3080 Mobile相当）
    
    最大圧縮を適用し、8GB制約内で動作することを保証します。
    
    特徴:
        - AR-SSM: 低ランク（max_rank=16）
        - HTT: 高圧縮（95%圧縮、rank=8）
        - Gradient Checkpointing: 有効
        - LNS: 無効（安定性優先）
    
    推奨用途:
        - RTX 3080 Mobile (8GB)
        - RTX 3070 (8GB)
        - 小規模モデルの訓練
    
    Returns:
        Phase1Config for 8GB VRAM
    
    Example:
        >>> config = get_preset_8gb()
        >>> model = create_phase1_model(vocab_size=50000, d_model=1024, config=config)
    """
    return Phase1Config(
        # AR-SSM: 低ランク設定
        ar_ssm_enabled=True,
        ar_ssm_max_rank=16,
        ar_ssm_min_rank=4,
        ar_ssm_gate_hidden_dim=None,  # Auto: d_model // 4
        ar_ssm_l1_regularization=0.01,  # 強いスパース化
        ar_ssm_use_fused_scan=True,
        
        # HTT: 高圧縮設定
        htt_enabled=True,
        htt_rank=8,
        htt_num_cores=2,
        htt_phase_encoding=True,
        htt_compression_target=0.05,  # 95% compression
        
        # LNS: 無効（安定性優先）
        lns_enabled=False,
        
        # Stability: 標準設定
        stability_monitoring_enabled=True,
        stability_threshold=1e-6,
        schatten_s1_bound=100.0,
        schatten_s2_bound=50.0,
        gradient_norm_threshold=10.0,
        
        # Memory: 最大最適化
        use_gradient_checkpointing=True,
        checkpoint_ar_ssm=True,
        checkpoint_htt=False,
        
        # Targets
        target_vram_gb=8.0,
        target_ppl_degradation=0.05,
        target_speedup=3.0,
    )


def get_preset_10gb() -> Phase1Config:
    """
    10GB VRAM用のプリセット（RTX 3080相当）
    
    バランスの取れた設定で、品質と効率を両立します。
    
    特徴:
        - AR-SSM: 中ランク（max_rank=32）
        - HTT: 標準圧縮（90%圧縮、rank=16）
        - Gradient Checkpointing: 有効
        - LNS: 無効（安定性優先）
    
    推奨用途:
        - RTX 3080 (10GB)
        - RTX 3090 (24GB、効率重視時）
        - 中規模モデルの訓練
    
    Returns:
        Phase1Config for 10GB VRAM
    
    Example:
        >>> config = get_preset_10gb()
        >>> model = create_phase1_model(vocab_size=50000, d_model=1024, config=config)
    """
    return Phase1Config(
        # AR-SSM: 中ランク設定
        ar_ssm_enabled=True,
        ar_ssm_max_rank=32,
        ar_ssm_min_rank=4,
        ar_ssm_gate_hidden_dim=None,
        ar_ssm_l1_regularization=0.001,  # 標準スパース化
        ar_ssm_use_fused_scan=True,
        
        # HTT: 標準圧縮設定
        htt_enabled=True,
        htt_rank=16,
        htt_num_cores=2,
        htt_phase_encoding=True,
        htt_compression_target=0.1,  # 90% compression
        
        # LNS: 無効
        lns_enabled=False,
        
        # Stability: 標準設定
        stability_monitoring_enabled=True,
        stability_threshold=1e-6,
        schatten_s1_bound=100.0,
        schatten_s2_bound=50.0,
        gradient_norm_threshold=10.0,
        
        # Memory: 標準最適化
        use_gradient_checkpointing=True,
        checkpoint_ar_ssm=True,
        checkpoint_htt=False,
        
        # Targets
        target_vram_gb=10.0,
        target_ppl_degradation=0.05,
        target_speedup=3.0,
    )


def get_preset_24gb() -> Phase1Config:
    """
    24GB VRAM用のプリセット（RTX 4090, RTX 3090相当）
    
    品質優先の設定で、圧縮を最小限に抑えます。
    
    特徴:
        - AR-SSM: 高ランク（max_rank=64）
        - HTT: 低圧縮（80%圧縮、rank=32）
        - Gradient Checkpointing: 無効（速度優先）
        - LNS: 無効
    
    推奨用途:
        - RTX 4090 (24GB)
        - RTX 3090 (24GB)
        - A100 (40GB/80GB、品質重視時）
        - 大規模モデルの訓練
    
    Returns:
        Phase1Config for 24GB VRAM
    
    Example:
        >>> config = get_preset_24gb()
        >>> model = create_phase1_model(vocab_size=50000, d_model=1024, config=config)
    """
    return Phase1Config(
        # AR-SSM: 高ランク設定
        ar_ssm_enabled=True,
        ar_ssm_max_rank=64,
        ar_ssm_min_rank=8,
        ar_ssm_gate_hidden_dim=None,
        ar_ssm_l1_regularization=0.0001,  # 弱いスパース化
        ar_ssm_use_fused_scan=True,
        
        # HTT: 低圧縮設定
        htt_enabled=True,
        htt_rank=32,
        htt_num_cores=2,
        htt_phase_encoding=True,
        htt_compression_target=0.2,  # 80% compression
        
        # LNS: 無効
        lns_enabled=False,
        
        # Stability: 標準設定
        stability_monitoring_enabled=True,
        stability_threshold=1e-6,
        schatten_s1_bound=100.0,
        schatten_s2_bound=50.0,
        gradient_norm_threshold=10.0,
        
        # Memory: 最小最適化（速度優先）
        use_gradient_checkpointing=False,
        checkpoint_ar_ssm=False,
        checkpoint_htt=False,
        
        # Targets
        target_vram_gb=24.0,
        target_ppl_degradation=0.01,  # 1% max (品質重視)
        target_speedup=3.0,
    )


# ============================================================================
# Use-Case Specific Presets
# ============================================================================

def get_preset_inference() -> Phase1Config:
    """
    推論専用プリセット
    
    LNSカーネルを有効化し、推論速度を最大化します。
    訓練には使用できません。
    
    特徴:
        - AR-SSM: 中ランク（max_rank=32）
        - HTT: 標準圧縮（90%圧縮、rank=16）
        - LNS: 有効（推論専用）
        - Gradient Checkpointing: 無効（推論では不要）
    
    推奨用途:
        - 本番環境での推論
        - リアルタイムアプリケーション
        - エッジデバイス展開
    
    Returns:
        Phase1Config for inference
    
    Example:
        >>> config = get_preset_inference()
        >>> model = create_phase1_model(vocab_size=50000, d_model=1024, config=config)
        >>> model.eval()  # 推論モードに設定
    """
    return Phase1Config(
        # AR-SSM: 中ランク設定
        ar_ssm_enabled=True,
        ar_ssm_max_rank=32,
        ar_ssm_min_rank=4,
        ar_ssm_gate_hidden_dim=None,
        ar_ssm_l1_regularization=0.001,
        ar_ssm_use_fused_scan=True,
        
        # HTT: 標準圧縮設定
        htt_enabled=True,
        htt_rank=16,
        htt_num_cores=2,
        htt_phase_encoding=True,
        htt_compression_target=0.1,
        
        # LNS: 有効（推論専用）
        lns_enabled=True,
        lns_block_size_m=128,
        lns_block_size_n=128,
        lns_block_size_k=32,
        lns_use_max_log=True,
        
        # Stability: 監視のみ（修正なし）
        stability_monitoring_enabled=True,
        stability_threshold=1e-6,
        schatten_s1_bound=100.0,
        schatten_s2_bound=50.0,
        gradient_norm_threshold=10.0,
        
        # Memory: 無効（推論では不要）
        use_gradient_checkpointing=False,
        checkpoint_ar_ssm=False,
        checkpoint_htt=False,
        
        # Targets
        target_vram_gb=8.0,
        target_ppl_degradation=0.05,
        target_speedup=5.0,  # LNSによる追加高速化
    )


def get_preset_maximum_quality() -> Phase1Config:
    """
    最高品質プリセット
    
    圧縮を最小限に抑え、PPL劣化を1%以下に抑えます。
    大規模VRAMが必要です。
    
    特徴:
        - AR-SSM: 超高ランク（max_rank=128）
        - HTT: 最小圧縮（70%圧縮、rank=64）
        - LNS: 無効
        - Gradient Checkpointing: 無効
    
    推奨用途:
        - ベンチマーク実験
        - 品質比較のベースライン
        - 大規模GPU（A100 80GB等）
    
    Returns:
        Phase1Config for maximum quality
    
    Example:
        >>> config = get_preset_maximum_quality()
        >>> model = create_phase1_model(vocab_size=50000, d_model=1024, config=config)
    """
    return Phase1Config(
        # AR-SSM: 超高ランク設定
        ar_ssm_enabled=True,
        ar_ssm_max_rank=128,
        ar_ssm_min_rank=16,
        ar_ssm_gate_hidden_dim=None,
        ar_ssm_l1_regularization=0.0001,  # 最小スパース化
        ar_ssm_use_fused_scan=True,
        
        # HTT: 最小圧縮設定
        htt_enabled=True,
        htt_rank=64,
        htt_num_cores=2,
        htt_phase_encoding=True,
        htt_compression_target=0.3,  # 70% compression
        
        # LNS: 無効
        lns_enabled=False,
        
        # Stability: 標準設定
        stability_monitoring_enabled=True,
        stability_threshold=1e-6,
        schatten_s1_bound=100.0,
        schatten_s2_bound=50.0,
        gradient_norm_threshold=10.0,
        
        # Memory: 無効（品質優先）
        use_gradient_checkpointing=False,
        checkpoint_ar_ssm=False,
        checkpoint_htt=False,
        
        # Targets
        target_vram_gb=40.0,  # A100 40GB相当
        target_ppl_degradation=0.01,  # 1% max
        target_speedup=3.0,
    )


def get_preset_maximum_efficiency() -> Phase1Config:
    """
    最高効率プリセット
    
    圧縮を最大化し、6GB VRAMでも動作可能にします。
    品質は多少犠牲になります。
    
    特徴:
        - AR-SSM: 超低ランク（max_rank=16）
        - HTT: 超高圧縮（95%圧縮、rank=8）
        - LNS: 無効（安定性優先）
        - Gradient Checkpointing: 有効
    
    推奨用途:
        - 超低VRAMデバイス（6GB）
        - プロトタイピング
        - 教育用途
    
    Returns:
        Phase1Config for maximum efficiency
    
    Example:
        >>> config = get_preset_maximum_efficiency()
        >>> model = create_phase1_model(vocab_size=50000, d_model=1024, config=config)
    """
    return Phase1Config(
        # AR-SSM: 超低ランク設定
        ar_ssm_enabled=True,
        ar_ssm_max_rank=16,
        ar_ssm_min_rank=2,
        ar_ssm_gate_hidden_dim=None,
        ar_ssm_l1_regularization=0.01,  # 強いスパース化
        ar_ssm_use_fused_scan=True,
        
        # HTT: 超高圧縮設定
        htt_enabled=True,
        htt_rank=8,
        htt_num_cores=2,
        htt_phase_encoding=True,
        htt_compression_target=0.05,  # 95% compression
        
        # LNS: 無効
        lns_enabled=False,
        
        # Stability: 標準設定
        stability_monitoring_enabled=True,
        stability_threshold=1e-6,
        schatten_s1_bound=100.0,
        schatten_s2_bound=50.0,
        gradient_norm_threshold=10.0,
        
        # Memory: 最大最適化
        use_gradient_checkpointing=True,
        checkpoint_ar_ssm=True,
        checkpoint_htt=False,
        
        # Targets
        target_vram_gb=6.0,
        target_ppl_degradation=0.10,  # 10% max (効率優先)
        target_speedup=3.0,
    )


# ============================================================================
# Preset Registry
# ============================================================================

PRESET_REGISTRY: Dict[str, Phase1Config] = {
    # Hardware-specific
    "8gb": get_preset_8gb(),
    "10gb": get_preset_10gb(),
    "24gb": get_preset_24gb(),
    
    # Use-case specific
    "inference": get_preset_inference(),
    "max_quality": get_preset_maximum_quality(),
    "max_efficiency": get_preset_maximum_efficiency(),
}


def get_preset(name: str) -> Phase1Config:
    """
    名前でプリセットを取得
    
    Args:
        name: プリセット名（"8gb", "10gb", "24gb", "inference", 
              "max_quality", "max_efficiency"）
              または "speed_oriented", "memory_oriented" などのエイリアス
    
    Returns:
        Phase1Config
    
    Raises:
        ValueError: 不明なプリセット名
    
    Example:
        >>> config = get_preset("8gb")
        >>> config = get_preset("inference")
        >>> config = get_preset("speed_oriented")  # エイリアス
    """
    # エイリアスのマッピング
    aliases = {
        "speed_oriented": "24gb",
        "memory_oriented": "8gb",
        "balanced": "10gb",
        "quality": "max_quality",
        "efficiency": "max_efficiency",
    }
    
    # エイリアスを解決
    resolved_name = aliases.get(name, name)
    
    if resolved_name not in PRESET_REGISTRY:
        available = ", ".join(list(PRESET_REGISTRY.keys()) + list(aliases.keys()))
        raise ValueError(
            f"Unknown preset '{name}'. Available presets: {available}"
        )
    
    return PRESET_REGISTRY[resolved_name]


def list_presets() -> Dict[str, str]:
    """
    利用可能なプリセットのリストと説明を返す
    
    Returns:
        Dict mapping preset name to description
    
    Example:
        >>> presets = list_presets()
        >>> for name, desc in presets.items():
        ...     print(f"{name}: {desc}")
    """
    return {
        "8gb": "8GB VRAM用（RTX 3080 Mobile相当）- 最大圧縮",
        "10gb": "10GB VRAM用（RTX 3080相当）- バランス型",
        "24gb": "24GB VRAM用（RTX 4090相当）- 品質優先",
        "inference": "推論専用 - LNS有効化、最高速度",
        "max_quality": "最高品質 - 圧縮最小、PPL劣化<1%",
        "max_efficiency": "最高効率 - 圧縮最大、6GB VRAM対応",
    }


def print_preset_comparison():
    """
    すべてのプリセットの比較表を出力
    
    Example:
        >>> print_preset_comparison()
    """
    print("=" * 80)
    print("Phase 1 Configuration Presets Comparison")
    print("=" * 80)
    print()
    
    headers = ["Preset", "VRAM", "AR-SSM Rank", "HTT Rank", "Compression", "Checkpointing", "LNS"]
    row_format = "{:<20} {:<8} {:<12} {:<10} {:<12} {:<15} {:<5}"
    
    print(row_format.format(*headers))
    print("-" * 80)
    
    for name, config in PRESET_REGISTRY.items():
        row = [
            name,
            f"{config.target_vram_gb:.0f}GB",
            f"{config.ar_ssm_min_rank}-{config.ar_ssm_max_rank}",
            str(config.htt_rank),
            f"{(1-config.htt_compression_target)*100:.0f}%",
            "Yes" if config.use_gradient_checkpointing else "No",
            "Yes" if config.lns_enabled else "No",
        ]
        print(row_format.format(*row))
    
    print()
    print("Recommendations:")
    print("  - 8GB VRAM (RTX 3080 Mobile): Use '8gb' preset")
    print("  - 10GB VRAM (RTX 3080): Use '10gb' preset")
    print("  - 24GB VRAM (RTX 4090): Use '24gb' preset")
    print("  - Inference deployment: Use 'inference' preset")
    print("  - Quality benchmarks: Use 'max_quality' preset")
    print("  - Ultra-low VRAM: Use 'max_efficiency' preset")
    print()
