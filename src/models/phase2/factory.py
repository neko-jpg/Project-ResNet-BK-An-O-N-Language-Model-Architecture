"""
Phase 2 Model Factory and Configuration Presets

このモジュールは、Phase 2モデルの生成とPhase 1からの変換を提供します。

主要機能:
1. create_phase2_model: 設定からPhase 2モデルを生成
2. convert_phase1_to_phase2: Phase 1モデルをPhase 2に変換
3. プリセット設定 (small, base, large)

Requirements: 6.1, 6.2
Author: Project MUSE Team
Date: 2025-01-20
"""

from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict
import warnings

import torch
import torch.nn as nn

from .integrated_model import Phase2IntegratedModel, Phase2Block
from .zeta_init import ZetaInitializer
from ..phase1.config import Phase1Config
from ..phase1.factory import Phase1IntegratedModel


@dataclass
class Phase2Config:
    """
    Phase 2の設定クラス
    
    Phase 1の設定を継承し、Phase 2固有のパラメータを追加します。
    
    Attributes:
        # Phase 1互換性
        phase1_config: Phase 1設定（オプション）
        
        # BK-Core Triton
        use_triton_bk: Tritonカーネルを使用するか
        triton_block_size: Tritonブロックサイズ
        
        # Non-Hermitian Potential
        base_decay: 最小減衰率 Γ_min
        adaptive_decay: 入力依存の減衰を使用するか
        schatten_p: Schatten Normのp値
        stability_threshold: 安定性閾値
        
        # Dissipative Hebbian
        hebbian_eta: Hebbian学習率 η
        hebbian_dt: 時間ステップ dt
        num_heads: ヘッド数
        head_dim: ヘッド次元
        
        # Lyapunov Stability
        gamma_adjust_rate: Γの自動調整係数
        energy_monitor_enabled: エネルギー監視を有効化するか
        
        # SNR Memory Filter
        snr_threshold: SNR閾値 τ
        snr_gamma_boost: 低SNR成分のΓ増加率
        snr_eta_boost: 高SNR成分のη増加率
        
        # Memory Resonance
        resonance_enabled: 共鳴層を有効化するか
        resonance_energy_threshold: 共鳴エネルギー閾値
        
        # Zeta Initialization
        use_zeta_init: ゼータ初期化を使用するか
        zeta_embedding_trainable: ゼータ埋め込みを学習可能にするか
        
        # Model Architecture
        vocab_size: 語彙サイズ
        d_model: モデル次元
        n_layers: レイヤー数
        n_seq: シーケンス長
        ffn_dim: FFN中間次元（Noneの場合は4*d_model）
        dropout: ドロップアウト率
        
        # Performance Targets
        target_vram_gb: 目標VRAM使用量（GB）
        target_speedup_triton: Tritonカーネルの目標高速化率
    
    Example:
        >>> # デフォルト設定
        >>> config = Phase2Config()
        >>> 
        >>> # Phase 1から変換
        >>> phase1_config = Phase1Config.for_hardware(vram_gb=8.0)
        >>> config = Phase2Config.from_phase1(phase1_config)
        >>> 
        >>> # カスタム設定
        >>> config = Phase2Config(
        ...     vocab_size=50000,
        ...     d_model=1024,
        ...     n_layers=12,
        ...     base_decay=0.02,
        ...     hebbian_eta=0.15
        ... )
    """
    
    # Phase 1互換性
    phase1_config: Optional[Phase1Config] = None
    
    # BK-Core Triton
    use_triton_bk: bool = True
    triton_block_size: int = 256
    
    # Non-Hermitian Potential
    base_decay: float = 0.001  # Reduced from 0.01 to prevent overdamping
    adaptive_decay: bool = True
    schatten_p: float = 1.0
    stability_threshold: float = 1e-3
    
    # Dissipative Hebbian
    hebbian_eta: float = 0.1
    hebbian_dt: float = 1.0
    num_heads: int = 8
    head_dim: int = 64
    
    # Lyapunov Stability
    gamma_adjust_rate: float = 0.01
    energy_monitor_enabled: bool = True
    
    # SNR Memory Filter
    snr_threshold: float = 2.0
    snr_gamma_boost: float = 2.0
    snr_eta_boost: float = 1.5
    
    # Memory Resonance
    resonance_enabled: bool = True
    resonance_energy_threshold: float = 0.1
    
    # Zeta Initialization
    use_zeta_init: bool = True
    zeta_embedding_trainable: bool = False
    
    # Model Architecture
    vocab_size: int = 50257
    d_model: int = 512
    n_layers: int = 6
    n_seq: int = 1024
    ffn_dim: Optional[int] = None
    dropout: float = 0.1
    
    # Performance Targets
    target_vram_gb: float = 8.0
    target_speedup_triton: float = 3.0
    
    def validate(self) -> None:
        """
        設定の整合性を検証
        
        Raises:
            ValueError: 設定が無効な場合
        """
        errors = []
        
        if self.base_decay <= 0:
            errors.append(f"base_decay must be > 0, got {self.base_decay}")
        
        if self.hebbian_eta <= 0:
            errors.append(f"hebbian_eta must be > 0, got {self.hebbian_eta}")
        
        if self.hebbian_dt <= 0:
            errors.append(f"hebbian_dt must be > 0, got {self.hebbian_dt}")
        
        if self.snr_threshold <= 0:
            errors.append(f"snr_threshold must be > 0, got {self.snr_threshold}")
        
        if self.resonance_energy_threshold <= 0:
            errors.append(f"resonance_energy_threshold must be > 0, got {self.resonance_energy_threshold}")
        
        if self.d_model <= 0:
            errors.append(f"d_model must be > 0, got {self.d_model}")
        
        if self.n_layers <= 0:
            errors.append(f"n_layers must be > 0, got {self.n_layers}")
        
        if self.n_seq <= 0:
            errors.append(f"n_seq must be > 0, got {self.n_seq}")
        
        if self.vocab_size <= 0:
            errors.append(f"vocab_size must be > 0, got {self.vocab_size}")
        
        if self.num_heads <= 0:
            errors.append(f"num_heads must be > 0, got {self.num_heads}")
        
        if self.head_dim <= 0:
            errors.append(f"head_dim must be > 0, got {self.head_dim}")
        
        if errors:
            raise ValueError("Phase2Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    @classmethod
    def from_phase1(cls, phase1_config: Phase1Config, **overrides) -> "Phase2Config":
        """
        Phase 1設定からPhase 2設定を生成
        
        Args:
            phase1_config: Phase 1の設定
            **overrides: 上書きするパラメータ
        
        Returns:
            Phase2Config
        
        Example:
            >>> phase1_config = Phase1Config.for_hardware(vram_gb=8.0)
            >>> phase2_config = Phase2Config.from_phase1(
            ...     phase1_config,
            ...     base_decay=0.02,
            ...     hebbian_eta=0.15
            ... )
        """
        config = cls(
            phase1_config=phase1_config,
            target_vram_gb=phase1_config.target_vram_gb,
            **overrides
        )
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Phase2Config":
        """辞書から設定を生成"""
        return cls(**config_dict)


# プリセット設定
def get_phase2_preset(preset_name: str) -> Phase2Config:
    """
    プリセット設定を取得
    
    Args:
        preset_name: プリセット名 ("small", "base", "large")
    
    Returns:
        Phase2Config
    
    Raises:
        ValueError: 未知のプリセット名
    
    Example:
        >>> config = get_phase2_preset("base")
        >>> model = create_phase2_model(config=config)
    """
    presets = {
        "small": Phase2Config(
            vocab_size=50257,
            d_model=256,
            n_layers=4,
            n_seq=512,
            num_heads=4,
            head_dim=64,
            ffn_dim=1024,
            target_vram_gb=4.0,
        ),
        "base": Phase2Config(
            vocab_size=50257,
            d_model=512,
            n_layers=6,
            n_seq=1024,
            num_heads=8,
            head_dim=64,
            ffn_dim=2048,
            target_vram_gb=8.0,
        ),
        "large": Phase2Config(
            vocab_size=50257,
            d_model=1024,
            n_layers=12,
            n_seq=2048,
            num_heads=16,
            head_dim=64,
            ffn_dim=4096,
            target_vram_gb=16.0,
        ),
    }
    
    if preset_name not in presets:
        raise ValueError(
            f"Unknown preset: {preset_name}. "
            f"Available presets: {', '.join(presets.keys())}"
        )
    
    return presets[preset_name]


def create_phase2_model(
    config: Optional[Phase2Config] = None,
    preset: Optional[str] = None,
    vocab_size: Optional[int] = None,
    d_model: Optional[int] = None,
    n_layers: Optional[int] = None,
    n_seq: Optional[int] = None,
    device: Optional[torch.device] = None,
    **kwargs
) -> Phase2IntegratedModel:
    """
    Phase 2統合モデルを作成するファクトリ関数
    
    設定またはプリセットからPhase 2モデルを生成します。
    
    Args:
        config: Phase2Config（Noneの場合はデフォルトまたはプリセット）
        preset: プリセット名 ("small", "base", "large")
        vocab_size: 語彙サイズ（configを上書き）
        d_model: モデル次元（configを上書き）
        n_layers: レイヤー数（configを上書き）
        n_seq: シーケンス長（configを上書き）
        device: デバイス（Noneの場合は自動検出）
        **kwargs: Phase2IntegratedModelへの追加引数
    
    Returns:
        Phase2IntegratedModel
    
    Example:
        >>> # デフォルト設定
        >>> model = create_phase2_model()
        >>> 
        >>> # プリセット使用
        >>> model = create_phase2_model(preset="base")
        >>> 
        >>> # カスタム設定
        >>> config = Phase2Config(
        ...     vocab_size=50000,
        ...     d_model=1024,
        ...     n_layers=12
        ... )
        >>> model = create_phase2_model(config=config)
        >>> 
        >>> # パラメータ直接指定
        >>> model = create_phase2_model(
        ...     vocab_size=50000,
        ...     d_model=1024,
        ...     n_layers=12,
        ...     n_seq=2048
        ... )
    
    Requirements: 6.1
    """
    # プリセットから設定を取得
    if preset is not None:
        if config is not None:
            warnings.warn(
                f"Both config and preset specified. Using preset '{preset}' and ignoring config.",
                UserWarning
            )
        config = get_phase2_preset(preset)
    
    # デフォルト設定
    if config is None:
        config = Phase2Config()
    
    # パラメータで上書き
    if vocab_size is not None:
        config.vocab_size = vocab_size
    if d_model is not None:
        config.d_model = d_model
    if n_layers is not None:
        config.n_layers = n_layers
    if n_seq is not None:
        config.n_seq = n_seq
    
    # 設定を検証
    config.validate()
    
    # デバイスを決定
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデルを作成
    model = Phase2IntegratedModel(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_seq=config.n_seq,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        use_triton=config.use_triton_bk,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
        base_decay=config.base_decay,
        adaptive_decay=config.adaptive_decay,
        hebbian_eta=config.hebbian_eta,
        hebbian_dt=config.hebbian_dt,
        snr_threshold=config.snr_threshold,
        snr_gamma_boost=config.snr_gamma_boost,
        snr_eta_boost=config.snr_eta_boost,
        resonance_enabled=config.resonance_enabled,
        resonance_energy_threshold=config.resonance_energy_threshold,
        use_zeta_init=config.use_zeta_init,
        zeta_embedding_trainable=config.zeta_embedding_trainable,
        **kwargs
    )
    
    # デバイスに移動
    model = model.to(device)
    
    # サマリーを表示
    print(_get_phase2_summary(model, config))
    
    return model


def _get_phase2_summary(model: Phase2IntegratedModel, config: Phase2Config) -> str:
    """
    Phase 2モデルのサマリーを生成
    
    Args:
        model: Phase2IntegratedModel
        config: Phase2Config
    
    Returns:
        サマリー文字列
    """
    # パラメータ数を計算
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    lines = [
        "=" * 60,
        "Phase 2 Integrated Model Summary",
        "=" * 60,
        "",
        "Architecture:",
        f"  - Vocabulary Size: {config.vocab_size:,}",
        f"  - Model Dimension: {config.d_model}",
        f"  - Number of Layers: {config.n_layers}",
        f"  - Sequence Length: {config.n_seq}",
        f"  - Number of Heads: {config.num_heads}",
        f"  - Head Dimension: {config.head_dim}",
        f"  - FFN Dimension: {config.ffn_dim or 4 * config.d_model}",
        "",
        "Phase 2 Components:",
        f"  - BK-Core Triton: {'Enabled' if config.use_triton_bk else 'Disabled'}",
        f"  - Non-Hermitian Forgetting: Enabled (Γ_base={config.base_decay})",
        f"  - Dissipative Hebbian: Enabled (η={config.hebbian_eta}, dt={config.hebbian_dt})",
        f"  - SNR Memory Filter: Enabled (τ={config.snr_threshold})",
        f"  - Memory Resonance: {'Enabled' if config.resonance_enabled else 'Disabled'}",
        f"  - Zeta Initialization: {'Enabled' if config.use_zeta_init else 'Disabled'}",
        "",
        "Parameters:",
        f"  - Total: {total_params:,}",
        f"  - Trainable: {trainable_params:,}",
        f"  - Non-trainable: {total_params - trainable_params:,}",
        "",
        "Performance Targets:",
        f"  - Target VRAM: {config.target_vram_gb:.1f} GB",
        f"  - Triton Speedup Target: {config.target_speedup_triton:.1f}x",
        "",
        "=" * 60,
    ]
    
    return "\n".join(lines)


def convert_phase1_to_phase2(
    phase1_model: Union[Phase1IntegratedModel, nn.Module],
    phase2_config: Optional[Phase2Config] = None,
    copy_compatible_weights: bool = True,
    freeze_phase1_weights: bool = False,
) -> Phase2IntegratedModel:
    """
    Phase 1モデルをPhase 2モデルに変換
    
    互換性のある層の重みをコピーし、新規層はゼータ初期化します。
    
    Args:
        phase1_model: Phase 1モデル
        phase2_config: Phase 2設定（Noneの場合はPhase 1から推定）
        copy_compatible_weights: 互換性のある重みをコピーするか
        freeze_phase1_weights: Phase 1由来の重みを凍結するか
    
    Returns:
        Phase2IntegratedModel
    
    Example:
        >>> # Phase 1モデルをロード
        >>> phase1_model = Phase1IntegratedModel(...)
        >>> 
        >>> # Phase 2に変換
        >>> phase2_model = convert_phase1_to_phase2(phase1_model)
        >>> 
        >>> # Phase 1の重みを凍結して、Phase 2固有の層のみ学習
        >>> phase2_model = convert_phase1_to_phase2(
        ...     phase1_model,
        ...     freeze_phase1_weights=True
        ... )
    
    Requirements: 6.2
    """
    # Phase 1の設定を取得
    if isinstance(phase1_model, Phase1IntegratedModel):
        phase1_config = phase1_model.config
        base_model = phase1_model.base_model
    else:
        phase1_config = None
        base_model = phase1_model
    
    # Phase 2設定を生成
    if phase2_config is None:
        if phase1_config is not None:
            phase2_config = Phase2Config.from_phase1(phase1_config)
        else:
            # Phase 1設定がない場合、モデルから推定
            phase2_config = _infer_phase2_config_from_model(base_model)
    
    # Phase 2モデルを作成
    phase2_model = create_phase2_model(config=phase2_config)
    
    # 互換性のある重みをコピー
    if copy_compatible_weights:
        _copy_compatible_weights(
            source_model=base_model,
            target_model=phase2_model,
            freeze_source_weights=freeze_phase1_weights
        )
    
    # 新規層にゼータ初期化を適用
    if phase2_config.use_zeta_init:
        _apply_zeta_init_to_new_layers(phase2_model)
    
    # 変換サマリーを表示
    print(_get_conversion_summary(phase1_model, phase2_model, phase2_config))
    
    return phase2_model


def _infer_phase2_config_from_model(model: nn.Module) -> Phase2Config:
    """
    モデルからPhase 2設定を推定
    
    Args:
        model: ベースモデル
    
    Returns:
        Phase2Config
    """
    # Embeddingから語彙サイズとモデル次元を推定
    vocab_size = 50257  # デフォルト
    d_model = 512  # デフォルト
    
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            vocab_size = module.num_embeddings
            d_model = module.embedding_dim
            break
    
    # レイヤー数を推定（簡易版）
    n_layers = 6  # デフォルト
    
    warnings.warn(
        f"Inferred Phase2Config from model: vocab_size={vocab_size}, "
        f"d_model={d_model}, n_layers={n_layers}. "
        "Please verify these values are correct.",
        UserWarning
    )
    
    return Phase2Config(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
    )


def _copy_compatible_weights(
    source_model: nn.Module,
    target_model: Phase2IntegratedModel,
    freeze_source_weights: bool = False
):
    """
    互換性のある重みをコピー
    
    Args:
        source_model: ソースモデル（Phase 1）
        target_model: ターゲットモデル（Phase 2）
        freeze_source_weights: ソース由来の重みを凍結するか
    """
    copied_count = 0
    
    # Token Embeddingをコピー
    for src_name, src_module in source_model.named_modules():
        if isinstance(src_module, nn.Embedding):
            # Phase 2のtoken_embeddingを探す
            if hasattr(target_model, 'token_embedding'):
                target_emb = target_model.token_embedding
                if isinstance(target_emb, nn.Embedding):
                    # サイズが一致する場合のみコピー
                    if (src_module.num_embeddings == target_emb.num_embeddings and
                        src_module.embedding_dim == target_emb.embedding_dim):
                        target_emb.weight.data.copy_(src_module.weight.data)
                        if freeze_source_weights:
                            target_emb.weight.requires_grad = False
                        copied_count += 1
                        print(f"Copied token embedding weights: {src_name}")
            break
    
    # Linear層をコピー（名前ベースのマッチング）
    source_linear_dict = {}
    for name, module in source_model.named_modules():
        if isinstance(module, nn.Linear):
            source_linear_dict[name] = module
    
    for tgt_name, tgt_module in target_model.named_modules():
        if isinstance(tgt_module, nn.Linear):
            # 対応するソース層を探す
            # 簡易版: 名前の一部が一致する場合
            for src_name, src_module in source_linear_dict.items():
                if _is_compatible_linear(src_module, tgt_module):
                    # 名前の類似性をチェック（簡易版）
                    if any(part in tgt_name for part in src_name.split('.')):
                        tgt_module.weight.data.copy_(src_module.weight.data)
                        if src_module.bias is not None and tgt_module.bias is not None:
                            tgt_module.bias.data.copy_(src_module.bias.data)
                        if freeze_source_weights:
                            tgt_module.weight.requires_grad = False
                            if tgt_module.bias is not None:
                                tgt_module.bias.requires_grad = False
                        copied_count += 1
                        print(f"Copied linear weights: {src_name} -> {tgt_name}")
                        break
    
    print(f"\nTotal compatible weights copied: {copied_count}")


def _is_compatible_linear(src: nn.Linear, tgt: nn.Linear) -> bool:
    """
    2つのLinear層が互換性があるかチェック
    
    Args:
        src: ソース層
        tgt: ターゲット層
    
    Returns:
        互換性があるか
    """
    return (src.in_features == tgt.in_features and
            src.out_features == tgt.out_features)


def _apply_zeta_init_to_new_layers(model: Phase2IntegratedModel):
    """
    新規層にゼータ初期化を適用
    
    Args:
        model: Phase2IntegratedModel
    """
    print("\nApplying Zeta initialization to new layers...")
    
    # Phase 2固有の層を識別して初期化
    for name, module in model.named_modules():
        # DissipativeHebbianLayerのLinear層
        if 'hebbian' in name.lower() and isinstance(module, nn.Linear):
            ZetaInitializer.initialize_linear_zeta(module, scale=0.02)
            print(f"  - Initialized {name} with Zeta")
        
        # MemoryResonanceLayerのLinear層
        elif 'resonance' in name.lower() and isinstance(module, nn.Linear):
            ZetaInitializer.initialize_linear_zeta(module, scale=0.02)
            print(f"  - Initialized {name} with Zeta")
        
        # NonHermitianPotentialのLinear層
        elif 'potential' in name.lower() and isinstance(module, nn.Linear):
            ZetaInitializer.initialize_linear_zeta(module, scale=0.02)
            print(f"  - Initialized {name} with Zeta")


def _get_conversion_summary(
    phase1_model: Union[Phase1IntegratedModel, nn.Module],
    phase2_model: Phase2IntegratedModel,
    config: Phase2Config
) -> str:
    """
    変換サマリーを生成
    
    Args:
        phase1_model: Phase 1モデル
        phase2_model: Phase 2モデル
        config: Phase2Config
    
    Returns:
        サマリー文字列
    """
    # パラメータ数を計算
    phase1_params = sum(p.numel() for p in phase1_model.parameters())
    phase2_params = sum(p.numel() for p in phase2_model.parameters())
    param_increase = phase2_params - phase1_params
    param_increase_pct = (param_increase / phase1_params) * 100 if phase1_params > 0 else 0
    
    lines = [
        "",
        "=" * 60,
        "Phase 1 → Phase 2 Conversion Summary",
        "=" * 60,
        "",
        "Parameter Count:",
        f"  - Phase 1: {phase1_params:,}",
        f"  - Phase 2: {phase2_params:,}",
        f"  - Increase: {param_increase:,} ({param_increase_pct:+.1f}%)",
        "",
        "New Components Added:",
        "  - Non-Hermitian Potential (V - iΓ)",
        "  - Dissipative Hebbian Fast Weights",
        "  - SNR-based Memory Filter",
        f"  - Memory Resonance Layer: {'Enabled' if config.resonance_enabled else 'Disabled'}",
        f"  - Zeta Initialization: {'Applied' if config.use_zeta_init else 'Not Applied'}",
        "",
        "Next Steps:",
        "  1. Fine-tune on your dataset to adapt Phase 2 dynamics",
        "  2. Monitor Γ (forgetting rate) evolution during training",
        "  3. Visualize memory resonance patterns",
        "  4. Compare perplexity with Phase 1 baseline",
        "",
        "=" * 60,
    ]
    
    return "\n".join(lines)
