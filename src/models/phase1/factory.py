"""
Phase 1 Model Factory and Integration Layer

このモジュールは、Phase 1の全コンポーネント（AR-SSM, HTT, LNS, Stability Monitor）を
統合し、既存のMUSEモデルアーキテクチャと互換性のあるモデルを生成するファクトリ関数を提供します。

Requirements:
    - 4.1: 既存インフラとの統合
    - 4.4: 後方互換性の実装

Author: Project MUSE Team
"""

from typing import Optional, Dict, Any, Union
import warnings

import torch
import torch.nn as nn

from .config import Phase1Config, Phase1Diagnostics, Phase1TrainingState
from .ar_ssm_layer import AdaptiveRankSemiseparableLayer
from .htt_embedding import (
    HolographicTTEmbedding,
    create_htt_embedding,
    replace_embedding_with_htt,
)
from .lns_linear import LNSLinear, convert_linear_to_lns
from .stability_monitor import BKStabilityMonitor, StabilityThresholds
from .gradient_monitor import GradientMonitor, create_gradient_monitor_from_config
from .errors import InvalidConfigError, check_cuda_available


class Phase1IntegratedModel(nn.Module):
    """
    Phase 1統合モデル
    
    AR-SSM、HTT、LNS、安定性監視を統合したモデルラッパー。
    既存のMUSEモデルアーキテクチャと互換性を保ちながら、
    Phase 1の効率化機能を提供します。
    
    Args:
        base_model: ベースとなるモデル（nn.Module）
        config: Phase1Config
        replace_embeddings: Embeddingを自動的にHTTに置き換えるか
        replace_linears: Linear層を自動的にLNSに置き換えるか（推論専用）
        enable_stability_monitoring: 安定性監視を有効化するか
        enable_gradient_monitoring: 勾配監視を有効化するか
    
    Attributes:
        base_model: ベースモデル
        config: Phase1Config
        stability_monitor: BKStabilityMonitor（有効な場合）
        gradient_monitor: GradientMonitor（有効な場合）
        diagnostics: Phase1Diagnostics
        training_state: Phase1TrainingState
    
    Example:
        >>> from src.models import LanguageModel
        >>> base_model = LanguageModel(vocab_size=50000, d_model=1024, n_layers=12)
        >>> config = Phase1Config.for_hardware(vram_gb=8.0)
        >>> model = Phase1IntegratedModel(base_model, config)
        >>> 
        >>> # Forward pass
        >>> input_ids = torch.randint(0, 50000, (4, 128))
        >>> output, diagnostics = model(input_ids)
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        config: Phase1Config,
        replace_embeddings: bool = True,
        replace_linears: bool = False,
        enable_stability_monitoring: bool = True,
        enable_gradient_monitoring: bool = True,
    ):
        super().__init__()
        
        # Validate config
        config.validate()
        
        self.base_model = base_model
        self.config = config
        
        # Initialize diagnostics and training state
        self.diagnostics = Phase1Diagnostics()
        self.training_state = Phase1TrainingState(
            rank_warmup_steps=1000,
        )
        
        # Replace embeddings with HTT if enabled
        if replace_embeddings and config.htt_enabled:
            self._replace_embeddings_with_htt()
        
        # Replace linear layers with LNS if enabled (inference only)
        if replace_linears and config.lns_enabled:
            if self.training:
                warnings.warn(
                    "LNS is inference-only. Skipping linear layer replacement during training.",
                    UserWarning
                )
            else:
                self._replace_linears_with_lns()
        
        # Initialize stability monitor
        self.stability_monitor = None
        if enable_stability_monitoring and config.stability_monitoring_enabled:
            self.stability_monitor = BKStabilityMonitor(
                thresholds=StabilityThresholds(
                    det_threshold=config.stability_threshold,
                    schatten_s1_bound=config.schatten_s1_bound,
                    schatten_s2_bound=config.schatten_s2_bound,
                )
            )
        
        # Initialize gradient monitor
        self.gradient_monitor = None
        if enable_gradient_monitoring:
            self.gradient_monitor = create_gradient_monitor_from_config(config)
    
    def _replace_embeddings_with_htt(self):
        """
        モデル内のすべてのnn.Embeddingをhtt_embeddingに置き換える
        
        Requirement 4.1: 既存インフラとの統合
        """
        replaced_count = 0
        
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Embedding):
                # Get parent module and attribute name
                parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else ''
                attr_name = name.split('.')[-1]
                
                if parent_name:
                    parent = self.base_model.get_submodule(parent_name)
                else:
                    parent = self.base_model
                
                # Create HTT embedding
                htt_emb = create_htt_embedding(
                    vocab_size=module.num_embeddings,
                    d_model=module.embedding_dim,
                    config=self.config,
                )
                
                # Replace
                setattr(parent, attr_name, htt_emb)
                replaced_count += 1
                
                print(f"Replaced {name} with HolographicTTEmbedding "
                      f"(compression: {htt_emb.get_compression_ratio():.2%})")
        
        if replaced_count == 0:
            warnings.warn(
                "No nn.Embedding layers found in base_model. "
                "HTT embedding replacement skipped.",
                UserWarning
            )
    
    def _replace_linears_with_lns(self):
        """
        モデル内の大きなLinear層をLNSLinearに置き換える（推論専用）
        
        Requirement 4.1: 既存インフラとの統合
        """
        replaced_count = 0
        min_size_threshold = 1024  # Only replace large linear layers
        
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                # Only replace large linear layers
                if module.in_features < min_size_threshold or module.out_features < min_size_threshold:
                    continue
                
                # Get parent module and attribute name
                parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else ''
                attr_name = name.split('.')[-1]
                
                if parent_name:
                    parent = self.base_model.get_submodule(parent_name)
                else:
                    parent = self.base_model
                
                # Create LNS linear
                lns_linear = convert_linear_to_lns(module, self.config)
                
                # Replace
                setattr(parent, attr_name, lns_linear)
                replaced_count += 1
                
                print(f"Replaced {name} with LNSLinear "
                      f"({module.in_features}x{module.out_features})")
        
        if replaced_count == 0:
            warnings.warn(
                f"No large Linear layers (>={min_size_threshold}) found in base_model. "
                "LNS linear replacement skipped.",
                UserWarning
            )
    
    def forward(
        self,
        *args,
        return_diagnostics: bool = True,
        **kwargs
    ) -> Union[torch.Tensor, tuple]:
        """
        Forward pass with Phase 1 diagnostics
        
        Args:
            *args: Positional arguments for base_model
            return_diagnostics: 診断情報を返すか
            **kwargs: Keyword arguments for base_model
        
        Returns:
            output: Base model output
            diagnostics: Phase1Diagnostics (if return_diagnostics=True)
        """
        import time
        
        # Track forward time
        start_time = time.time()
        
        # Forward pass through base model
        output = self.base_model(*args, **kwargs)
        
        # Track forward time
        forward_time_ms = (time.time() - start_time) * 1000
        self.diagnostics.forward_time_ms = forward_time_ms
        
        # Update diagnostics from HTT embeddings
        self._update_htt_diagnostics()
        
        # Update diagnostics from AR-SSM layers
        self._update_ar_ssm_diagnostics()
        
        # Track VRAM usage
        if torch.cuda.is_available():
            self.diagnostics.peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        if return_diagnostics:
            return output, self.diagnostics
        else:
            return output
    
    def _update_htt_diagnostics(self):
        """HTT Embeddingから診断情報を更新"""
        for module in self.base_model.modules():
            if isinstance(module, HolographicTTEmbedding):
                self.diagnostics.htt_compression_ratio = module.get_compression_ratio()
                # Reconstruction error would require additional computation
                # Skip for now to avoid overhead
                break
    
    def _update_ar_ssm_diagnostics(self):
        """AR-SSMレイヤーから診断情報を更新"""
        total_effective_rank = 0.0
        total_gate_sparsity = 0.0
        count = 0
        
        for module in self.base_model.modules():
            if isinstance(module, AdaptiveRankSemiseparableLayer):
                if hasattr(module, '_last_diagnostics'):
                    diag = module._last_diagnostics
                    if 'effective_rank' in diag:
                        total_effective_rank += diag['effective_rank'].item()
                    if 'gates' in diag:
                        # Sparsity: fraction of gates near 0
                        gates = diag['gates']
                        sparsity = (gates < 0.1).float().mean().item()
                        total_gate_sparsity += sparsity
                    count += 1
        
        if count > 0:
            self.diagnostics.ar_ssm_effective_rank = total_effective_rank / count
            self.diagnostics.ar_ssm_gate_sparsity = total_gate_sparsity / count
    
    def update_rank_schedule(self):
        """
        カリキュラム学習のためのランクスケジュールを更新
        
        すべてのAR-SSMレイヤーのcurrent_max_rankを更新します。
        """
        self.training_state.update_rank_schedule(self.config)
        
        # Update all AR-SSM layers
        for module in self.base_model.modules():
            if isinstance(module, AdaptiveRankSemiseparableLayer):
                module.current_max_rank = self.training_state.current_max_rank
    
    def enable_checkpointing(self):
        """すべてのAR-SSMレイヤーでgradient checkpointingを有効化"""
        for module in self.base_model.modules():
            if isinstance(module, AdaptiveRankSemiseparableLayer):
                module.enable_checkpointing()
    
    def disable_checkpointing(self):
        """すべてのAR-SSMレイヤーでgradient checkpointingを無効化"""
        for module in self.base_model.modules():
            if isinstance(module, AdaptiveRankSemiseparableLayer):
                module.disable_checkpointing()
    
    def get_phase1_summary(self) -> str:
        """Phase 1統合の概要を返す"""
        lines = [
            "=== Phase 1 Integrated Model Summary ===",
            "",
            f"Base Model: {self.base_model.__class__.__name__}",
            f"Config: {self.config.__class__.__name__}",
            "",
            "Components:",
        ]
        
        # Count components
        htt_count = sum(1 for m in self.base_model.modules() if isinstance(m, HolographicTTEmbedding))
        ar_ssm_count = sum(1 for m in self.base_model.modules() if isinstance(m, AdaptiveRankSemiseparableLayer))
        lns_count = sum(1 for m in self.base_model.modules() if isinstance(m, LNSLinear))
        
        lines.append(f"  - HTT Embeddings: {htt_count}")
        lines.append(f"  - AR-SSM Layers: {ar_ssm_count}")
        lines.append(f"  - LNS Linear Layers: {lns_count}")
        lines.append(f"  - Stability Monitor: {'Enabled' if self.stability_monitor else 'Disabled'}")
        lines.append(f"  - Gradient Monitor: {'Enabled' if self.gradient_monitor else 'Disabled'}")
        lines.append("")
        lines.append("Configuration:")
        lines.append(f"  - Target VRAM: {self.config.target_vram_gb:.1f} GB")
        lines.append(f"  - AR-SSM Max Rank: {self.config.ar_ssm_max_rank}")
        lines.append(f"  - HTT Rank: {self.config.htt_rank}")
        lines.append(f"  - HTT Compression Target: {self.config.htt_compression_target:.1%}")
        lines.append(f"  - Gradient Checkpointing: {'Enabled' if self.config.use_gradient_checkpointing else 'Disabled'}")
        
        return "\n".join(lines)


def create_phase1_model(
    base_model: Optional[nn.Module] = None,
    config: Optional[Phase1Config] = None,
    vocab_size: Optional[int] = None,
    d_model: Optional[int] = None,
    n_layers: Optional[int] = None,
    model_type: str = "resnet_bk",
    replace_embeddings: bool = True,
    replace_linears: bool = False,
    enable_stability_monitoring: bool = True,
    enable_gradient_monitoring: bool = True,
    **model_kwargs
) -> Phase1IntegratedModel:
    """
    Phase 1統合モデルを作成するファクトリ関数
    
    既存のモデルをPhase 1コンポーネントで拡張するか、
    新しいモデルを作成してPhase 1機能を統合します。
    
    Args:
        base_model: 既存のベースモデル（Noneの場合は新規作成）
        config: Phase1Config（Noneの場合はデフォルト）
        vocab_size: 語彙サイズ（新規作成時）
        d_model: モデル次元（新規作成時）
        n_layers: レイヤー数（新規作成時）
        model_type: モデルタイプ（"resnet_bk", "physics_informed", "koopman"）
        replace_embeddings: Embeddingを自動的にHTTに置き換えるか
        replace_linears: Linear層を自動的にLNSに置き換えるか
        enable_stability_monitoring: 安定性監視を有効化するか
        enable_gradient_monitoring: 勾配監視を有効化するか
        **model_kwargs: ベースモデルへの追加引数
    
    Returns:
        Phase1IntegratedModel
    
    Example:
        >>> # 既存モデルを拡張
        >>> from src.models import LanguageModel
        >>> base = LanguageModel(vocab_size=50000, d_model=1024, n_layers=12)
        >>> model = create_phase1_model(base_model=base)
        >>> 
        >>> # 新規作成
        >>> model = create_phase1_model(
        ...     vocab_size=50000,
        ...     d_model=1024,
        ...     n_layers=12,
        ...     model_type="resnet_bk"
        ... )
        >>> 
        >>> # ハードウェア制約に基づく設定
        >>> config = Phase1Config.for_hardware(vram_gb=8.0)
        >>> model = create_phase1_model(
        ...     vocab_size=50000,
        ...     d_model=1024,
        ...     n_layers=12,
        ...     config=config
        ... )
    
    Requirements: 4.1, 4.4
    """
    # Create default config if not provided
    if config is None:
        config = Phase1Config()
    
    # Create base model if not provided
    if base_model is None:
        if vocab_size is None or d_model is None or n_layers is None:
            raise InvalidConfigError(
                param_name="base_model",
                param_value=None,
                reason="Either base_model or (vocab_size, d_model, n_layers) must be provided"
            )
        
        base_model = _create_base_model(
            model_type=model_type,
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            **model_kwargs
        )
    
    # Create Phase 1 integrated model
    model = Phase1IntegratedModel(
        base_model=base_model,
        config=config,
        replace_embeddings=replace_embeddings,
        replace_linears=replace_linears,
        enable_stability_monitoring=enable_stability_monitoring,
        enable_gradient_monitoring=enable_gradient_monitoring,
    )
    
    # Enable gradient checkpointing if configured
    if config.use_gradient_checkpointing and config.checkpoint_ar_ssm:
        model.enable_checkpointing()
    
    # Print summary
    print(model.get_phase1_summary())
    
    return model


def _create_base_model(
    model_type: str,
    vocab_size: int,
    d_model: int,
    n_layers: int,
    **kwargs
) -> nn.Module:
    """
    ベースモデルを作成
    
    Args:
        model_type: モデルタイプ
        vocab_size: 語彙サイズ
        d_model: モデル次元
        n_layers: レイヤー数
        **kwargs: 追加引数
    
    Returns:
        Base model instance
    """
    if model_type == "resnet_bk":
        from ..resnet_bk import LanguageModel
        return LanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            **kwargs
        )
    elif model_type == "physics_informed":
        from ..physics_informed_layer import PhysicsInformedLanguageModel
        return PhysicsInformedLanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            **kwargs
        )
    elif model_type == "koopman":
        from ..koopman_layer import KoopmanLanguageModel
        return KoopmanLanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            **kwargs
        )
    else:
        raise InvalidConfigError(
            param_name="model_type",
            param_value=model_type,
            reason=f"Unknown model type. Supported: resnet_bk, physics_informed, koopman"
        )


def add_ar_ssm_to_model(
    model: nn.Module,
    config: Phase1Config,
    layer_indices: Optional[list] = None,
) -> nn.Module:
    """
    既存モデルにAR-SSMレイヤーを追加
    
    指定されたレイヤーインデックスにAR-SSMレイヤーを挿入します。
    
    Args:
        model: 対象モデル
        config: Phase1Config
        layer_indices: AR-SSMを追加するレイヤーインデックス（Noneの場合は全レイヤー）
    
    Returns:
        Modified model (in-place)
    
    Example:
        >>> model = LanguageModel(vocab_size=50000, d_model=1024, n_layers=12)
        >>> config = Phase1Config()
        >>> model = add_ar_ssm_to_model(model, config, layer_indices=[0, 3, 6, 9])
    
    Requirements: 4.1, 4.4
    """
    # This is a placeholder for more sophisticated integration
    # In practice, you would need to know the model architecture
    # and insert AR-SSM layers at appropriate positions
    
    warnings.warn(
        "add_ar_ssm_to_model is a placeholder. "
        "Manual integration required for specific model architectures.",
        UserWarning
    )
    
    return model


def convert_model_to_phase1(
    model: nn.Module,
    config: Optional[Phase1Config] = None,
    inplace: bool = False,
) -> nn.Module:
    """
    既存モデルをPhase 1モデルに変換
    
    Embeddingをhtt_embeddingに、大きなLinear層をLNSLinearに置き換えます。
    
    Args:
        model: 変換対象モデル
        config: Phase1Config
        inplace: In-placeで変換するか（Falseの場合はコピーを作成）
    
    Returns:
        Converted model
    
    Example:
        >>> model = LanguageModel(vocab_size=50000, d_model=1024, n_layers=12)
        >>> phase1_model = convert_model_to_phase1(model)
    
    Requirements: 4.1, 4.4
    """
    if config is None:
        config = Phase1Config()
    
    if not inplace:
        import copy
        model = copy.deepcopy(model)
    
    # Replace embeddings
    if config.htt_enabled:
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Embedding):
                parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else ''
                attr_name = name.split('.')[-1]
                
                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                
                htt_emb = create_htt_embedding(
                    vocab_size=module.num_embeddings,
                    d_model=module.embedding_dim,
                    config=config,
                )
                
                setattr(parent, attr_name, htt_emb)
    
    # Replace large linear layers (inference only)
    if config.lns_enabled and not model.training:
        min_size = 1024
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear):
                if module.in_features >= min_size or module.out_features >= min_size:
                    parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else ''
                    attr_name = name.split('.')[-1]
                    
                    if parent_name:
                        parent = model.get_submodule(parent_name)
                    else:
                        parent = model
                    
                    lns_linear = convert_linear_to_lns(module, config)
                    setattr(parent, attr_name, lns_linear)
    
    return model
