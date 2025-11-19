"""
Phase 1 Efficiency Engine - Custom Exception Classes

このモジュールは、Phase 1コンポーネントで発生する可能性のある
エラーに対するカスタム例外クラスを定義します。

各例外クラスは、エラーの診断情報と実行可能な解決策を提供します。
"""

from typing import List, Dict, Any, Optional


class VRAMExhaustedError(RuntimeError):
    """
    VRAM使用量が安全な閾値を超えた場合に発生する例外。
    
    この例外は、GPUメモリが不足している場合に発生し、
    メモリ使用量を削減するための具体的な提案を提供します。
    
    Attributes:
        current_mb: 現在のVRAM使用量（MB）
        limit_mb: VRAM制限値（MB）
        suggestions: メモリ削減のための提案リスト
    
    Requirements: 5.3, 10.4
    """
    
    def __init__(
        self,
        current_mb: float,
        limit_mb: float,
        suggestions: List[str]
    ):
        """
        VRAMExhaustedErrorを初期化します。
        
        Args:
            current_mb: 現在のVRAM使用量（MB）
            limit_mb: VRAM制限値（MB）
            suggestions: メモリ削減のための提案リスト
        """
        self.current_mb = current_mb
        self.limit_mb = limit_mb
        self.suggestions = suggestions
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """エラーメッセージをフォーマットします。"""
        msg = f"VRAM exhausted: {self.current_mb:.1f}MB / {self.limit_mb:.1f}MB\n"
        msg += f"Usage: {(self.current_mb / self.limit_mb * 100):.1f}%\n\n"
        msg += "Suggestions to reduce memory usage:\n"
        for i, suggestion in enumerate(self.suggestions, 1):
            msg += f"  {i}. {suggestion}\n"
        return msg


class NumericalInstabilityError(RuntimeError):
    """
    数値的不安定性が検出された場合に発生する例外。
    
    この例外は、NaN/Inf値の出現、勾配爆発、または
    Birman-Schwinger演算子の特異性などの数値的問題を示します。
    
    Attributes:
        component: 不安定性が検出されたコンポーネント名
        diagnostics: 診断情報の辞書
    
    Requirements: 5.3, 10.4
    """
    
    def __init__(
        self,
        component: str,
        diagnostics: Dict[str, Any]
    ):
        """
        NumericalInstabilityErrorを初期化します。
        
        Args:
            component: 不安定性が検出されたコンポーネント名
            diagnostics: 診断情報の辞書
        """
        self.component = component
        self.diagnostics = diagnostics
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """エラーメッセージをフォーマットします。"""
        msg = f"Numerical instability detected in {self.component}\n\n"
        msg += "Diagnostics:\n"
        for key, value in self.diagnostics.items():
            if isinstance(value, float):
                msg += f"  {key}: {value:.6e}\n"
            else:
                msg += f"  {key}: {value}\n"
        msg += "\nRecommended actions:\n"
        msg += "  1. Check input data for NaN/Inf values\n"
        msg += "  2. Reduce learning rate (current LR may be too high)\n"
        msg += "  3. Enable gradient clipping (max_norm=1.0 recommended)\n"
        msg += "  4. Increase stability_threshold in Phase1Config\n"
        msg += "  5. Reduce AR-SSM max_rank to improve conditioning\n"
        return msg


class InvalidConfigError(ValueError):
    """
    Phase1Configの検証が失敗した場合に発生する例外。
    
    この例外は、設定パラメータが無効な値を持つ場合、
    または矛盾する設定の組み合わせが検出された場合に発生します。
    
    Attributes:
        param_name: 無効なパラメータ名
        param_value: 無効なパラメータ値
        reason: 無効である理由
    
    Requirements: 5.3, 10.4
    """
    
    def __init__(
        self,
        param_name: str,
        param_value: Any,
        reason: str
    ):
        """
        InvalidConfigErrorを初期化します。
        
        Args:
            param_name: 無効なパラメータ名
            param_value: 無効なパラメータ値
            reason: 無効である理由
        """
        self.param_name = param_name
        self.param_value = param_value
        self.reason = reason
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """エラーメッセージをフォーマットします。"""
        msg = f"Invalid configuration parameter: '{self.param_name}'\n"
        msg += f"  Value: {self.param_value}\n"
        msg += f"  Reason: {self.reason}\n\n"
        msg += "Please check the Phase1Config documentation for valid parameter ranges.\n"
        return msg


class HardwareCompatibilityError(RuntimeError):
    """
    ハードウェア要件が満たされていない場合に発生する例外。
    
    この例外は、CUDA非対応環境、Tritonライブラリの欠落、
    または計算能力の不足などのハードウェア互換性問題を示します。
    
    Attributes:
        required: 必要なハードウェア/ソフトウェア
        available: 利用可能なハードウェア/ソフトウェア
        fallback: 代替手段（存在する場合）
    
    Requirements: 5.3, 10.4
    """
    
    def __init__(
        self,
        required: str,
        available: str,
        fallback: Optional[str] = None
    ):
        """
        HardwareCompatibilityErrorを初期化します。
        
        Args:
            required: 必要なハードウェア/ソフトウェア
            available: 利用可能なハードウェア/ソフトウェア
            fallback: 代替手段（存在する場合）
        """
        self.required = required
        self.available = available
        self.fallback = fallback
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """エラーメッセージをフォーマットします。"""
        msg = "Hardware compatibility requirement not met\n\n"
        msg += f"  Required: {self.required}\n"
        msg += f"  Available: {self.available}\n"
        
        if self.fallback:
            msg += f"\n  Fallback option: {self.fallback}\n"
        else:
            msg += "\n  No fallback available. Please upgrade your hardware/software.\n"
        
        return msg


# 便利な例外生成関数

def raise_vram_exhausted(
    current_mb: float,
    limit_mb: float,
    batch_size: Optional[int] = None,
    seq_len: Optional[int] = None,
    ar_ssm_enabled: bool = True,
    lns_enabled: bool = False,
    gradient_checkpointing: bool = False
) -> None:
    """
    VRAM不足エラーを発生させます（コンテキストに応じた提案付き）。
    
    Args:
        current_mb: 現在のVRAM使用量（MB）
        limit_mb: VRAM制限値（MB）
        batch_size: 現在のバッチサイズ
        seq_len: 現在のシーケンス長
        ar_ssm_enabled: AR-SSMが有効かどうか
        lns_enabled: LNSカーネルが有効かどうか
        gradient_checkpointing: 勾配チェックポイントが有効かどうか
    """
    suggestions = []
    
    # バッチサイズの削減を提案
    if batch_size and batch_size > 1:
        new_batch_size = max(1, batch_size // 2)
        suggestions.append(
            f"Reduce batch_size from {batch_size} to {new_batch_size}"
        )
    
    # シーケンス長の削減を提案
    if seq_len and seq_len > 512:
        new_seq_len = max(512, seq_len // 2)
        suggestions.append(
            f"Reduce sequence length from {seq_len} to {new_seq_len}"
        )
    
    # 勾配チェックポイントの有効化を提案
    if not gradient_checkpointing:
        suggestions.append(
            "Enable gradient checkpointing (set use_gradient_checkpointing=True)"
        )
    
    # AR-SSMランクの削減を提案
    if ar_ssm_enabled:
        suggestions.append(
            "Reduce ar_ssm_max_rank (e.g., from 32 to 16)"
        )
    
    # LNSカーネルの無効化を提案
    if lns_enabled:
        suggestions.append(
            "Disable LNS kernel (set lns_enabled=False)"
        )
    
    # 混合精度の使用を提案
    suggestions.append(
        "Use mixed precision training (torch.cuda.amp.autocast)"
    )
    
    # 一般的な提案
    suggestions.append(
        "Use a GPU with more VRAM (e.g., RTX 3090 24GB)"
    )
    
    raise VRAMExhaustedError(current_mb, limit_mb, suggestions)


def raise_numerical_instability(
    component: str,
    has_nan: bool = False,
    has_inf: bool = False,
    max_value: Optional[float] = None,
    min_value: Optional[float] = None,
    gradient_norm: Optional[float] = None,
    det_condition: Optional[float] = None,
    **extra_diagnostics
) -> None:
    """
    数値的不安定性エラーを発生させます。
    
    Args:
        component: 不安定性が検出されたコンポーネント名
        has_nan: NaN値が存在するかどうか
        has_inf: Inf値が存在するかどうか
        max_value: テンソルの最大値
        min_value: テンソルの最小値
        gradient_norm: 勾配ノルム
        det_condition: Birman-Schwinger演算子の行列式条件
        **extra_diagnostics: 追加の診断情報
    """
    diagnostics = {}
    
    if has_nan:
        diagnostics['has_nan'] = True
    if has_inf:
        diagnostics['has_inf'] = True
    if max_value is not None:
        diagnostics['max_value'] = max_value
    if min_value is not None:
        diagnostics['min_value'] = min_value
    if gradient_norm is not None:
        diagnostics['gradient_norm'] = gradient_norm
    if det_condition is not None:
        diagnostics['det_condition'] = det_condition
    
    # 追加の診断情報をマージ
    diagnostics.update(extra_diagnostics)
    
    raise NumericalInstabilityError(component, diagnostics)


def check_cuda_available(require_triton: bool = False) -> None:
    """
    CUDAとTritonの利用可能性をチェックします。
    
    Args:
        require_triton: Tritonライブラリが必要かどうか
    
    Raises:
        HardwareCompatibilityError: CUDAまたはTritonが利用できない場合
    """
    import torch
    
    if not torch.cuda.is_available():
        raise HardwareCompatibilityError(
            required="CUDA-capable GPU",
            available="CPU only",
            fallback="Some Phase 1 features will use CPU fallback (slower)"
        )
    
    if require_triton:
        try:
            import triton
        except ImportError:
            raise HardwareCompatibilityError(
                required="Triton library for custom kernels",
                available="Not installed",
                fallback="Install with: pip install triton"
            )
