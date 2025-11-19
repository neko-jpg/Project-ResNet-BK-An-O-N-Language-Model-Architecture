"""
Complex Number Support Infrastructure for Phase 2 Preparation

Phase 2では非エルミート演算子による「忘却機構」を実装するため、
複素数テンソルのサポートが必要になります。

このモジュールは、Phase 1コンポーネントを複素数対応にするための
ユーティリティ関数を提供します。

Physical Intuition (物理的直観):
    - Phase 2の非エルミート演算子は複素固有値を持つ
    - 複素数の実部: エネルギー（記憶の強度）
    - 複素数の虚部: 減衰率（忘却の速度）
    - exp(-iHt)の時間発展により、自然な忘却機構を実現

Mathematical Foundation:
    - 実数テンソル: R^n
    - 複素数テンソル: C^n = R^n + iR^n
    - 変換: [real, imag] ↔ complex
    - 演算: 複素数対応の線形演算、活性化関数

Requirements:
    - 11.1: 実数↔複素数変換関数
    - 11.2: AR-SSMのdtypeチェックと変換
    - 11.3: HTTの複素位相回転 exp(iθ)
    - 11.4: BK-Coreとの互換性
    - 11.5: 混合実数/複素数演算のサポート

Author: Project MUSE Team
"""

import torch
import torch.nn as nn
from typing import Union, Tuple, Optional
import warnings


def is_complex_tensor(x: torch.Tensor) -> bool:
    """
    テンソルが複素数型かどうかを判定
    
    Args:
        x: 入力テンソル
    
    Returns:
        True if complex dtype (complex64 or complex128)
    
    Example:
        >>> x = torch.randn(10, 20)
        >>> is_complex_tensor(x)  # False
        >>> x_complex = torch.complex(x, torch.zeros_like(x))
        >>> is_complex_tensor(x_complex)  # True
    """
    return x.dtype in (torch.complex64, torch.complex128)


def real_to_complex(
    x: torch.Tensor,
    imag: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """
    実数テンソルを複素数テンソルに変換
    
    Args:
        x: 実数テンソル（実部として使用）
        imag: 虚部テンソル（Noneの場合はゼロ）
        dtype: 出力の複素数型（complex64 or complex128）
    
    Returns:
        複素数テンソル
    
    Example:
        >>> real = torch.randn(10, 20)
        >>> complex_tensor = real_to_complex(real)
        >>> complex_tensor.shape  # (10, 20)
        >>> complex_tensor.dtype  # torch.complex64
    
    Requirement 11.1: Implement utility functions for real ↔ complex conversion
    """
    if is_complex_tensor(x):
        # Already complex, just convert dtype if needed
        return x.to(dtype)
    
    if imag is None:
        imag = torch.zeros_like(x)
    
    if x.shape != imag.shape:
        raise ValueError(
            f"Real and imaginary parts must have same shape: "
            f"real={x.shape}, imag={imag.shape}"
        )
    
    # Create complex tensor
    if dtype == torch.complex64:
        return torch.complex(x.float(), imag.float())
    elif dtype == torch.complex128:
        return torch.complex(x.double(), imag.double())
    else:
        raise ValueError(f"Unsupported complex dtype: {dtype}")


def complex_to_real(
    x: torch.Tensor,
    mode: str = 'concat',
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    複素数テンソルを実数表現に変換
    
    Args:
        x: 複素数テンソル
        mode: 変換モード
            - 'concat': [real, imag]を最後の次元で結合 (default)
            - 'separate': (real, imag)のタプルを返す
            - 'magnitude': 絶対値のみ返す
            - 'phase': 位相のみ返す
    
    Returns:
        実数テンソル（modeに応じて形式が異なる）
    
    Example:
        >>> x = torch.complex(torch.randn(10, 20), torch.randn(10, 20))
        >>> real_concat = complex_to_real(x, mode='concat')
        >>> real_concat.shape  # (10, 20, 2)
        >>> real, imag = complex_to_real(x, mode='separate')
        >>> real.shape, imag.shape  # (10, 20), (10, 20)
    
    Requirement 11.1: Implement utility functions for real ↔ complex conversion
    """
    if not is_complex_tensor(x):
        # Already real
        if mode == 'separate':
            return x, torch.zeros_like(x)
        elif mode == 'concat':
            return torch.stack([x, torch.zeros_like(x)], dim=-1)
        elif mode == 'magnitude':
            return x.abs()
        elif mode == 'phase':
            return torch.zeros_like(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    real_part = x.real
    imag_part = x.imag
    
    if mode == 'concat':
        # Stack along last dimension: (..., D) → (..., D, 2)
        return torch.stack([real_part, imag_part], dim=-1)
    elif mode == 'separate':
        return real_part, imag_part
    elif mode == 'magnitude':
        return torch.abs(x)
    elif mode == 'phase':
        return torch.angle(x)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def ensure_complex(
    x: torch.Tensor,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """
    テンソルが複素数型であることを保証（必要なら変換）
    
    Args:
        x: 入力テンソル
        dtype: 目標複素数型
    
    Returns:
        複素数テンソル
    
    Example:
        >>> x = torch.randn(10, 20)
        >>> x_complex = ensure_complex(x)
        >>> x_complex.dtype  # torch.complex64
    
    Requirement 11.2: Add dtype checking and conversion in AR-SSM
    """
    if is_complex_tensor(x):
        return x.to(dtype)
    else:
        return real_to_complex(x, dtype=dtype)


def ensure_real(
    x: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    テンソルが実数型であることを保証（必要なら変換）
    
    複素数の場合は実部のみを取得します。
    
    Args:
        x: 入力テンソル
        dtype: 目標実数型
    
    Returns:
        実数テンソル
    
    Example:
        >>> x = torch.complex(torch.randn(10, 20), torch.randn(10, 20))
        >>> x_real = ensure_real(x)
        >>> x_real.dtype  # torch.float32
    
    Requirement 11.2: Add dtype checking and conversion in AR-SSM
    """
    if is_complex_tensor(x):
        return x.real.to(dtype)
    else:
        return x.to(dtype)


def complex_phase_rotation(
    x: torch.Tensor,
    phase: torch.Tensor,
    use_full_complex: bool = True,
) -> torch.Tensor:
    """
    複素位相回転を適用: x * exp(iθ)
    
    Phase 2では完全な複素位相回転を使用します。
    Phase 1では実数近似（cos(θ)による振幅変調）を使用できます。
    
    Args:
        x: 入力テンソル (..., D)
        phase: 位相パラメータ (D,) or scalar
        use_full_complex: True=exp(iθ), False=cos(θ)近似
    
    Returns:
        位相回転後のテンソル
    
    Example:
        >>> x = torch.randn(10, 20, 128)
        >>> phase = torch.randn(128)
        >>> # Phase 2: 完全な複素回転
        >>> x_rotated = complex_phase_rotation(x, phase, use_full_complex=True)
        >>> # Phase 1: 実数近似
        >>> x_approx = complex_phase_rotation(x, phase, use_full_complex=False)
    
    Requirement 11.3: Implement complex-valued phase rotation in HTT (exp(iθ))
    """
    if use_full_complex:
        # 完全な複素位相回転: x * exp(iθ)
        # exp(iθ) = cos(θ) + i*sin(θ)
        phase_factor = torch.exp(1j * phase)
        
        # 入力が実数の場合は複素数に変換
        if not is_complex_tensor(x):
            x = ensure_complex(x)
        
        # ブロードキャスト: phase_factor (D,) → (..., D)
        return x * phase_factor
    else:
        # 実数近似: x * cos(θ)
        # Phase 1で使用（既存のHTT実装との互換性）
        phase_mod = torch.cos(phase)
        return x * phase_mod


def check_dtype_compatibility(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    operation: str = "operation",
) -> bool:
    """
    2つのテンソルのdtype互換性をチェック
    
    Args:
        tensor1: 第1テンソル
        tensor2: 第2テンソル
        operation: 演算名（エラーメッセージ用）
    
    Returns:
        True if compatible
    
    Raises:
        TypeError: 互換性がない場合
    
    Example:
        >>> x = torch.randn(10, 20)
        >>> y = torch.complex(torch.randn(10, 20), torch.randn(10, 20))
        >>> check_dtype_compatibility(x, y, "addition")  # Raises TypeError
    
    Requirement 11.4: Ensure no dtype conflicts in integration
    """
    t1_complex = is_complex_tensor(tensor1)
    t2_complex = is_complex_tensor(tensor2)
    
    if t1_complex != t2_complex:
        raise TypeError(
            f"Dtype mismatch in {operation}: "
            f"tensor1.dtype={tensor1.dtype}, tensor2.dtype={tensor2.dtype}. "
            f"Cannot mix real and complex tensors without explicit conversion."
        )
    
    return True


def safe_complex_operation(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    operation: str = "add",
    auto_convert: bool = True,
) -> torch.Tensor:
    """
    安全な複素数演算（自動型変換付き）
    
    Args:
        tensor1: 第1テンソル
        tensor2: 第2テンソル
        operation: 演算タイプ ('add', 'mul', 'matmul')
        auto_convert: 自動的に複素数に変換するか
    
    Returns:
        演算結果
    
    Example:
        >>> x = torch.randn(10, 20)  # Real
        >>> y = torch.complex(torch.randn(10, 20), torch.randn(10, 20))  # Complex
        >>> result = safe_complex_operation(x, y, operation='add')
        >>> result.dtype  # torch.complex64
    
    Requirement 11.5: Mixed real/complex operations support
    """
    t1_complex = is_complex_tensor(tensor1)
    t2_complex = is_complex_tensor(tensor2)
    
    # 両方実数 → そのまま演算
    if not t1_complex and not t2_complex:
        if operation == 'add':
            return tensor1 + tensor2
        elif operation == 'mul':
            return tensor1 * tensor2
        elif operation == 'matmul':
            return torch.matmul(tensor1, tensor2)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    # 両方複素数 → そのまま演算
    if t1_complex and t2_complex:
        if operation == 'add':
            return tensor1 + tensor2
        elif operation == 'mul':
            return tensor1 * tensor2
        elif operation == 'matmul':
            return torch.matmul(tensor1, tensor2)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    # 混在 → 自動変換
    if auto_convert:
        if not t1_complex:
            tensor1 = ensure_complex(tensor1)
        if not t2_complex:
            tensor2 = ensure_complex(tensor2)
        
        if operation == 'add':
            return tensor1 + tensor2
        elif operation == 'mul':
            return tensor1 * tensor2
        elif operation == 'matmul':
            return torch.matmul(tensor1, tensor2)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    else:
        raise TypeError(
            f"Cannot perform {operation} on mixed real/complex tensors "
            f"without auto_convert=True"
        )


class ComplexLinear(nn.Module):
    """
    複素数対応のLinear層
    
    Phase 2で使用する複素数重みを持つ線形層。
    実部と虚部を独立に学習します。
    
    Args:
        in_features: 入力次元
        out_features: 出力次元
        bias: バイアス項を使用するか
    
    Example:
        >>> layer = ComplexLinear(128, 256)
        >>> x = torch.complex(torch.randn(10, 128), torch.randn(10, 128))
        >>> y = layer(x)
        >>> y.shape  # (10, 256)
        >>> y.dtype  # torch.complex64
    
    Requirement 11.4: BirmanSchwingerCore complex-valued operations compatibility
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.complex64,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # 複素数重み: W = W_real + i*W_imag
        self.weight_real = nn.Parameter(
            torch.randn(out_features, in_features) * 0.02
        )
        self.weight_imag = nn.Parameter(
            torch.randn(out_features, in_features) * 0.02
        )
        
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_features))
            self.bias_imag = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
        
        self.dtype = dtype
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with complex matrix multiplication
        
        Args:
            x: (..., in_features) 複素数または実数テンソル
        
        Returns:
            y: (..., out_features) 複素数テンソル
        """
        # 入力を複素数に変換
        if not is_complex_tensor(x):
            x = ensure_complex(x, dtype=self.dtype)
        
        # 重みを複素数に変換
        weight = torch.complex(self.weight_real, self.weight_imag)
        
        # 複素数行列積: y = x @ W^T
        y = torch.matmul(x, weight.T)
        
        # バイアス項
        if self.bias_real is not None:
            bias = torch.complex(self.bias_real, self.bias_imag)
            y = y + bias
        
        return y


def document_complex_support() -> dict:
    """
    Phase 1コンポーネントの複素数サポート状況を文書化
    
    Returns:
        dict: 各コンポーネントのサポート状況
    
    Requirement 11.3: Document which operations support complex tensors
    """
    return {
        'AR-SSM Layer': {
            'complex_input_support': 'Partial',
            'status': 'Real-valued only in Phase 1',
            'phase2_ready': True,
            'notes': (
                'AR-SSM can accept complex inputs but will convert to real internally. '
                'Full complex support (complex gates, complex projections) will be '
                'added in Phase 2.'
            ),
        },
        'HTT Embedding': {
            'complex_input_support': 'No',
            'status': 'Real-valued phase rotation (cos(θ)) in Phase 1',
            'phase2_ready': True,
            'notes': (
                'HTT uses real-valued phase rotation (cos(θ)) in Phase 1. '
                'Phase 2 will upgrade to full complex phase rotation exp(iθ) '
                'using complex_phase_rotation() with use_full_complex=True.'
            ),
        },
        'LNS Kernel': {
            'complex_input_support': 'No',
            'status': 'Real-valued only',
            'phase2_ready': False,
            'notes': (
                'LNS kernel operates in logarithmic domain which is inherently '
                'real-valued. Complex number support is not planned. '
                'LNS will be disabled when complex operations are needed.'
            ),
        },
        'BK-Core': {
            'complex_input_support': 'Yes',
            'status': 'Full complex support (complex64/complex128)',
            'phase2_ready': True,
            'notes': (
                'BirmanSchwingerCore already supports complex-valued operations. '
                'Uses complex128 internally for numerical stability, '
                'outputs complex64 for memory efficiency.'
            ),
        },
        'Stability Monitor': {
            'complex_input_support': 'Yes',
            'status': 'Complex-aware stability checks',
            'phase2_ready': True,
            'notes': (
                'Stability monitor can handle complex-valued resolvent G_ii. '
                'Determinant and eigenvalue computations work with complex tensors.'
            ),
        },
        'Fused Scan Kernel': {
            'complex_input_support': 'Partial',
            'status': 'Real-valued in Phase 1, complex support possible',
            'phase2_ready': True,
            'notes': (
                'Triton kernel currently operates on real tensors. '
                'Can be extended to complex by treating as 2-channel real tensor. '
                'Fallback torch.cumsum already supports complex.'
            ),
        },
    }


def get_complex_conversion_guide() -> str:
    """
    Phase 2移行のための複素数変換ガイドを返す
    
    Returns:
        str: 変換ガイドのテキスト
    
    Requirement 11.5: Provide utility functions for Phase 2 integration
    """
    guide = """
    Phase 2 Complex Number Migration Guide
    ======================================
    
    1. AR-SSM Layer
       - Current: Real-valued gates and projections
       - Phase 2: Complex-valued gates for non-Hermitian dynamics
       - Migration:
         * Replace gate network output with complex_phase_rotation()
         * Use ComplexLinear for U_proj and V_proj
         * Update forward() to handle complex inputs
    
    2. HTT Embedding
       - Current: cos(θ) phase modulation (real approximation)
       - Phase 2: exp(iθ) full complex phase rotation
       - Migration:
         * Change phase_encoding to use complex_phase_rotation(use_full_complex=True)
         * Update einsum contraction to handle complex cores
         * Output will be complex, add ensure_real() if needed downstream
    
    3. Integration with BK-Core
       - BK-Core already outputs complex tensors (G_ii: complex64)
       - Ensure AR-SSM can accept complex inputs from BK-Core
       - Use safe_complex_operation() for mixed operations
    
    4. Testing Strategy
       - Test each component with complex inputs independently
       - Test integration: BK-Core → AR-SSM → HTT
       - Verify gradient flow through complex operations
       - Check numerical stability with complex eigenvalues
    
    5. Backward Compatibility
       - Keep real-valued mode as default (phase1_mode=True)
       - Add complex_mode flag to enable Phase 2 features
       - Provide conversion utilities for existing checkpoints
    """
    return guide


__all__ = [
    # Type checking
    'is_complex_tensor',
    
    # Conversion functions
    'real_to_complex',
    'complex_to_real',
    'ensure_complex',
    'ensure_real',
    
    # Complex operations
    'complex_phase_rotation',
    'check_dtype_compatibility',
    'safe_complex_operation',
    
    # Complex layers
    'ComplexLinear',
    
    # Documentation
    'document_complex_support',
    'get_complex_conversion_guide',
]