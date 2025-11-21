"""
Complex Operations for Phase 3

このモジュールは、複素数ニューラルネットワークの基本演算を実装します。

Components:
    - ComplexLinear: 複素線形層
    - ModReLU: 位相保存活性化関数
    - ComplexLayerNorm: 複素正規化層

Physical Intuition:
    複素数ニューラルネットワークは、実部（振幅）と虚部（位相）を独立かつ相互作用させながら処理します。
    これにより、否定形・皮肉・多義語などの干渉効果を量子力学的にモデリングできます。

Requirements:
    - Requirement 1.5: ComplexLinearの基本構造
    - Requirement 1.6: 複素行列積の実装
    - Requirement 1.7: Xavier初期化（複素数版）
    - Requirement 1.9: ModReLU活性化関数
    - Requirement 1.11: ComplexLayerNorm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union, Optional

from .complex_tensor import ComplexTensor


class ComplexLinear(nn.Module):
    """
    複素線形層（Complex Linear Layer）
    
    この層は、複素重み行列を使用して複素入力を変換します。
    Planar形式のメモリレイアウトに最適化されています。
    
    Mathematical Formula:
        W = A + iB (複素重み行列)
        z = x + iy (複素入力)
        Wz = (Ax - By) + i(Bx + Ay)
    
    Physical Intuition:
        複素線形変換は、振幅と位相の両方を変換します。
        これにより、言語の意味（振幅）と文脈（位相）を同時に処理できます。
    
    Memory Layout:
        Planar形式を採用:
        - weight_real: (out_features, in_features) float16
        - weight_imag: (out_features, in_features) float16
        - bias_real: (out_features,) float16
        - bias_imag: (out_features,) float16
    
    Memory Efficiency:
        - use_complex32=True: 4 bytes/element (50%削減)
        - use_complex32=False: 8 bytes/element (complex64相当)
    
    Args:
        in_features (int): 入力特徴量の次元数
        out_features (int): 出力特徴量の次元数
        bias (bool): バイアス項を使用するか（デフォルト: True）
        use_complex32 (bool): float16を使用するか（デフォルト: True）
    
    Attributes:
        weight_real (nn.Parameter): 重み行列の実部
        weight_imag (nn.Parameter): 重み行列の虚部
        bias_real (nn.Parameter): バイアスの実部
        bias_imag (nn.Parameter): バイアスの虚部
    
    Examples:
        >>> # ComplexTensor入力
        >>> layer = ComplexLinear(64, 128)
        >>> x = ComplexTensor(torch.randn(4, 10, 64, dtype=torch.float16),
        ...                   torch.randn(4, 10, 64, dtype=torch.float16))
        >>> y = layer(x)
        >>> print(y.shape)  # torch.Size([4, 10, 128])
        
        >>> # complex64入力（互換性）
        >>> x_complex64 = torch.randn(4, 10, 64, dtype=torch.complex64)
        >>> y_complex64 = layer(x_complex64)
        >>> print(y_complex64.dtype)  # torch.complex64
    
    Requirements:
        - Requirement 1.5: weight_real, weight_imag, bias_real, bias_imagパラメータ
        - Requirement 1.6: 複素行列積 (A + iB)(x + iy) = (Ax - By) + i(Bx + Ay)
        - Requirement 1.7: Xavier初期化（複素数版）
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        use_complex32: bool = True
    ):
        """
        ComplexLinearの初期化
        
        Args:
            in_features: 入力特徴量の次元数
            out_features: 出力特徴量の次元数
            bias: バイアス項を使用するか
            use_complex32: float16を使用するか（メモリ効率化）
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_complex32 = use_complex32
        
        # Requirement 1.5: Planar形式のパラメータ定義
        # 実部と虚部を分離して保持
        dtype = torch.float16 if use_complex32 else torch.float32
        
        # 重み行列: (out_features, in_features)
        self.weight_real = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype)
        )
        self.weight_imag = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype)
        )
        
        # バイアス: (out_features,)
        if bias:
            self.bias_real = nn.Parameter(
                torch.zeros(out_features, dtype=dtype)
            )
            self.bias_imag = nn.Parameter(
                torch.zeros(out_features, dtype=dtype)
            )
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
        
        # Requirement 1.7: Xavier初期化（複素数版）
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        パラメータの初期化（Xavier初期化の複素数版）
        
        Xavier初期化の複素数版:
            複素数の大きさが適切な範囲に収まるように、実部と虚部を独立に初期化します。
            
            標準偏差 = √(2 / (in_features + out_features))
            
            この初期化により、複素数の大きさの期待値が1に近くなります。
        
        Physical Intuition:
            初期状態では、振幅と位相がランダムに分布します。
            これにより、学習初期から多様な表現が可能になります。
        
        Numerical Stability:
            - 実部と虚部を独立に初期化することで、複素数の大きさが爆発/消失しない
            - バイアスはゼロ初期化（標準的な手法）
        
        Requirements:
            - Requirement 1.7: Xavier初期化を実部と虚部に適用
        """
        # Xavier uniform初期化
        # 複素数の場合、実部と虚部を独立に初期化
        # 標準偏差を√2で割ることで、複素数の大きさの期待値を調整
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        
        # 実部の初期化
        nn.init.uniform_(
            self.weight_real,
            -std,
            std
        )
        
        # 虚部の初期化
        nn.init.uniform_(
            self.weight_imag,
            -std,
            std
        )
        
        # バイアスはゼロ初期化（既に__init__で実施済み）
        # 複素数の場合、実部と虚部の両方をゼロにする
    
    def forward(
        self,
        z: Union[ComplexTensor, torch.Tensor]
    ) -> Union[ComplexTensor, torch.Tensor]:
        """
        順伝播（Forward Pass）
        
        Mathematical Formula:
            (A + iB)(x + iy) = (Ax - By) + i(Bx + Ay)
            
            where:
                A = weight_real
                B = weight_imag
                x = input.real
                y = input.imag
        
        Implementation Strategy:
            1. 入力の型を判定（ComplexTensor or complex64）
            2. 実部と虚部を抽出
            3. 複素行列積を計算
            4. バイアスを加算
            5. 入力と同じ型で返す
        
        Numerical Stability:
            - float16の場合、中間計算でオーバーフローを防ぐため、
              必要に応じてfloat32にキャストする
            - ただし、メモリ効率を優先し、基本的にはfloat16で計算
        
        Args:
            z: 複素入力
                - ComplexTensor: (B, N, in_features)
                - torch.complex64: (B, N, in_features)
        
        Returns:
            複素出力（入力と同じ型）
                - ComplexTensor: (B, N, out_features)
                - torch.complex64: (B, N, out_features)
        
        Requirements:
            - Requirement 1.6: 複素行列積の実装
        """
        # Requirement 1.6: ComplexTensorとcomplex64の両方に対応
        
        # 入力の型を判定
        if isinstance(z, ComplexTensor):
            # ComplexTensor入力
            x = z.real  # (B, N, in_features)
            y = z.imag  # (B, N, in_features)
            return_complex_tensor = True
        elif z.is_complex():
            # PyTorch complex64入力
            x = z.real  # (B, N, in_features)
            y = z.imag  # (B, N, in_features)
            return_complex_tensor = False
        else:
            raise TypeError(
                f"Input must be ComplexTensor or complex type, got {type(z)}"
            )
        
        # 複素行列積: (A + iB)(x + iy) = (Ax - By) + i(Bx + Ay)
        
        # 実部の計算: Ax - By
        # F.linear(x, W) = xW^T
        out_real = F.linear(x, self.weight_real, self.bias_real)  # Ax + bias_real
        out_real = out_real - F.linear(y, self.weight_imag, None)  # Ax - By + bias_real
        
        # 虚部の計算: Bx + Ay
        out_imag = F.linear(x, self.weight_imag, self.bias_imag)  # Bx + bias_imag
        out_imag = out_imag + F.linear(y, self.weight_real, None)  # Bx + Ay + bias_imag
        
        # 出力の型を入力に合わせる
        if return_complex_tensor:
            # ComplexTensor形式で返す
            if self.use_complex32:
                # float16のまま返す
                return ComplexTensor(out_real, out_imag)
            else:
                # float32の場合もComplexTensorで返す
                return ComplexTensor(out_real, out_imag)
        else:
            # complex64形式で返す
            return torch.complex(out_real, out_imag)
    
    def extra_repr(self) -> str:
        """モジュールの文字列表現"""
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'bias={self.bias_real is not None}, '
            f'use_complex32={self.use_complex32}'
        )


class ModReLU(nn.Module):
    """
    ModReLU活性化関数（位相保存活性化関数）
    
    ModReLUは、複素数の振幅をフィルタリングしながら、位相を保存する活性化関数です。
    
    Mathematical Formula:
        z' = ReLU(|z| + b) · z / |z|
        
        where:
            |z| = √(real² + imag²)  (振幅)
            z / |z|  (位相を保存)
    
    Physical Intuition:
        - 振幅: 情報の「強さ」を表す → ReLUでフィルタリング
        - 位相: 情報の「方向性」を表す → 保存
        
        これにより、弱い信号を抑制しながら、情報の方向性（文脈）を保持できます。
    
    Numerical Stability:
        - ゼロ除算対策: |z|にイプシロン（1e-6）を加算
        - アンダーフロー対策: float32で計算してからfloat16に戻す
    
    Args:
        features (int): 特徴量の次元数
        use_half (bool): float16を使用するか（デフォルト: True）
    
    Attributes:
        bias (nn.Parameter): バイアスパラメータ（学習可能）
    
    Examples:
        >>> modrelu = ModReLU(64)
        >>> z = ComplexTensor(torch.randn(4, 10, 64, dtype=torch.float16),
        ...                   torch.randn(4, 10, 64, dtype=torch.float16))
        >>> z_activated = modrelu(z)
        >>> # 位相が保存されていることを確認
        >>> phase_before = z.angle()
        >>> phase_after = z_activated.angle()
        >>> print(torch.allclose(phase_before, phase_after, atol=1e-3))  # True
    
    Requirements:
        - Requirement 1.9: ModReLU数式の実装
        - Requirement 1.10: 位相保存の検証
    """
    
    def __init__(self, features: int, use_half: bool = True):
        """
        ModReLUの初期化
        
        Args:
            features: 特徴量の次元数
            use_half: float16を使用するか
        """
        super().__init__()
        dtype = torch.float16 if use_half else torch.float32
        
        # バイアスパラメータ（学習可能）
        # 初期値はゼロ（標準的なReLUと同じ挙動）
        self.bias = nn.Parameter(torch.zeros(features, dtype=dtype))
    
    def forward(
        self,
        z: Union[ComplexTensor, torch.Tensor]
    ) -> Union[ComplexTensor, torch.Tensor]:
        """
        順伝播（Forward Pass）
        
        Formula:
            z' = ReLU(|z| + b) · z / |z|
        
        Steps:
            1. 振幅を計算: |z| = √(real² + imag²)
            2. 位相を計算: z / |z|
            3. 振幅をフィルタリング: ReLU(|z| + b)
            4. 新しい振幅と位相を合成: new_mag · phase
        
        Numerical Stability:
            - ゼロ除算対策: |z|にイプシロン（1e-6）を加算
            - 位相保存: z / |z|により単位複素数を計算
        
        Args:
            z: 複素入力（ComplexTensor or complex64）
        
        Returns:
            複素出力（入力と同じ型）
        
        Requirements:
            - Requirement 1.9: z' = ReLU(|z| + b) · z / |z|
        """
        # 入力の型を判定
        if isinstance(z, ComplexTensor):
            # ComplexTensor入力
            # Requirement 1.9: 振幅を計算
            mag = z.abs()  # 振幅: |z| = √(real² + imag²)
            
            # ゼロ除算対策: イプシロンを加算
            mag_safe = mag + 1e-6
            
            # 位相を計算: z / |z|（単位複素数）
            # ComplexTensorの除算: real/mag, imag/mag
            phase = ComplexTensor(
                z.real / mag_safe,
                z.imag / mag_safe
            )
            
            # 振幅のフィルタリング: ReLU(|z| + b)
            new_mag = F.relu(mag + self.bias)
            
            # 新しい複素数を合成: new_mag · phase
            # ComplexTensorのスカラー乗算
            return ComplexTensor(
                phase.real * new_mag,
                phase.imag * new_mag
            )
        
        elif z.is_complex():
            # PyTorch complex64入力
            mag = torch.abs(z)  # 振幅
            
            # ゼロ除算対策
            mag_safe = mag + 1e-6
            
            # 位相: z / |z|
            phase = z / mag_safe
            
            # 振幅のフィルタリング
            new_mag = F.relu(mag + self.bias)
            
            # 新しい複素数を合成
            return new_mag * phase
        
        else:
            raise TypeError(
                f"Input must be ComplexTensor or complex type, got {type(z)}"
            )


class ComplexLayerNorm(nn.Module):
    """
    複素正規化層（Complex Layer Normalization）
    
    ComplexLayerNormは、複素平面上で正規化を行います。
    
    Mathematical Formula:
        z' = γ · (z - μ) / √(σ² + ε) + β
        
        where:
            μ = E[z]  (複素平均)
            σ² = E[|z - μ|²]  (複素分散)
            γ, β: 学習可能なアフィン変換パラメータ（実数）
    
    Physical Intuition:
        複素正規化は、振幅と位相の両方を正規化します。
        これにより、学習の安定性が向上し、勾配消失/爆発を防ぎます。
    
    Numerical Stability:
        - ゼロ除算対策: 分散にイプシロン（eps）を加算
        - オーバーフロー対策: float32で計算してからfloat16に戻す
    
    Args:
        normalized_shape (int or tuple): 正規化する次元の形状
        eps (float): 数値安定性のためのイプシロン（デフォルト: 1e-5）
        elementwise_affine (bool): アフィン変換を使用するか（デフォルト: True）
    
    Attributes:
        gamma (nn.Parameter): スケールパラメータ（実数）
        beta (nn.Parameter): シフトパラメータ（実数）
    
    Examples:
        >>> norm = ComplexLayerNorm(64)
        >>> z = ComplexTensor(torch.randn(4, 10, 64, dtype=torch.float16),
        ...                   torch.randn(4, 10, 64, dtype=torch.float16))
        >>> z_norm = norm(z)
        >>> # 正規化後の平均が0、分散が1に近いことを確認
        >>> mean = z_norm.mean(dim=(0, 1))
        >>> print(mean.abs().mean())  # ~0.0
    
    Requirements:
        - Requirement 1.11: 複素平均と複素分散の計算
        - Requirement 1.12: アフィン変換（実数パラメータ）
    """
    
    def __init__(
        self,
        normalized_shape: Union[int, tuple],
        eps: float = 1e-5,
        elementwise_affine: bool = True
    ):
        """
        ComplexLayerNormの初期化
        
        Args:
            normalized_shape: 正規化する次元の形状
            eps: 数値安定性のためのイプシロン
            elementwise_affine: アフィン変換を使用するか
        """
        super().__init__()
        
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        # Requirement 1.12: アフィン変換パラメータ（実数）
        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            self.beta = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
    
    def forward(
        self,
        z: Union[ComplexTensor, torch.Tensor]
    ) -> Union[ComplexTensor, torch.Tensor]:
        """
        順伝播（Forward Pass）
        
        Formula:
            1. 複素平均: μ = E[z]
            2. 複素分散: σ² = E[|z - μ|²]
            3. 正規化: z' = (z - μ) / √(σ² + ε)
            4. アフィン変換: z'' = γ · z' + β
        
        Args:
            z: 複素入力（ComplexTensor or complex64）
        
        Returns:
            複素出力（入力と同じ型）
        
        Requirements:
            - Requirement 1.11: 複素平均と複素分散の計算
        """
        # 入力の型を判定
        if isinstance(z, ComplexTensor):
            # ComplexTensor入力
            # 正規化する次元を計算
            # normalized_shape = (D,) の場合、最後の次元で正規化
            dims = tuple(range(-len(self.normalized_shape), 0))
            
            # Requirement 1.11: 複素平均
            mean = z.mean(dim=dims, keepdim=True)
            
            # Requirement 1.11: 複素分散
            # σ² = E[|z - μ|²]
            centered = z - mean
            var = centered.abs_squared().mean(dim=dims, keepdim=True)
            
            # 正規化
            # z' = (z - μ) / √(σ² + ε)
            std = torch.sqrt(var + self.eps)
            # ComplexTensorの除算: 実部と虚部を個別に割る
            # stdはスカラーまたは(B, N, 1)の形状なので、broadcastingで対応
            z_norm = ComplexTensor(
                centered.real / std,
                centered.imag / std
            )
            
            # アフィン変換（実数パラメータ）
            if self.elementwise_affine:
                # γとβは実数なので、実部と虚部に同じ変換を適用
                # 入力の次元数に応じてunsqueezeする
                ndim = len(z.shape)
                gamma = self.gamma
                beta = self.beta
                for _ in range(ndim - len(self.normalized_shape)):
                    gamma = gamma.unsqueeze(0)
                    beta = beta.unsqueeze(0)
                
                # dtypeを合わせる
                if z_norm.dtype == torch.float16:
                    gamma = gamma.half()
                    beta = beta.half()
                z_norm = ComplexTensor(
                    z_norm.real * gamma + beta,
                    z_norm.imag * gamma
                )
            
            return z_norm
        
        elif z.is_complex():
            # PyTorch complex64入力
            dims = tuple(range(-len(self.normalized_shape), 0))
            
            # 複素平均
            mean = z.mean(dim=dims, keepdim=True)
            
            # 複素分散
            centered = z - mean
            var = (centered.real ** 2 + centered.imag ** 2).mean(dim=dims, keepdim=True)
            
            # 正規化
            std = torch.sqrt(var + self.eps)
            z_norm = centered / std
            
            # アフィン変換
            if self.elementwise_affine:
                z_norm = z_norm * self.gamma
                z_norm = z_norm + self.beta
            
            return z_norm
        
        else:
            raise TypeError(
                f"Input must be ComplexTensor or complex type, got {type(z)}"
            )
    
    def extra_repr(self) -> str:
        """モジュールの文字列表現"""
        return (
            f'{self.normalized_shape}, '
            f'eps={self.eps}, '
            f'elementwise_affine={self.elementwise_affine}'
        )
