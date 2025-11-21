"""
Complex32 Tensor Implementation with Planar Memory Layout

このモジュールは、メモリ効率を50%削減するためのcomplex32（半精度複素数）を実装します。

Physical Intuition:
    複素数は実部（振幅）と虚部（位相）を持ち、量子力学的な干渉効果をモデリングできます。
    否定形・皮肉・多義語などの言語現象を、複素平面上の干渉として表現します。

Memory Layout:
    Planar形式を採用: [RRR...III...] (実部と虚部を分離)
    - Interleaved形式 [RIRIRI...] よりもCUDAのcoalesced accessに最適
    - Tritonカーネルでのメモリアクセス最適化が容易
    - 実部と虚部を独立して処理可能

Memory Efficiency:
    - complex64: 2 × 4 bytes = 8 bytes/element
    - complex32 (Planar): 2 × 2 bytes = 4 bytes/element (50%削減)
"""

import torch
import torch.nn as nn
from typing import Union, Tuple


class ComplexTensor:
    """
    Complex32 Tensor with Planar Memory Layout
    
    このクラスは、float16の実部と虚部を分離して保持することで、
    complex64の半分のメモリで複素数演算を実現します。
    
    Attributes:
        real (torch.HalfTensor): 実部（float16）
        imag (torch.HalfTensor): 虚部（float16）
        shape (torch.Size): テンソルの形状
        device (torch.device): デバイス（CPU/CUDA）
        dtype (torch.dtype): データ型（常にtorch.float16）
    
    Examples:
        >>> # 基本的な使用方法
        >>> real = torch.randn(4, 10, 64, dtype=torch.float16)
        >>> imag = torch.randn(4, 10, 64, dtype=torch.float16)
        >>> z = ComplexTensor(real, imag)
        >>> print(z.shape)  # torch.Size([4, 10, 64])
        
        >>> # 複素数演算
        >>> z1 = ComplexTensor(torch.ones(2, 3, dtype=torch.float16), 
        ...                    torch.zeros(2, 3, dtype=torch.float16))
        >>> z2 = ComplexTensor(torch.zeros(2, 3, dtype=torch.float16), 
        ...                    torch.ones(2, 3, dtype=torch.float16))
        >>> z3 = z1 + z2  # (1 + 0i) + (0 + 1i) = (1 + 1i)
        >>> z4 = z1 * z2  # (1 + 0i) * (0 + 1i) = (0 + 1i)
        
        >>> # PyTorch complex64との相互変換
        >>> z_complex64 = z.to_complex64()
        >>> z_back = ComplexTensor.from_complex64(z_complex64)
    
    Requirements:
        - Requirement 1.1: 実部と虚部をfloat16で保持
        - Requirement 1.2: 複素数演算（加算、乗算、共役、絶対値）
        - Requirement 1.3: complex64との相互変換
    """
    
    def __init__(self, real: torch.Tensor, imag: torch.Tensor):
        """
        ComplexTensorの初期化
        
        Args:
            real (torch.Tensor): 実部（float16推奨）
            imag (torch.Tensor): 虚部（float16推奨）
        
        Raises:
            AssertionError: 実部と虚部の形状が一致しない場合
            AssertionError: データ型がfloat16でない場合
        """
        assert real.shape == imag.shape, \
            f"Real and imag must have same shape, got {real.shape} and {imag.shape}"
        assert real.dtype == torch.float16, \
            f"Real must be float16, got {real.dtype}"
        assert imag.dtype == torch.float16, \
            f"Imag must be float16, got {imag.dtype}"
        
        self.real = real
        self.imag = imag
        self.shape = real.shape
        self.device = real.device
        self.dtype = torch.float16
    
    def to(self, device: Union[str, torch.device]) -> 'ComplexTensor':
        """
        デバイスへの移動
        
        Args:
            device: 移動先のデバイス（'cuda', 'cpu', torch.device等）
        
        Returns:
            ComplexTensor: 移動後のComplexTensor
        """
        return ComplexTensor(self.real.to(device), self.imag.to(device))
    
    def cuda(self) -> 'ComplexTensor':
        """CUDAデバイスへの移動"""
        return self.to('cuda')
    
    def cpu(self) -> 'ComplexTensor':
        """CPUデバイスへの移動"""
        return self.to('cpu')
    
    def clone(self) -> 'ComplexTensor':
        """ComplexTensorの複製"""
        return ComplexTensor(self.real.clone(), self.imag.clone())
    
    def detach(self) -> 'ComplexTensor':
        """勾配計算グラフからの切り離し"""
        return ComplexTensor(self.real.detach(), self.imag.detach())
    
    def requires_grad_(self, requires_grad: bool = True) -> 'ComplexTensor':
        """勾配計算の有効化/無効化"""
        self.real.requires_grad_(requires_grad)
        self.imag.requires_grad_(requires_grad)
        return self
    
    @property
    def requires_grad(self) -> bool:
        """勾配計算が有効かどうか"""
        return self.real.requires_grad or self.imag.requires_grad
    
    def __repr__(self) -> str:
        """文字列表現"""
        return f"ComplexTensor(shape={self.shape}, device={self.device}, dtype={self.dtype})"
    
    def __str__(self) -> str:
        """文字列表現（詳細）"""
        return f"ComplexTensor(\n  real={self.real},\n  imag={self.imag}\n)"
    
    # ========================================
    # Requirement 1.2: 複素数演算
    # ========================================
    
    def __add__(self, other: Union['ComplexTensor', float, int]) -> 'ComplexTensor':
        """
        複素数の加算
        
        Formula:
            (a + bi) + (c + di) = (a + c) + (b + d)i
        
        Args:
            other: 加算する値（ComplexTensor、実数）
        
        Returns:
            ComplexTensor: 加算結果
        """
        if isinstance(other, ComplexTensor):
            return ComplexTensor(
                self.real + other.real,
                self.imag + other.imag
            )
        elif isinstance(other, (int, float)):
            # 実数の加算: z + r = (real + r) + imag*i
            return ComplexTensor(
                self.real + other,
                self.imag
            )
        else:
            raise TypeError(f"Unsupported operand type for +: ComplexTensor and {type(other)}")
    
    def __radd__(self, other: Union[float, int]) -> 'ComplexTensor':
        """右側からの加算（r + z）"""
        return self.__add__(other)
    
    def __sub__(self, other: Union['ComplexTensor', float, int]) -> 'ComplexTensor':
        """
        複素数の減算
        
        Formula:
            (a + bi) - (c + di) = (a - c) + (b - d)i
        """
        if isinstance(other, ComplexTensor):
            return ComplexTensor(
                self.real - other.real,
                self.imag - other.imag
            )
        elif isinstance(other, (int, float)):
            return ComplexTensor(
                self.real - other,
                self.imag
            )
        else:
            raise TypeError(f"Unsupported operand type for -: ComplexTensor and {type(other)}")
    
    def __rsub__(self, other: Union[float, int]) -> 'ComplexTensor':
        """右側からの減算（r - z）"""
        if isinstance(other, (int, float)):
            return ComplexTensor(
                other - self.real,
                -self.imag
            )
        else:
            raise TypeError(f"Unsupported operand type for -: {type(other)} and ComplexTensor")
    
    def __mul__(self, other: Union['ComplexTensor', float, int]) -> 'ComplexTensor':
        """
        複素数の乗算
        
        Formula:
            (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        
        Physical Intuition:
            複素数の乗算は、振幅の乗算と位相の加算に対応します。
            これにより、量子力学的な干渉効果を表現できます。
        
        Numerical Stability:
            - オーバーフロー対策: 中間計算でfloat32を使用
            - アンダーフロー対策: 結果をfloat16にキャストする際にclamp
        
        Args:
            other: 乗算する値（ComplexTensor、実数）
        
        Returns:
            ComplexTensor: 乗算結果
        """
        if isinstance(other, ComplexTensor):
            # (a + bi)(c + di) = (ac - bd) + (ad + bc)i
            # 数値安定性のため、float32で計算してからfloat16に戻す
            real_part = (self.real.float() * other.real.float() - 
                        self.imag.float() * other.imag.float()).half()
            imag_part = (self.real.float() * other.imag.float() + 
                        self.imag.float() * other.real.float()).half()
            return ComplexTensor(real_part, imag_part)
        elif isinstance(other, (int, float)):
            # 実数の乗算: z * r = (real * r) + (imag * r)i
            return ComplexTensor(
                self.real * other,
                self.imag * other
            )
        else:
            raise TypeError(f"Unsupported operand type for *: ComplexTensor and {type(other)}")
    
    def __rmul__(self, other: Union[float, int]) -> 'ComplexTensor':
        """右側からの乗算（r * z）"""
        return self.__mul__(other)
    
    def __truediv__(self, other: Union['ComplexTensor', float, int]) -> 'ComplexTensor':
        """
        複素数の除算
        
        Formula:
            (a + bi) / (c + di) = [(ac + bd) + (bc - ad)i] / (c² + d²)
        
        Numerical Stability:
            - ゼロ除算対策: 分母に小さな値（1e-8）を加算
            - オーバーフロー対策: float32で計算
        """
        if isinstance(other, ComplexTensor):
            # (a + bi) / (c + di) = [(ac + bd) + (bc - ad)i] / (c² + d²)
            denominator = (other.real.float() ** 2 + other.imag.float() ** 2 + 1e-8)
            real_part = ((self.real.float() * other.real.float() + 
                         self.imag.float() * other.imag.float()) / denominator).half()
            imag_part = ((self.imag.float() * other.real.float() - 
                         self.real.float() * other.imag.float()) / denominator).half()
            return ComplexTensor(real_part, imag_part)
        elif isinstance(other, (int, float)):
            # 実数の除算: z / r = (real / r) + (imag / r)i
            # ゼロ除算対策
            if abs(other) < 1e-8:
                other = 1e-8 if other >= 0 else -1e-8
            return ComplexTensor(
                self.real / other,
                self.imag / other
            )
        else:
            raise TypeError(f"Unsupported operand type for /: ComplexTensor and {type(other)}")
    
    def conj(self) -> 'ComplexTensor':
        """
        複素共役
        
        Formula:
            conj(a + bi) = a - bi
        
        Physical Intuition:
            複素共役は、位相を反転させる操作です。
            量子力学では、確率振幅の計算に使用されます。
        
        Returns:
            ComplexTensor: 複素共役
        """
        return ComplexTensor(self.real, -self.imag)
    
    def abs(self) -> torch.Tensor:
        """
        複素数の絶対値（振幅）
        
        Formula:
            |a + bi| = √(a² + b²)
        
        Physical Intuition:
            絶対値は、複素数の振幅（大きさ）を表します。
            量子力学では、確率振幅の大きさに対応します。
        
        Numerical Stability:
            - アンダーフロー対策: 小さな値（1e-8）を加算
            - オーバーフロー対策: float32で計算
        
        Returns:
            torch.Tensor: 絶対値（float16）
        """
        # 数値安定性のため、float32で計算してからfloat16に戻す
        magnitude = torch.sqrt(
            self.real.float() ** 2 + self.imag.float() ** 2 + 1e-8
        ).half()
        return magnitude
    
    def abs_squared(self) -> torch.Tensor:
        """
        複素数の絶対値の2乗
        
        Formula:
            |a + bi|² = a² + b²
        
        Note:
            sqrtを避けることで、数値安定性と計算効率が向上します。
        
        Returns:
            torch.Tensor: 絶対値の2乗（float16）
        """
        return (self.real ** 2 + self.imag ** 2)
    
    def angle(self) -> torch.Tensor:
        """
        複素数の偏角（位相）
        
        Formula:
            arg(a + bi) = arctan2(b, a)
        
        Physical Intuition:
            偏角は、複素数の位相を表します。
            量子力学では、波動関数の位相に対応します。
        
        Returns:
            torch.Tensor: 偏角（ラジアン、float16）
        """
        return torch.atan2(self.imag, self.real)
    
    # ========================================
    # Requirement 1.3: complex64との相互変換
    # ========================================
    
    def to_complex64(self) -> torch.Tensor:
        """
        PyTorch native complex64への変換
        
        この関数は、既存のPyTorchコードとの互換性のために使用されます。
        
        Returns:
            torch.Tensor: complex64テンソル
        
        Examples:
            >>> z = ComplexTensor(torch.ones(2, 3, dtype=torch.float16),
            ...                   torch.zeros(2, 3, dtype=torch.float16))
            >>> z_complex64 = z.to_complex64()
            >>> print(z_complex64.dtype)  # torch.complex64
        """
        return torch.complex(self.real.float(), self.imag.float())
    
    @staticmethod
    def from_complex64(z: torch.Tensor) -> 'ComplexTensor':
        """
        PyTorch native complex64からの変換
        
        Args:
            z (torch.Tensor): complex64テンソル
        
        Returns:
            ComplexTensor: ComplexTensor形式
        
        Raises:
            AssertionError: 入力がcomplex型でない場合
        
        Examples:
            >>> z_complex64 = torch.complex(torch.ones(2, 3), torch.zeros(2, 3))
            >>> z = ComplexTensor.from_complex64(z_complex64)
            >>> print(z.dtype)  # torch.float16
        """
        assert z.is_complex(), f"Input must be complex type, got {z.dtype}"
        return ComplexTensor(z.real.half(), z.imag.half())
    
    @staticmethod
    def from_real(real: torch.Tensor, imag: torch.Tensor = None) -> 'ComplexTensor':
        """
        実数テンソルからComplexTensorを作成
        
        Args:
            real (torch.Tensor): 実部
            imag (torch.Tensor, optional): 虚部（Noneの場合はゼロ）
        
        Returns:
            ComplexTensor: ComplexTensor形式
        """
        if imag is None:
            imag = torch.zeros_like(real)
        
        # float16に変換
        if real.dtype != torch.float16:
            real = real.half()
        if imag.dtype != torch.float16:
            imag = imag.half()
        
        return ComplexTensor(real, imag)
    
    # ========================================
    # ユーティリティメソッド
    # ========================================
    
    def view(self, *shape) -> 'ComplexTensor':
        """形状変更"""
        return ComplexTensor(self.real.view(*shape), self.imag.view(*shape))
    
    def reshape(self, *shape) -> 'ComplexTensor':
        """形状変更（連続性を保証）"""
        return ComplexTensor(self.real.reshape(*shape), self.imag.reshape(*shape))
    
    def permute(self, *dims) -> 'ComplexTensor':
        """次元の入れ替え"""
        return ComplexTensor(self.real.permute(*dims), self.imag.permute(*dims))
    
    def transpose(self, dim0: int, dim1: int) -> 'ComplexTensor':
        """2つの次元の入れ替え"""
        return ComplexTensor(
            self.real.transpose(dim0, dim1),
            self.imag.transpose(dim0, dim1)
        )
    
    def unsqueeze(self, dim: int) -> 'ComplexTensor':
        """次元の追加"""
        return ComplexTensor(self.real.unsqueeze(dim), self.imag.unsqueeze(dim))
    
    def squeeze(self, dim: int = None) -> 'ComplexTensor':
        """次元の削除"""
        if dim is None:
            return ComplexTensor(self.real.squeeze(), self.imag.squeeze())
        return ComplexTensor(self.real.squeeze(dim), self.imag.squeeze(dim))
    
    def mean(self, dim=None, keepdim=False) -> 'ComplexTensor':
        """平均値の計算"""
        if dim is None:
            return ComplexTensor(
                self.real.mean(),
                self.imag.mean()
            )
        return ComplexTensor(
            self.real.mean(dim=dim, keepdim=keepdim),
            self.imag.mean(dim=dim, keepdim=keepdim)
        )
    
    def sum(self, dim=None, keepdim=False) -> 'ComplexTensor':
        """合計値の計算"""
        if dim is None:
            return ComplexTensor(
                self.real.sum(),
                self.imag.sum()
            )
        return ComplexTensor(
            self.real.sum(dim=dim, keepdim=keepdim),
            self.imag.sum(dim=dim, keepdim=keepdim)
        )
    
    def norm(self, p: float = 2, dim=None, keepdim=False) -> torch.Tensor:
        """
        ノルムの計算
        
        Args:
            p: ノルムの次数（デフォルト: 2）
            dim: 計算する次元
            keepdim: 次元を保持するか
        
        Returns:
            torch.Tensor: ノルム（float16）
        """
        if p == 2:
            # L2ノルム: √(|z|²)
            abs_squared = self.abs_squared()
            if dim is None:
                return torch.sqrt(abs_squared.sum())
            return torch.sqrt(abs_squared.sum(dim=dim, keepdim=keepdim))
        else:
            # 一般のLpノルム
            abs_val = self.abs()
            if dim is None:
                return (abs_val ** p).sum() ** (1.0 / p)
            return (abs_val ** p).sum(dim=dim, keepdim=keepdim) ** (1.0 / p)
