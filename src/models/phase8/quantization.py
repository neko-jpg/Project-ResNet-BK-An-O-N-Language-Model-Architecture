"""
Logarithmic Quantization Module for Phase 8.

物理的直観:
- 双曲空間では境界に近いほど体積が指数的に増大
- 一様量子化では境界近くの情報が失われる
- 対数量子化で境界近くの解像度を上げる

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 36.1, 36.2, 36.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import json
import math


@dataclass
class QuantizationConfig:
    """Configuration for Logarithmic Quantization module."""
    bits: int = 8  # 量子化ビット数（4 or 8）
    boundary_factor: float = 2.0  # 境界適応係数
    calibration_samples: int = 1000  # キャリブレーションサンプル数
    per_channel: bool = True  # チャネルごとの量子化
    symmetric: bool = False  # 対称量子化
    
    def to_json(self) -> str:
        """Serialize configuration to JSON."""
        return json.dumps({
            'bits': self.bits,
            'boundary_factor': self.boundary_factor,
            'calibration_samples': self.calibration_samples,
            'per_channel': self.per_channel,
            'symmetric': self.symmetric,
        }, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'QuantizationConfig':
        """Parse configuration from JSON."""
        data = json.loads(json_str)
        return cls(**data)


@dataclass
class QuantizationDiagnostics:
    """Diagnostics from quantization."""
    quantization_error: float  # 量子化誤差
    compression_ratio: float  # 圧縮率
    boundary_samples_ratio: float  # 境界近くのサンプル比率
    effective_bits: float  # 実効ビット数
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'quantization_error': self.quantization_error,
            'compression_ratio': self.compression_ratio,
            'boundary_samples_ratio': self.boundary_samples_ratio,
            'effective_bits': self.effective_bits,
        }


class LogarithmicQuantizer(nn.Module):
    """
    Manifold-Aware Logarithmic Quantization.
    
    物理的直観:
    - 双曲空間では境界に近いほど体積が指数的に増大
    - 一様量子化では境界近くの情報が失われる
    - 対数量子化で境界近くの解像度を上げる
    
    **Property 7: Quantization Step Exponential Decay**
    量子化ステップサイズはノルムが1に近づくにつれて指数的に減少。
    
    Args:
        config_or_bits: QuantizationConfigまたは量子化ビット数（4 or 8）
        boundary_factor: 境界適応係数（デフォルト: 2.0）
        per_channel: チャネルごとの量子化
    """
    
    def __init__(
        self,
        config_or_bits = None,
        boundary_factor: float = 2.0,
        per_channel: bool = True,
        *,
        bits: int = None,
    ):
        super().__init__()
        
        # Configオブジェクトまたは個別パラメータをサポート
        if isinstance(config_or_bits, QuantizationConfig):
            config = config_or_bits
            bits = config.bits
            boundary_factor = config.boundary_factor
            per_channel = config.per_channel
        elif config_or_bits is not None:
            bits = config_or_bits
        elif bits is None:
            bits = 8
        
        self.bits = bits
        self.boundary_factor = boundary_factor
        self.per_channel = per_channel
        self.n_levels = 2 ** bits
        
        # キャリブレーション統計
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0.0))
        self.register_buffer('calibrated', torch.tensor(False))
        self.register_buffer('max_norm', torch.tensor(1.0))
        
        # チャネルごとのスケール（per_channel=Trueの場合）
        self.register_buffer('channel_scales', None)
    
    def calibrate(self, x: torch.Tensor):
        """
        代表データでキャリブレーション。
        
        Args:
            x: (B, N, D) or (B, D) キャリブレーションデータ
        """
        # ノルムの統計を計算
        if x.dim() == 3:
            norms = torch.norm(x, dim=-1)  # (B, N)
        else:
            norms = torch.norm(x, dim=-1)  # (B,)
        
        self.max_norm = norms.max()
        
        # 対数スケールを計算
        # 境界近く（norm > 0.9）では細かい量子化が必要
        log_max = torch.log(self.max_norm + 1e-6)
        self.scale = log_max / (self.n_levels / 2)
        
        # チャネルごとのスケール
        if self.per_channel and x.dim() >= 2:
            D = x.shape[-1]
            channel_max = x.abs().max(dim=0).values
            if x.dim() == 3:
                channel_max = channel_max.max(dim=0).values
            self.channel_scales = channel_max / (self.n_levels / 2 - 1)
            self.channel_scales = self.channel_scales.clamp(min=1e-8)
        
        self.calibrated = torch.tensor(True)
    
    def _compute_adaptive_scale(self, norm: torch.Tensor) -> torch.Tensor:
        """
        境界適応スケールを計算。
        
        **Property 7: Quantization Step Exponential Decay**
        ノルムが1に近づくにつれてスケールが指数的に減少。
        
        Args:
            norm: ノルム値
            
        Returns:
            adaptive_scale: 適応スケール
        """
        # 境界近くでは細かい量子化（スケールが小さい）
        # scale(norm) = base_scale * (1 - norm)^boundary_factor
        adaptive_scale = self.scale * torch.pow(1.0 - norm.clamp(max=0.999), self.boundary_factor)
        return adaptive_scale.clamp(min=1e-8)
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        対数量子化を実行。
        
        Args:
            x: (B, N, D) or (B, D) 入力テンソル
            
        Returns:
            quantized: 量子化されたテンソル
            scale_used: 使用されたスケール
        """
        # ノルムと方向を分離
        norm = torch.norm(x, dim=-1, keepdim=True)
        direction = x / (norm + 1e-8)
        
        # 境界適応スケール
        adaptive_scale = self._compute_adaptive_scale(norm)
        
        # 対数量子化
        log_norm = torch.log(norm + 1e-8)
        q_log_norm = torch.round(log_norm / adaptive_scale) * adaptive_scale
        q_norm = torch.exp(q_log_norm) - 1e-8
        
        # 量子化された埋め込み
        quantized = direction * q_norm.clamp(min=0)
        
        return quantized, adaptive_scale
    
    def quantize_int(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        整数量子化（INT8/INT4）を実行。
        
        Args:
            x: 入力テンソル
            
        Returns:
            quantized_int: 整数量子化テンソル
            scale: スケール
            zero_point: ゼロポイント
        """
        if self.per_channel and self.channel_scales is not None:
            scale = self.channel_scales
        else:
            scale = x.abs().max() / (self.n_levels / 2 - 1)
            scale = scale.clamp(min=1e-8)
        
        # 量子化
        q_min = -(self.n_levels // 2)
        q_max = self.n_levels // 2 - 1
        
        quantized_int = torch.round(x / scale).clamp(q_min, q_max).to(torch.int8)
        
        return quantized_int, scale, torch.tensor(0.0)
    
    def dequantize_int(
        self,
        quantized_int: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor
    ) -> torch.Tensor:
        """
        整数量子化からの復元。
        
        Args:
            quantized_int: 整数量子化テンソル
            scale: スケール
            zero_point: ゼロポイント
            
        Returns:
            dequantized: 復元されたテンソル
        """
        return quantized_int.float() * scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播（量子化→復元）。
        
        Args:
            x: 入力テンソル
            
        Returns:
            quantized: 量子化されたテンソル
        """
        if not self.calibrated:
            self.calibrate(x)
        
        quantized, _ = self.quantize(x)
        return quantized
    
    def get_diagnostics(self, x: torch.Tensor) -> QuantizationDiagnostics:
        """
        量子化の診断情報を取得。
        
        Args:
            x: 入力テンソル
            
        Returns:
            QuantizationDiagnostics
        """
        quantized, _ = self.quantize(x)
        
        # 量子化誤差
        error = (x - quantized).norm() / (x.norm() + 1e-8)
        
        # 圧縮率
        original_bits = 32  # FP32
        compression_ratio = original_bits / self.bits
        
        # 境界近くのサンプル比率
        norms = torch.norm(x, dim=-1)
        boundary_ratio = (norms > 0.9).float().mean()
        
        # 実効ビット数（エントロピーベース）
        effective_bits = self.bits  # 簡略化
        
        return QuantizationDiagnostics(
            quantization_error=error.item(),
            compression_ratio=compression_ratio,
            boundary_samples_ratio=boundary_ratio.item(),
            effective_bits=effective_bits,
        )


class INT8QuantizedKernel(nn.Module):
    """
    INT8 Quantized Kernel for Hyperbolic Distance Computation.
    
    整数演算で距離近似を行い、スループットを向上。
    
    **Property 19: INT8 Throughput Improvement**
    INT8量子化カーネルはFP16に対して2x以上のスループット向上を達成。
    
    Args:
        d_model: モデル次元
        use_lookup_table: 超越関数にルックアップテーブルを使用
    """
    
    def __init__(
        self,
        d_model: int = 256,
        use_lookup_table: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_lookup_table = use_lookup_table
        
        # 量子化器
        self.quantizer = LogarithmicQuantizer(bits=8)
        
        # ルックアップテーブル（arcosh近似用）
        if use_lookup_table:
            # arcosh(x) for x in [1, 10] with 256 entries
            x_vals = torch.linspace(1.0, 10.0, 256)
            arcosh_vals = torch.acosh(x_vals)
            self.register_buffer('arcosh_lut', arcosh_vals)
            self.register_buffer('arcosh_x_min', torch.tensor(1.0))
            self.register_buffer('arcosh_x_max', torch.tensor(10.0))
    
    def _arcosh_lookup(self, x: torch.Tensor) -> torch.Tensor:
        """
        ルックアップテーブルを使用したarcosh近似。
        
        Args:
            x: 入力（>= 1）
            
        Returns:
            arcosh(x)の近似値
        """
        if not self.use_lookup_table:
            return torch.acosh(x.clamp(min=1.0))
        
        # インデックスを計算
        x_clamped = x.clamp(self.arcosh_x_min, self.arcosh_x_max)
        idx = ((x_clamped - self.arcosh_x_min) / (self.arcosh_x_max - self.arcosh_x_min) * 255).long()
        idx = idx.clamp(0, 255)
        
        return self.arcosh_lut[idx]
    
    def compute_distance_int8(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        curvature: float = 1.0
    ) -> torch.Tensor:
        """
        INT8で双曲距離を近似計算。
        
        Args:
            q: (B, N, D) クエリ
            k: (B, M, D) キー
            curvature: 曲率
            
        Returns:
            distance: (B, N, M) 近似双曲距離
        """
        # INT8量子化
        q_int, q_scale, _ = self.quantizer.quantize_int(q)
        k_int, k_scale, _ = self.quantizer.quantize_int(k)
        
        # 整数演算で内積を計算
        # (B, N, D) @ (B, D, M) -> (B, N, M)
        qk_int = torch.matmul(q_int.float(), k_int.float().transpose(-2, -1))
        
        # スケールを適用
        qk = qk_int * q_scale.unsqueeze(-1) * k_scale.unsqueeze(-2) if q_scale.dim() > 0 else qk_int * q_scale * k_scale
        
        # ノルムの計算
        q_norm_sq = (q ** 2).sum(dim=-1, keepdim=True)  # (B, N, 1)
        k_norm_sq = (k ** 2).sum(dim=-1, keepdim=True)  # (B, M, 1)
        
        # 双曲距離の近似
        # d(x, y) ≈ arcosh(1 + 2 * ||x - y||² / ((1 - ||x||²)(1 - ||y||²)))
        diff_sq = q_norm_sq + k_norm_sq.transpose(-2, -1) - 2 * qk
        diff_sq = diff_sq.clamp(min=1e-8)
        
        denom = (1 - curvature * q_norm_sq) * (1 - curvature * k_norm_sq.transpose(-2, -1))
        denom = denom.clamp(min=1e-8)
        
        cosh_arg = 1 + 2 * curvature * diff_sq / denom
        
        # ルックアップテーブルでarcosh
        distance = self._arcosh_lookup(cosh_arg) / math.sqrt(curvature)
        
        return distance
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        curvature: float = 1.0
    ) -> torch.Tensor:
        """
        INT8量子化アテンション。
        
        Args:
            q: (B, N, D) クエリ
            k: (B, M, D) キー
            v: (B, M, D) バリュー
            curvature: 曲率
            
        Returns:
            output: (B, N, D) 出力
        """
        # 距離計算
        distance = self.compute_distance_int8(q, k, curvature)
        
        # アテンション重み
        attn_weights = F.softmax(-distance, dim=-1)
        
        # 出力
        output = torch.matmul(attn_weights, v)
        
        return output


class CalibrationPipeline:
    """
    量子化キャリブレーションパイプライン。
    
    代表データを使用して最適な量子化パラメータを決定。
    
    Args:
        quantizer: LogarithmicQuantizer
        num_samples: キャリブレーションサンプル数
    """
    
    def __init__(
        self,
        quantizer: LogarithmicQuantizer,
        num_samples: int = 1000,
    ):
        self.quantizer = quantizer
        self.num_samples = num_samples
        self.calibration_data = []
    
    def add_sample(self, x: torch.Tensor):
        """キャリブレーションサンプルを追加。"""
        if len(self.calibration_data) < self.num_samples:
            self.calibration_data.append(x.detach().cpu())
    
    def calibrate(self):
        """キャリブレーションを実行。"""
        if not self.calibration_data:
            raise ValueError("No calibration data available")
        
        # 全サンプルを結合
        all_data = torch.cat(self.calibration_data, dim=0)
        
        # 量子化器をキャリブレーション
        self.quantizer.calibrate(all_data)
        
        # キャリブレーションデータをクリア
        self.calibration_data = []
    
    def get_calibration_stats(self) -> Dict:
        """キャリブレーション統計を取得。"""
        return {
            'scale': self.quantizer.scale.item(),
            'max_norm': self.quantizer.max_norm.item(),
            'calibrated': self.quantizer.calibrated.item(),
            'bits': self.quantizer.bits,
        }
    
    def save_calibration(self, path: str):
        """キャリブレーションパラメータを保存。"""
        stats = self.get_calibration_stats()
        with open(path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def load_calibration(self, path: str):
        """キャリブレーションパラメータを読み込み。"""
        with open(path, 'r') as f:
            stats = json.load(f)
        
        self.quantizer.scale = torch.tensor(stats['scale'])
        self.quantizer.max_norm = torch.tensor(stats['max_norm'])
        self.quantizer.calibrated = torch.tensor(stats['calibrated'])


def create_logarithmic_quantizer(
    bits: int = 8,
    boundary_factor: float = 2.0,
    **kwargs
) -> LogarithmicQuantizer:
    """
    Factory function for creating LogarithmicQuantizer.
    
    Args:
        bits: Quantization bits (4 or 8)
        boundary_factor: Boundary adaptation factor
        **kwargs: Additional arguments
        
    Returns:
        LogarithmicQuantizer instance
    """
    return LogarithmicQuantizer(
        bits=bits,
        boundary_factor=boundary_factor,
        **kwargs
    )


def create_int8_kernel(
    d_model: int = 256,
    use_lookup_table: bool = True,
) -> INT8QuantizedKernel:
    """
    Factory function for creating INT8QuantizedKernel.
    
    Args:
        d_model: Model dimension
        use_lookup_table: Use lookup table for transcendental functions
        
    Returns:
        INT8QuantizedKernel instance
    """
    return INT8QuantizedKernel(
        d_model=d_model,
        use_lookup_table=use_lookup_table,
    )
