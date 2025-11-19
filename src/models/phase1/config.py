"""
Phase 1 Efficiency Engine - Configuration System

このモジュールは、Phase 1の3つのコアコンポーネント（AR-SSM, HTT, LNS）の
設定を管理するデータクラスを提供します。

Requirements: 4.1, 4.2, 4.3, 12.3
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import warnings


@dataclass
class Phase1Config:
    """
    Phase 1 Efficiency Engineの統合設定クラス。
    
    このクラスは、AR-SSM、HTT、LNS、安定性監視、メモリ最適化の
    すべての設定パラメータを一元管理します。
    
    Attributes:
        # AR-SSM Configuration
        ar_ssm_enabled: AR-SSMレイヤーを有効化するか
        ar_ssm_max_rank: 最大ランク（複雑な入力用）
        ar_ssm_min_rank: 最小ランク（単純な入力用）
        ar_ssm_gate_hidden_dim: 複雑度ゲートの隠れ層次元（Noneの場合はd_model//4）
        ar_ssm_l1_regularization: ゲート活性化のL1正則化係数
        ar_ssm_use_fused_scan: Tritonカーネルによる高速化を使用するか
        
        # HTT Embedding Configuration
        htt_enabled: HTT Embeddingを有効化するか
        htt_rank: Tensor Trainのランク（圧縮率に影響）
        htt_num_cores: Tensor Trainのコア数（通常は2）
        htt_phase_encoding: 位相回転エンコーディングを使用するか
        htt_compression_target: 目標圧縮率（0.1 = 90%圧縮）
        
        # LNS Kernel Configuration
        lns_enabled: LNSカーネルを有効化するか（実験的、推論専用）
        lns_block_size_m: Tritonカーネルのブロックサイズ（M次元）
        lns_block_size_n: Tritonカーネルのブロックサイズ（N次元）
        lns_block_size_k: Tritonカーネルのブロックサイズ（K次元）
        lns_use_max_log: Max-log近似を使用するか
        
        # Stability Monitoring Configuration
        stability_monitoring_enabled: 安定性監視を有効化するか
        stability_threshold: Birman-Schwinger演算子の安定性閾値
        schatten_s1_bound: Schatten S1ノルムの上限
        schatten_s2_bound: Schatten S2ノルムの上限
        gradient_norm_threshold: 勾配ノルムの警告閾値
        
        # Memory Optimization Configuration
        use_gradient_checkpointing: 勾配チェックポイントを使用するか
        checkpoint_ar_ssm: AR-SSMレイヤーでチェックポイントを使用するか
        checkpoint_htt: HTTでチェックポイントを使用するか
        
        # Performance Targets
        target_vram_gb: 目標VRAM使用量（GB）
        target_ppl_degradation: 許容PPL劣化率（0.05 = 5%）
        target_speedup: 目標高速化率（fused scanで3x）
    
    Requirements:
        - 4.1: 既存インフラとの統合
        - 4.2: テストファイルの作成
        - 4.3: AGENTS.md準拠
        - 12.3: ハイパーパラメータドキュメント
    """
    
    # AR-SSM Configuration
    ar_ssm_enabled: bool = True
    ar_ssm_max_rank: int = 32
    ar_ssm_min_rank: int = 4
    ar_ssm_gate_hidden_dim: Optional[int] = None  # Default: d_model // 4
    ar_ssm_l1_regularization: float = 0.001
    ar_ssm_use_fused_scan: bool = True
    
    # HTT Embedding Configuration
    htt_enabled: bool = True
    htt_rank: int = 16
    htt_num_cores: int = 2
    htt_phase_encoding: bool = True
    htt_compression_target: float = 0.1  # 90% compression
    
    # LNS Kernel Configuration (Experimental)
    lns_enabled: bool = False  # Inference-only, experimental
    lns_block_size_m: int = 128
    lns_block_size_n: int = 128
    lns_block_size_k: int = 32
    lns_use_max_log: bool = True
    
    # Stability Monitoring Configuration
    stability_monitoring_enabled: bool = True
    stability_threshold: float = 1e-6
    schatten_s1_bound: float = 100.0
    schatten_s2_bound: float = 50.0
    gradient_norm_threshold: float = 10.0
    
    # Memory Optimization Configuration
    use_gradient_checkpointing: bool = True
    checkpoint_ar_ssm: bool = True
    checkpoint_htt: bool = False  # HTT is already memory-efficient
    
    # Performance Targets (for validation)
    target_vram_gb: float = 8.0
    target_ppl_degradation: float = 0.05  # 5% max
    target_speedup: float = 3.0  # 3x for fused scan
    
    def validate(self) -> None:
        """
        設定の整合性を検証します。
        
        Raises:
            ValueError: 設定が無効な場合
            
        物理的直観:
        - ランクは min <= max でなければならない（数学的制約）
        - 圧縮率は 0 < ratio < 1 でなければならない（定義域）
        - PPL劣化率は正でなければならない（品質保証）
        """
        errors = []
        
        # AR-SSM validation
        if self.ar_ssm_max_rank < self.ar_ssm_min_rank:
            errors.append(
                f"ar_ssm_max_rank ({self.ar_ssm_max_rank}) must be >= "
                f"ar_ssm_min_rank ({self.ar_ssm_min_rank})"
            )
        
        if self.ar_ssm_min_rank < 1:
            errors.append(
                f"ar_ssm_min_rank ({self.ar_ssm_min_rank}) must be >= 1"
            )
        
        if self.ar_ssm_l1_regularization < 0:
            errors.append(
                f"ar_ssm_l1_regularization ({self.ar_ssm_l1_regularization}) "
                f"must be >= 0"
            )
        
        # HTT validation
        if not (0.0 < self.htt_compression_target < 1.0):
            errors.append(
                f"htt_compression_target ({self.htt_compression_target}) "
                f"must be in range (0, 1)"
            )
        
        if self.htt_rank < 1:
            errors.append(
                f"htt_rank ({self.htt_rank}) must be >= 1"
            )
        
        if self.htt_num_cores < 2:
            errors.append(
                f"htt_num_cores ({self.htt_num_cores}) must be >= 2"
            )
        
        # LNS validation
        if self.lns_block_size_m < 1 or self.lns_block_size_n < 1 or self.lns_block_size_k < 1:
            errors.append(
                f"LNS block sizes must be >= 1 (got M={self.lns_block_size_m}, "
                f"N={self.lns_block_size_n}, K={self.lns_block_size_k})"
            )
        
        # Stability validation
        if self.stability_threshold <= 0:
            errors.append(
                f"stability_threshold ({self.stability_threshold}) must be > 0"
            )
        
        if self.schatten_s1_bound <= 0 or self.schatten_s2_bound <= 0:
            errors.append(
                f"Schatten norm bounds must be > 0 (got S1={self.schatten_s1_bound}, "
                f"S2={self.schatten_s2_bound})"
            )
        
        if self.gradient_norm_threshold <= 0:
            errors.append(
                f"gradient_norm_threshold ({self.gradient_norm_threshold}) must be > 0"
            )
        
        # Performance targets validation
        if self.target_vram_gb <= 0:
            errors.append(
                f"target_vram_gb ({self.target_vram_gb}) must be > 0"
            )
        
        if self.target_ppl_degradation <= 0:
            errors.append(
                f"target_ppl_degradation ({self.target_ppl_degradation}) must be > 0"
            )
        
        if self.target_speedup <= 0:
            errors.append(
                f"target_speedup ({self.target_speedup}) must be > 0"
            )
        
        # Warnings for potentially problematic configurations
        if self.lns_enabled:
            warnings.warn(
                "LNS kernel is experimental and inference-only. "
                "It may cause numerical instability during training.",
                UserWarning
            )
        
        if self.ar_ssm_use_fused_scan:
            try:
                import triton
            except ImportError:
                warnings.warn(
                    "ar_ssm_use_fused_scan=True but Triton is not installed. "
                    "Will fall back to torch.cumsum.",
                    UserWarning
                )
        
        if errors:
            raise ValueError(
                "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式に変換します（ログ記録用）。"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Phase1Config":
        """辞書から設定を復元します。"""
        return cls(**config_dict)
    
    @classmethod
    def for_hardware(cls, vram_gb: float) -> "Phase1Config":
        """
        ハードウェア制約に基づいてプリセット設定を作成します。
        
        Args:
            vram_gb: 利用可能なVRAM容量（GB）
            
        Returns:
            最適化されたPhase1Config
            
        物理的直観:
        - 8GB: RTX 3080 Mobile相当、最大圧縮
        - 10GB: RTX 3080相当、バランス型
        - 24GB: RTX 4090相当、品質優先
        """
        if vram_gb <= 8:
            # Maximum compression for 8GB
            return cls(
                ar_ssm_max_rank=16,
                ar_ssm_min_rank=4,
                htt_rank=8,
                htt_compression_target=0.05,  # 95% compression
                use_gradient_checkpointing=True,
                checkpoint_ar_ssm=True,
                target_vram_gb=8.0,
            )
        elif vram_gb <= 10:
            # Balanced for 10GB (default)
            return cls(
                ar_ssm_max_rank=32,
                ar_ssm_min_rank=4,
                htt_rank=16,
                htt_compression_target=0.1,  # 90% compression
                use_gradient_checkpointing=True,
                checkpoint_ar_ssm=True,
                target_vram_gb=10.0,
            )
        else:
            # Quality-focused for 24GB+
            return cls(
                ar_ssm_max_rank=64,
                ar_ssm_min_rank=8,
                htt_rank=32,
                htt_compression_target=0.2,  # 80% compression
                use_gradient_checkpointing=False,
                checkpoint_ar_ssm=False,
                target_vram_gb=24.0,
            )
    
    @classmethod
    def for_inference(cls) -> "Phase1Config":
        """
        推論専用の設定を作成します（LNS有効化）。
        
        Returns:
            推論最適化されたPhase1Config
        """
        config = cls()
        config.lns_enabled = True
        config.use_gradient_checkpointing = False
        config.checkpoint_ar_ssm = False
        config.checkpoint_htt = False
        return config
    
    @classmethod
    def for_maximum_quality(cls) -> "Phase1Config":
        """
        品質最優先の設定を作成します（圧縮最小化）。
        
        Returns:
            品質優先のPhase1Config
        """
        return cls(
            ar_ssm_max_rank=128,
            ar_ssm_min_rank=16,
            htt_rank=64,
            htt_compression_target=0.3,  # 70% compression
            ar_ssm_l1_regularization=0.0001,  # Weaker sparsity
            target_ppl_degradation=0.01,  # 1% max degradation
        )
    
    @classmethod
    def for_maximum_efficiency(cls) -> "Phase1Config":
        """
        効率最優先の設定を作成します（圧縮最大化）。
        
        Returns:
            効率優先のPhase1Config
        """
        return cls(
            ar_ssm_max_rank=16,
            ar_ssm_min_rank=2,
            htt_rank=8,
            htt_compression_target=0.05,  # 95% compression
            ar_ssm_l1_regularization=0.01,  # Strong sparsity
            use_gradient_checkpointing=True,
            checkpoint_ar_ssm=True,
            target_vram_gb=6.0,  # Ultra-low VRAM target
        )


@dataclass
class Phase1Diagnostics:
    """
    Phase 1コンポーネントの診断情報を追跡するデータクラス。
    
    このクラスは、AR-SSM、HTT、LNS、安定性監視の各コンポーネントから
    収集されたメトリクスを保持し、ログ記録や分析に使用されます。
    
    Attributes:
        # AR-SSM Diagnostics
        ar_ssm_effective_rank: ゲーティング後の実効ランク（平均）
        ar_ssm_gate_sparsity: ゲートが0に近い割合
        ar_ssm_memory_saved_mb: フルランクと比較した節約メモリ量（MB）
        
        # HTT Diagnostics
        htt_compression_ratio: 実際に達成された圧縮率
        htt_reconstruction_error: 再構成誤差 ||E - E_reconstructed||_F
        
        # LNS Diagnostics (if enabled)
        lns_speedup: 標準matmulと比較した高速化率
        lns_accuracy_loss: Max-log近似による数値誤差
        
        # Stability Diagnostics
        bk_det_condition: |det(I - K_ε)| の値
        bk_schatten_s1: Schatten S1ノルム
        bk_schatten_s2: Schatten S2ノルム
        bk_min_eigenvalue: (I - K_ε)の最小固有値
        stability_warnings: 安定性警告のリスト
        
        # Performance Diagnostics
        forward_time_ms: 順伝播時間（ミリ秒）
        backward_time_ms: 逆伝播時間（ミリ秒）
        peak_vram_mb: ピークVRAM使用量（MB）
        throughput_tokens_per_sec: スループット（トークン/秒）
    
    Requirements:
        - 4.1: 既存インフラとの統合
        - 6.1-6.6: テストと検証
    """
    
    # AR-SSM Diagnostics
    ar_ssm_effective_rank: float = 0.0
    ar_ssm_gate_sparsity: float = 0.0
    ar_ssm_memory_saved_mb: float = 0.0
    
    # HTT Diagnostics
    htt_compression_ratio: float = 0.0
    htt_reconstruction_error: float = 0.0
    
    # LNS Diagnostics (optional)
    lns_speedup: Optional[float] = None
    lns_accuracy_loss: Optional[float] = None
    
    # Stability Diagnostics
    bk_det_condition: float = 1.0
    bk_schatten_s1: float = 0.0
    bk_schatten_s2: float = 0.0
    bk_min_eigenvalue: float = 1.0
    stability_warnings: List[str] = field(default_factory=list)
    
    # Performance Diagnostics
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    peak_vram_mb: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """診断情報を辞書形式に変換します（ログ記録用）。"""
        return asdict(self)
    
    def is_healthy(self, config: Phase1Config) -> bool:
        """
        すべてのメトリクスが許容範囲内かチェックします。
        
        Args:
            config: 検証に使用するPhase1Config
            
        Returns:
            すべてのメトリクスが健全な場合True
            
        物理的直観:
        - 安定性条件: det(I - K_ε) > δ （特異点回避）
        - Schattenノルム: 理論的上限以下（演算子有界性）
        - VRAM: 目標値以下（ハードウェア制約）
        - 警告なし: システム正常動作
        """
        checks = [
            self.bk_det_condition > config.stability_threshold,
            self.bk_schatten_s1 < config.schatten_s1_bound,
            self.bk_schatten_s2 < config.schatten_s2_bound,
            self.peak_vram_mb < config.target_vram_gb * 1024,
            len(self.stability_warnings) == 0,
        ]
        return all(checks)
    
    def get_summary(self) -> str:
        """
        診断情報の人間可読なサマリーを生成します。
        
        Returns:
            フォーマットされたサマリー文字列
        """
        lines = [
            "=== Phase 1 Diagnostics Summary ===",
            "",
            "AR-SSM:",
            f"  Effective Rank: {self.ar_ssm_effective_rank:.2f}",
            f"  Gate Sparsity: {self.ar_ssm_gate_sparsity:.2%}",
            f"  Memory Saved: {self.ar_ssm_memory_saved_mb:.1f} MB",
            "",
            "HTT:",
            f"  Compression Ratio: {self.htt_compression_ratio:.2%}",
            f"  Reconstruction Error: {self.htt_reconstruction_error:.6f}",
            "",
        ]
        
        if self.lns_speedup is not None:
            lines.extend([
                "LNS:",
                f"  Speedup: {self.lns_speedup:.2f}x",
                f"  Accuracy Loss: {self.lns_accuracy_loss:.6f}",
                "",
            ])
        
        lines.extend([
            "Stability:",
            f"  det(I - K_ε): {self.bk_det_condition:.2e}",
            f"  Schatten S1: {self.bk_schatten_s1:.2f}",
            f"  Schatten S2: {self.bk_schatten_s2:.2f}",
            f"  Min Eigenvalue: {self.bk_min_eigenvalue:.2e}",
            f"  Warnings: {len(self.stability_warnings)}",
            "",
            "Performance:",
            f"  Forward Time: {self.forward_time_ms:.2f} ms",
            f"  Backward Time: {self.backward_time_ms:.2f} ms",
            f"  Peak VRAM: {self.peak_vram_mb:.1f} MB",
            f"  Throughput: {self.throughput_tokens_per_sec:.1f} tokens/sec",
        ])
        
        if self.stability_warnings:
            lines.extend([
                "",
                "Stability Warnings:",
            ])
            for warning in self.stability_warnings:
                lines.append(f"  - {warning}")
        
        return "\n".join(lines)


@dataclass
class Phase1TrainingState:
    """
    Phase 1コンポーネントの訓練状態を管理するデータクラス。
    
    このクラスは、カリキュラム学習のためのランクスケジューリング、
    安定性追跡、パフォーマンス追跡を行います。
    
    Attributes:
        # Rank Scheduling (Curriculum Learning)
        current_max_rank: 現在の最大ランク
        rank_schedule_step: ランクスケジュールのステップ数
        rank_warmup_steps: ウォームアップステップ数
        
        # Stability Tracking
        stability_violations: 安定性違反の累積回数
        last_stable_checkpoint: 最後の安定チェックポイントパス
        
        # Performance Tracking
        best_ppl: 最良のPerplexity
        best_vram_mb: 最良のVRAM使用量
        
        # Component-specific State
        ar_ssm_gate_stats: AR-SSMゲート統計
        htt_phase_stats: HTT位相統計
    
    Requirements:
        - 4.1: 既存インフラとの統合
        - 12.3: ハイパーパラメータドキュメント
    """
    
    # Rank Scheduling (Curriculum Learning)
    current_max_rank: int = 4
    rank_schedule_step: int = 0
    rank_warmup_steps: int = 1000
    
    # Stability Tracking
    stability_violations: int = 0
    last_stable_checkpoint: Optional[str] = None
    
    # Performance Tracking
    best_ppl: float = float('inf')
    best_vram_mb: float = float('inf')
    
    # Component-specific State
    ar_ssm_gate_stats: Dict[str, float] = field(default_factory=dict)
    htt_phase_stats: Dict[str, float] = field(default_factory=dict)
    
    def update_rank_schedule(self, config: Phase1Config) -> None:
        """
        カリキュラムスケジュールに基づいて現在の最大ランクを更新します。
        
        Args:
            config: Phase1Config（min_rank, max_rankを含む）
            
        物理的直観:
        - 訓練初期: 低ランク（単純な構造を学習）
        - 訓練後期: 高ランク（複雑な構造を学習）
        - 線形スケジューリング: 安定した学習曲線
        """
        progress = min(1.0, self.rank_schedule_step / self.rank_warmup_steps)
        self.current_max_rank = int(
            config.ar_ssm_min_rank + 
            progress * (config.ar_ssm_max_rank - config.ar_ssm_min_rank)
        )
        self.rank_schedule_step += 1
    
    def record_stability_violation(self, checkpoint_path: Optional[str] = None) -> None:
        """
        安定性違反を記録します。
        
        Args:
            checkpoint_path: 現在のチェックポイントパス（あれば）
        """
        self.stability_violations += 1
        if checkpoint_path and self.last_stable_checkpoint is None:
            # 最初の違反前のチェックポイントを保存
            self.last_stable_checkpoint = checkpoint_path
    
    def update_best_metrics(
        self,
        ppl: Optional[float] = None,
        vram_mb: Optional[float] = None
    ) -> Dict[str, bool]:
        """
        最良メトリクスを更新します。
        
        Args:
            ppl: 現在のPerplexity
            vram_mb: 現在のVRAM使用量
            
        Returns:
            更新されたメトリクスの辞書 {'ppl': bool, 'vram': bool}
        """
        updated = {'ppl': False, 'vram': False}
        
        if ppl is not None and ppl < self.best_ppl:
            self.best_ppl = ppl
            updated['ppl'] = True
        
        if vram_mb is not None and vram_mb < self.best_vram_mb:
            self.best_vram_mb = vram_mb
            updated['vram'] = True
        
        return updated
    
    def to_dict(self) -> Dict[str, Any]:
        """訓練状態を辞書形式に変換します（チェックポイント保存用）。"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]) -> "Phase1TrainingState":
        """辞書から訓練状態を復元します。"""
        return cls(**state_dict)
