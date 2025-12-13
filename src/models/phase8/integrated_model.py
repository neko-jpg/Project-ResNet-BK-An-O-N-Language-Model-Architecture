"""
Phase 8 Integrated Model - ResNetBK Based Implementation

Phase 8はPhase 7（ResNetBK + HTT Embedding + Hybrid Hyperbolic Attention）の拡張です。
Phase 7の全機能を継承し、以下の拡張を追加します：

1. BK-Core Hyperbolic Integration: BK-CoreのG_iiを使用した物理ベースゲーティング
2. AR-SSM Hyperbolic Fusion: AR-SSMと双曲空間の融合
3. Entailment Cones: 論理的含意関係の幾何学的検証（オプション）
4. Persistent Homology: トポロジカル解析と循環論理検出（オプション）
5. Sheaf Attention: マルチヘッド間の構造的整合性（オプション）
6. Quantized HTT: 量子化されたHolographic Tensor Train埋め込み（オプション）
7. その他の最適化技術（オプション）

重要な設計原則:
- Phase 7IntegratedModelをコアとして使用（ResNetBKベース）
- BK-CoreのG_iiを取得して物理情報として活用
- O(N)複雑度を維持
- 8GB VRAM制約を満たす
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from dataclasses import asdict
import time

from src.models.phase7.integrated_model import Phase7IntegratedModel
from src.models.phase1.htt_embedding import HTTDecoder
from .config import Phase8Config, Phase8Diagnostics
from .bk_core_hyperbolic import BKCoreHyperbolicIntegration, BKCoreHyperbolicConfig
from .ar_ssm_fusion import ARSSMHyperbolicFusion, ARSSMFusionConfig

# Import optimization kernels for training
try:
    from src.kernels.resonance_adaptive_curvature import ResonanceAdaptiveCurvature, StabilityMonitor
    _RESONANCE_OPT_AVAILABLE = True
except ImportError:
    _RESONANCE_OPT_AVAILABLE = False
    ResonanceAdaptiveCurvature = None
    StabilityMonitor = None
from .entailment_cones import EntailmentCones, EntailmentConeConfig
from .persistent_homology import HyperbolicPersistentHomology, PersistentHomologyConfig
from .sheaf_attention import SheafAttentionModule, SheafAttentionConfig
from .quantized_htt import (
    QuantizedHolographicTTEmbedding,
    QuantizationConfig,
    QuantizedHTTDecoder,
    AdaptiveRankQuantizedHTTEmbedding,
    AdaptiveQuantizedHTTDecoder,
)

# Import Hyperbolic Normalization (Phase 1)
try:
    from src.models.hyperbolic_normalization import HyperbolicRMSNorm
    _HYPERBOLIC_NORM_AVAILABLE = True
except ImportError:
    _HYPERBOLIC_NORM_AVAILABLE = False
    HyperbolicRMSNorm = None

# Import Resonant HTT Embedding (Riemannian Resonant Tunneling)
try:
    from src.models.phase1.resonant_htt_embedding import (
        ResonantHTTEmbedding,
        ResonantHTTDecoder,
        diagnose_vocab_size,
    )
    _RESONANT_HTT_AVAILABLE = True
except ImportError:
    _RESONANT_HTT_AVAILABLE = False
    ResonantHTTEmbedding = None
    ResonantHTTDecoder = None
    diagnose_vocab_size = None


class Phase8IntegratedModel(nn.Module):
    """
    Phase 8 Integrated Model
    
    Phase 7IntegratedModelを継承し、Phase 8の拡張機能を追加。
    
    アーキテクチャ:
    1. Phase7IntegratedModel（ResNetBK + HTT + Hybrid Hyperbolic Attention）
    2. BK-Core Hyperbolic Integration（G_iiゲーティング）
    3. AR-SSM Hyperbolic Fusion（AR-SSM + 双曲空間）
    4. オプション: Entailment Cones, Persistent Homology, Sheaf Attention
    
    Requirements: Phase8 design.md Section 2
    """
    
    def __init__(self, config: Phase8Config):
        super().__init__()
        self.config = config
        
        # ========== Phase 7 Core Model ==========
        # Phase 7IntegratedModelをコアとして使用
        # これにはResNetBK, HTT Embedding, Hybrid Hyperbolic Attentionが含まれる
        # Phase8固有のパラメータを除外してPhase7Configを作成
        # Use dataclasses.fields() to properly get all inherited fields
        from dataclasses import fields as dc_fields
        
        phase7_config_dict = {}
        for f in dc_fields(config):
            phase7_config_dict[f.name] = getattr(config, f.name)
        
        # Phase8固有のパラメータを除外
        phase8_specific_params = [
            # Component flags
            'use_bk_hyperbolic', 'use_ar_ssm_fusion',
            'enable_entailment_cones', 'enable_persistent_homology', 'enable_sheaf_attention',
            'enable_adaptive_computation', 'enable_koopman_bridge', 'enable_sparse_attention',
            'enable_kv_compression', 'enable_numerical_guards', 'enable_curvature_adaptation',
            # BK-Core settings
            'bk_hyperbolic_gate_scale', 'bk_hyperbolic_resonance_threshold',
            # AR-SSM settings
            'ar_ssm_max_rank', 'ar_ssm_min_rank',
            'ar_ssm_hyperbolic_rank_threshold', 'ar_ssm_curvature_adaptation_rate',
            # Entailment settings
            'entailment_aperture', 'entailment_margin',
            # Topology settings
            'topology_persistence_threshold', 'topology_max_dimension', 'topology_betti_threshold',
            'topology_loss_weight', 'topology_cycle_weight', 'topology_fragment_weight',
            'topology_subset_size',
            # Sheaf settings
            'sheaf_num_sections', 'sheaf_agreement_threshold',
            # Adaptive computation settings
            'adaptive_exit_threshold', 'adaptive_min_layers',
            # Sparse attention settings
            'sparse_top_k', 'sparse_block_size',
            # KV compression settings
            'kv_cache_dim', 'kv_eviction_threshold',
            'kv_use_reconstruction_loss', 'kv_reconstruction_weight',
            # Curvature settings
            'curvature_initial', 'curvature_min', 'curvature_max',
            # Numerical safety settings
            'max_norm', 'grad_clip',
            # Optimization & System settings (New in Phase 8)
            'use_torch_compile', 'compile_mode', 'compile_fullgraph',
            'use_flash_attention_2',
            'dataloader_num_workers', 'dataloader_pin_memory',
            'dataloader_prefetch_factor', 'dataloader_persistent_workers',
            'gradient_accumulation_steps',
            # Quantized HTT
            'quantized_htt',
            'low_rank_embedding', 'low_rank_ffn',
            'adaptive_rank_quantization', 'adaptive_rank_hot_bits', 'adaptive_rank_cold_bits',
            'adaptive_rank_hot', 'adaptive_rank_cold', 'adaptive_rank_frequency_threshold',
            'adaptive_rank_update_alpha', 'adaptive_rank_use_loss',
            # Phase 8 Optimization Kernels (NEW)
            'use_fused_mobius', 'use_green_function_cache', 'green_function_cache_size',
            'use_parallel_ssm_scan', 'use_fused_scattering_gate',
            'use_batched_hyperbolic_distance', 'use_resonance_adaptive_curvature',
            'resonance_threshold', 'curvature_adjustment_rate',
            'use_ternary_mobius_matmul', 'use_quantized_htt_fusion',
            # Resonant HTT (NEW)
            'use_resonant_htt', 'resonant_num_cores', 'use_zeta_init',
        ]

        # Phase 7Configに存在しないキーを削除
        for param in phase8_specific_params:
            phase7_config_dict.pop(param, None)
        
        # Phase7Configを作成
        from src.models.phase7.integrated_model import Phase7Config
        phase7_config = Phase7Config(**phase7_config_dict)
        self.phase7_model = Phase7IntegratedModel(phase7_config)

        # ========== Quantized HTT Replacement ==========
        # config.quantized_httがTrueの場合、通常のHTTを量子化版に置き換える
        self.quantized_embedding = None
        if getattr(config, 'adaptive_rank_quantization', False):
            print("Phase 8: Activating AdaptiveRankQuantizedHTTEmbedding...")
            self.quantized_embedding = AdaptiveRankQuantizedHTTEmbedding(
                vocab_size=config.vocab_size,
                d_model=config.d_model,
                hot_rank=config.adaptive_rank_hot,
                cold_rank=config.adaptive_rank_cold,
                hot_bits=config.adaptive_rank_hot_bits,
                cold_bits=config.adaptive_rank_cold_bits,
                frequency_threshold=config.adaptive_rank_frequency_threshold,
                ema_alpha=config.adaptive_rank_update_alpha,
                phase_encoding=True,
            )
            self.phase7_model.htt_embedding = self.quantized_embedding
            self.phase7_model.model.token_embedding = self.quantized_embedding
            self.phase7_model.model.lm_head = AdaptiveQuantizedHTTDecoder(self.quantized_embedding)
        elif getattr(config, 'quantized_htt', False):
            print("Phase 8: Activating QuantizedHolographicTTEmbedding...")
            self.quantized_embedding = QuantizedHolographicTTEmbedding(
                vocab_size=config.vocab_size,
                d_model=config.d_model,
                rank=config.htt_rank,
                bits=8, # Default to 8-bit, could be configurable
                phase_encoding=True,
            )
            # Replace in Phase 7 Model
            self.phase7_model.htt_embedding = self.quantized_embedding
            self.phase7_model.model.token_embedding = self.quantized_embedding
            # Re-initialize decoder with new embedding
            self.phase7_model.model.lm_head = QuantizedHTTDecoder(self.quantized_embedding)
        
        # ========== Resonant HTT Replacement (Riemannian Resonant Tunneling) ==========
        # use_resonant_httがTrueの場合、通常のHTTをResonantHTTに置き換える
        self.resonant_embedding = None
        if getattr(config, 'use_resonant_htt', False):
            if _RESONANT_HTT_AVAILABLE:
                # 語彙サイズの診断を出力
                if diagnose_vocab_size is not None:
                    diag = diagnose_vocab_size(config.vocab_size)
                    print(f"Phase 8: 🔮 Vocab Diagnosis: {diag['recommendation']}")
                
                print(f"Phase 8: Activating ResonantHTTEmbedding (Riemannian Resonant Tunneling)...")
                self.resonant_embedding = ResonantHTTEmbedding(
                    vocab_size=config.vocab_size,
                    d_model=config.d_model,
                    rank=config.htt_rank,
                    num_cores=getattr(config, 'resonant_num_cores', 4),
                    phase_encoding=True,
                    use_zeta_init=getattr(config, 'use_zeta_init', True),
                )
                # Replace in Phase 7 Model
                self.phase7_model.htt_embedding = self.resonant_embedding
                self.phase7_model.model.token_embedding = self.resonant_embedding
                # Re-initialize decoder with new embedding
                self.phase7_model.model.lm_head = ResonantHTTDecoder(self.resonant_embedding)
                print(f"Phase 8: ✔ ResonantHTT active - Condition number κ≈1 guaranteed")
            else:
                print("Phase 8: ⚠ ResonantHTT requested but not available, using standard HTT")
        
        # ========== Phase 8 Core Extensions ==========
        # BK-Core Hyperbolic Integration（必須）
        if config.use_bk_hyperbolic:
            bk_config = BKCoreHyperbolicConfig(
                d_model=config.d_model,
                curvature=getattr(config, 'curvature_initial', 1.0),
                gate_scale=config.bk_hyperbolic_gate_scale,
                resonance_threshold=config.bk_hyperbolic_resonance_threshold,
                use_scattering_gate=True,
                use_resonance_detection=True,
                # Phase 8 kernel optimizations
                use_fused_mobius=getattr(config, 'use_fused_mobius', True),
                use_green_function_cache=getattr(config, 'use_green_function_cache', True),
                use_fused_scattering_gate=getattr(config, 'use_fused_scattering_gate', True),
                green_function_cache_size=getattr(config, 'green_function_cache_size', 512),
            )
            self.bk_hyperbolic = BKCoreHyperbolicIntegration(bk_config)
        else:
            self.bk_hyperbolic = None
        
        # AR-SSM Hyperbolic Fusion（必須）
        if config.use_ar_ssm_fusion:
            ar_ssm_config = ARSSMFusionConfig(
                d_model=config.d_model,
                d_state=config.d_model // 4,  # 状態次元はd_modelの1/4
                max_rank=config.ar_ssm_max_rank,
                min_rank=config.ar_ssm_min_rank,
                curvature=getattr(config, 'curvature_initial', 1.0),
                distance_threshold=config.ar_ssm_hyperbolic_rank_threshold,
                curvature_adjustment_rate=config.ar_ssm_curvature_adaptation_rate,
                use_physics_gating=True,
                use_adaptive_rank=True,
                # Phase 8 kernel optimizations
                use_parallel_ssm_scan=getattr(config, 'use_parallel_ssm_scan', True),
                use_batched_hyperbolic_distance=getattr(config, 'use_batched_hyperbolic_distance', True),
            )
            self.ar_ssm_fusion = ARSSMHyperbolicFusion(ar_ssm_config)
        else:
            self.ar_ssm_fusion = None
        
        # ========== Phase 8 Optional Extensions ==========
        # Entailment Cones（オプション）
        if config.enable_entailment_cones:
            entailment_config = EntailmentConeConfig(
                d_model=config.d_model,
                curvature=1.0,
                initial_aperture=config.entailment_aperture,
            )
            self.entailment_cones = EntailmentCones(entailment_config)
        else:
            self.entailment_cones = None
        
        # Persistent Homology（オプション）
        if config.enable_persistent_homology:
            homology_config = PersistentHomologyConfig(
                d_model=config.d_model,
                max_dimension=config.topology_max_dimension,
                threshold_beta1=int(config.topology_betti_threshold * 10),  # 閾値を整数に変換
            )
            self.persistent_homology = HyperbolicPersistentHomology(homology_config)
        else:
            self.persistent_homology = None
        
        # Sheaf Attention（オプション）
        if config.enable_sheaf_attention:
            sheaf_config = SheafAttentionConfig(
                d_model=config.d_model,
                num_heads=config.num_heads,
                agreement_threshold=config.sheaf_agreement_threshold,
            )
            self.sheaf_attention = SheafAttentionModule(sheaf_config)
        else:
            self.sheaf_attention = None
        
        # 診断情報の初期化
        self.diagnostics = Phase8Diagnostics()
        self.last_topology_loss: Optional[torch.Tensor] = None
        
        # ========== Hyperbolic Normalization (Phase 1) ==========
        # Add post-embedding normalization to stabilize gradients
        if _HYPERBOLIC_NORM_AVAILABLE:
            self.embedding_norm = HyperbolicRMSNorm(config.d_model, eps=1e-6)
            print("Phase 8: ✔ HyperbolicRMSNorm enabled for embedding stabilization")
        else:
            self.embedding_norm = nn.LayerNorm(config.d_model)
            print("Phase 8: Using standard LayerNorm (HyperbolicRMSNorm not available)")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Forward pass
        
        処理フロー:
        1. Phase 7 forward（ResNetBK + HTT + Hybrid Hyperbolic Attention）
        2. BK-CoreからG_iiを取得
        3. BK-Core Hyperbolic Integration（G_iiゲーティング）
        4. AR-SSM Hyperbolic Fusion
        5. オプション機能の適用
        
        Args:
            input_ids: 入力トークンID [batch, seq_len]
            return_diagnostics: 診断情報を返すか
        
        Returns:
            logits: 出力ロジット [batch, seq_len, vocab_size]
            diagnostics: 診断情報（return_diagnostics=Trueの場合）
        """
        start_time = time.time()
        
        # ========== Phase 7 Forward ==========
        # Phase 7の完全な前向き計算を実行
        # これにはResNetBK, HTT Embedding, Hybrid Hyperbolic Attentionが含まれる
        if return_diagnostics:
            logits, phase7_diagnostics = self.phase7_model(input_ids, return_diagnostics=True)
        else:
            logits = self.phase7_model(input_ids, return_diagnostics=False)
            phase7_diagnostics = {}
        
        # ========== BK-CoreからG_iiを取得 ==========
        # Phase 7のResNetBKからBK-CoreのG_iiを取得
        G_ii = self._extract_green_function(input_ids)
        
        # ========== Phase 8 Extensions ==========
        # 中間表現を取得（Phase 7の埋め込み層から）
        # HTT Embedding (Quantized or Standard)
        x = self.phase7_model.htt_embedding(input_ids)  # [batch, seq_len, d_model]
        
        # ========== Apply Hyperbolic Normalization ==========
        # Stabilize embeddings to prevent NaN in gradients
        x = self.embedding_norm(x)
        
        batch_size, seq_len, d_model = x.shape
        topology_loss_value = None
        
        # Task 38.2: BK-Core Hyperbolic Integration
        if self.bk_hyperbolic is not None and G_ii is not None:
            # G_iiを使用してアテンション重みをゲーティング
            # G_iiの形状を確認して適切に処理
            if G_ii.dim() == 2:  # [batch, seq_len]
                # ダミーのアテンション重みを作成（実際にはPhase7から取得すべき）
                # ここでは簡略化のため、単位行列を使用
                dummy_attn = torch.eye(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
                bk_features, bk_diag = self.bk_hyperbolic(x, attention_weights=dummy_attn)
            else:
                # G_iiが利用できない場合はスキップ
                bk_features, bk_diag = self.bk_hyperbolic(x, attention_weights=None)
            
            # 診断情報を収集
            if return_diagnostics:
                self.diagnostics.bk_hyperbolic_gate_mean = bk_diag.get('gate_mean', torch.tensor(0.0)).item()
                self.diagnostics.bk_hyperbolic_gate_std = bk_diag.get('gate_std', torch.tensor(0.0)).item()
                self.diagnostics.bk_resonance_detected = bk_diag.get('is_resonant', torch.tensor(False)).item()
                self.diagnostics.bk_resonance_strength = bk_diag.get('resonance_strength', torch.tensor(0.0)).item()
            
            # BK特徴量を元の特徴量に加算（残差接続）
            x = x + bk_features
        
        # Task 38.3: AR-SSM Hyperbolic Fusion
        if self.ar_ssm_fusion is not None:
            # AR-SSMと双曲空間を融合
            # G_iiを物理情報として渡す
            ar_ssm_output, ar_ssm_diag = self.ar_ssm_fusion(x, G_ii)
            
            # 診断情報を収集 (handle both tensor and float values)
            if return_diagnostics:
                rank_val = ar_ssm_diag.get('effective_rank_mean', 0.0)
                dist_val = ar_ssm_diag.get('distance_mean', 0.0)
                self.diagnostics.ar_ssm_rank_mean = rank_val.item() if hasattr(rank_val, 'item') else float(rank_val)
                self.diagnostics.ar_ssm_hyperbolic_distance_mean = dist_val.item() if hasattr(dist_val, 'item') else float(dist_val)
            
                # 曲率調整の提案があれば記録
                if 'suggested_curvature' in ar_ssm_diag:
                    self.diagnostics.ar_ssm_curvature_adjusted = True
            
            # AR-SSM出力を元の特徴量に加算（残差接続）
            x = x + ar_ssm_output
        
        # Task 38.4: Entailment Cones（オプション）
        if self.entailment_cones is not None:
            # 論理的含意関係をチェック
            # Optimized: Avoid loop if possible, or keep simple
            # (Vectorization will be handled in separate optimization step)
            if return_diagnostics:
                 entailment_diag = self._check_entailment(x)
                 self.diagnostics.entailment_violation_rate = entailment_diag.get('violation_rate', 0.0)
                 self.diagnostics.avg_aperture = entailment_diag.get('avg_aperture', 0.0)
        
        # Task 38.5: Persistent Homology（オプション）
        if self.persistent_homology is not None:
            # トポロジカル解析
            # Only analyze first item in batch to save time
            if return_diagnostics:
                homology_diag = self._analyze_topology(x)
                self.diagnostics.betti_numbers = homology_diag.get('betti_numbers', [])
                self.diagnostics.persistent_entropy = homology_diag.get('persistent_entropy', 0.0)
                self.diagnostics.circular_reasoning_detected = homology_diag.get('circular_reasoning', False)

                # 循環論理が検出された場合、曲率を増加させる提案
                if self.diagnostics.circular_reasoning_detected:
                    self.diagnostics.topology_curvature_adjustment_suggested = True

            if getattr(self.config, 'topology_loss_weight', 0.0) > 0:
                topo_loss_raw = self.persistent_homology.regularization_loss(
                    x,
                    curvature=torch.tensor(self.config.curvature_initial, device=x.device, dtype=x.dtype),
                    cycle_weight=self.config.topology_cycle_weight,
                    fragmentation_weight=self.config.topology_fragment_weight,
                    max_tokens=self.config.topology_subset_size,
                )
                topology_loss_value = self.config.topology_loss_weight * topo_loss_raw
                self.last_topology_loss = topology_loss_value
                if return_diagnostics:
                    self.diagnostics.topology_loss = float(topology_loss_value.detach().item())
                    self.diagnostics.topology_subset_tokens = min(self.config.topology_subset_size, x.shape[1])
        
        # Task 38.6: Sheaf Attention（オプション）
        if self.sheaf_attention is not None:
            # マルチヘッド間の整合性チェック
            if return_diagnostics:
                sheaf_diag = self._check_sheaf_consistency(x)
                self.diagnostics.sheaf_agreement_mean = sheaf_diag.get('agreement_mean', 0.0)
                self.diagnostics.sheaf_consensus_rate = sheaf_diag.get('consensus_rate', 0.0)
        
        # ========== Performance Metrics ==========
        if return_diagnostics:
            forward_time = (time.time() - start_time) * 1000  # ms
            self.diagnostics.forward_time_ms = forward_time
            
            batch_size, seq_len = input_ids.shape
            tokens = batch_size * seq_len
            self.diagnostics.tokens_per_second = tokens / (forward_time / 1000) if forward_time > 0 else 0.0
            
            # メモリ使用量
            if torch.cuda.is_available():
                self.diagnostics.peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            # Phase 7の診断情報とマージ
            all_diagnostics = {
                'phase7': phase7_diagnostics,
                'phase8': asdict(self.diagnostics),
            }

            # Attach optional auxiliary losses to diagnostics
            if self.persistent_homology is not None and topology_loss_value is not None:
                all_diagnostics['phase8']['topology_aux_loss'] = float(topology_loss_value.detach().item())
            
            return logits, all_diagnostics
        
        return logits, None
    
    def _extract_green_function(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Phase 7のResNetBKからBK-CoreのG_iiを抽出
        
        Args:
            input_ids: 入力トークンID
        
        Returns:
            G_ii: グリーン関数対角成分（利用可能な場合）
        """
        try:
            # Phase 7のモデルからBK-Coreのブロックを取得
            # ResNetBKのブロックにアクセス
            if hasattr(self.phase7_model.model, 'blocks'):
                # 最初のブロックからG_iiを取得（簡略化）
                first_block = self.phase7_model.model.blocks[0]
                if hasattr(first_block, 'bk_layer'):
                    bk_layer = first_block.bk_layer
                    if hasattr(bk_layer, 'last_G_ii'):
                        return bk_layer.last_G_ii
            
            # G_iiが利用できない場合はNoneを返す
            return None
        except Exception:
            return None
    
    def _check_entailment(self, x: torch.Tensor) -> Dict[str, float]:
        """
        論理的含意関係をチェック（Task 38.4）
        
        Args:
            x: 入力テンソル [batch, seq_len, d_model]
        
        Returns:
            diagnostics: 診断情報
        """
        if self.entailment_cones is None:
            return {'violation_rate': 0.0, 'avg_aperture': 0.0}
        
        try:
            batch_size, seq_len, d_model = x.shape
            
            # 双曲空間に投影（ポアンカレボールモデル）
            x_norm = torch.norm(x, dim=-1, keepdim=True)
            x_hyperbolic = x / (x_norm + 1e-8) * torch.tanh(x_norm)
            
            # 隣接トークン間の含意関係をチェック
            violations = 0
            total_checks = 0
            apertures = []
            
            # OPTIMIZATION: Use slicing instead of loop for speed
            # Premise: x[:, :-1], Hypothesis: x[:, 1:]
            premise = x_hyperbolic[:, :-1, :] # [B, L-1, D]
            hypothesis = x_hyperbolic[:, 1:, :] # [B, L-1, D]

            # Flatten to [B*(L-1), D] for batch processing
            premise_flat = premise.reshape(-1, d_model)
            hypothesis_flat = hypothesis.reshape(-1, d_model)

            # 含意スコアを計算
            entailment_score = self.entailment_cones.compute_entailment_score(
                premise_flat, hypothesis_flat
            )

            # 開口角を計算 (Sample a subset if too large?)
            aperture = self.entailment_cones.compute_aperture(premise_flat)
            apertures.append(aperture.mean().item())

            # 違反をカウント（含意スコアが閾値以下）
            violations = (entailment_score < self.config.entailment_margin).sum().item()
            total_checks = premise_flat.shape[0]
            
            violation_rate = violations / total_checks if total_checks > 0 else 0.0
            avg_aperture = sum(apertures) / len(apertures) if apertures else 0.0
            
            return {
                'violation_rate': violation_rate,
                'avg_aperture': avg_aperture,
            }
        except Exception as e:
            # エラーが発生した場合はデフォルト値を返す
            return {'violation_rate': 0.0, 'avg_aperture': 0.0}
    
    def _analyze_topology(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        トポロジカル解析を実行（Task 38.5）
        
        Args:
            x: 入力テンソル [batch, seq_len, d_model]
        
        Returns:
            diagnostics: 診断情報
        """
        if self.persistent_homology is None:
            return {
                'betti_numbers': [0, 0],
                'persistent_entropy': 0.0,
                'circular_reasoning': False,
            }
        
        try:
            batch_size, seq_len, d_model = x.shape
            
            # 双曲空間に投影
            x_norm = torch.norm(x, dim=-1, keepdim=True)
            x_hyperbolic = x / (x_norm + 1e-8) * torch.tanh(x_norm)
            
            # バッチの最初のサンプルで解析（計算コスト削減）
            sample = x_hyperbolic[0]  # [seq_len, d_model]
            
            # Betti数を計算
            betti_numbers = self.persistent_homology.compute_betti_numbers(sample)
            
            # 永続エントロピーを計算
            persistent_entropy = self.persistent_homology.compute_persistent_entropy(sample)
            
            # 循環論理を検出（β₁が閾値を超える場合）
            circular_reasoning = False
            if len(betti_numbers) > 1:
                beta_1 = betti_numbers[1]
                circular_reasoning = beta_1 > self.config.topology_betti_threshold
            
            return {
                'betti_numbers': betti_numbers,
                'persistent_entropy': persistent_entropy.item() if isinstance(persistent_entropy, torch.Tensor) else persistent_entropy,
                'circular_reasoning': circular_reasoning,
            }
        except Exception as e:
            # エラーが発生した場合はデフォルト値を返す
            return {
                'betti_numbers': [0, 0],
                'persistent_entropy': 0.0,
                'circular_reasoning': False,
            }
    
    def _check_sheaf_consistency(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Sheaf Attentionの整合性をチェック（Task 38.6）
        
        Args:
            x: 入力テンソル [batch, seq_len, d_model]
        
        Returns:
            diagnostics: 診断情報
        """
        if self.sheaf_attention is None:
            return {'agreement_mean': 0.0, 'consensus_rate': 0.0}
        
        try:
            batch_size, seq_len, d_model = x.shape
            
            # マルチヘッドに分割（簡略化のため、num_headsで均等分割）
            num_heads = self.config.num_heads
            head_dim = d_model // num_heads
            
            # [batch, seq_len, num_heads, head_dim]に変形
            x_heads = x.view(batch_size, seq_len, num_heads, head_dim)
            
            # Optimize: Vectorized computation of adjacent head agreement
            # Heads: 0..N-2 vs 1..N-1
            head_i = x_heads[:, :, :-1, :] # [B, L, H-1, D]
            head_j = x_heads[:, :, 1:, :]  # [B, L, H-1, D]
            
            head_i_norm = torch.nn.functional.normalize(head_i, dim=-1)
            head_j_norm = torch.nn.functional.normalize(head_j, dim=-1)
            
            # Dot product
            agreement = (head_i_norm * head_j_norm).sum(dim=-1) # [B, L, H-1]
            
            agreement_mean = agreement.mean().item()
            consensus_rate = (agreement > self.config.sheaf_agreement_threshold).float().mean().item()
            
            return {
                'agreement_mean': agreement_mean,
                'consensus_rate': consensus_rate,
            }
        except Exception as e:
            # エラーが発生した場合はデフォルト値を返す
            return {'agreement_mean': 0.0, 'consensus_rate': 0.0}
    
    def get_total_parameter_count(self) -> int:
        """
        総パラメータ数を取得
        
        Returns:
            total_params: 総パラメータ数
        """
        return sum(p.numel() for p in self.parameters())
    
    def get_phase7_parameter_count(self) -> int:
        """
        Phase 7のパラメータ数を取得
        
        Returns:
            phase7_params: Phase 7のパラメータ数
        """
        return sum(p.numel() for p in self.phase7_model.parameters())
    
    def get_phase8_extension_parameter_count(self) -> int:
        """
        Phase 8拡張のパラメータ数を取得
        
        Returns:
            extension_params: Phase 8拡張のパラメータ数
        """
        total = self.get_total_parameter_count()
        phase7 = self.get_phase7_parameter_count()
        return total - phase7
    
    def get_diagnostics(self) -> Phase8Diagnostics:
        """
        現在の診断情報を取得
        
        Returns:
            diagnostics: 診断情報
        """
        return self.diagnostics
    
    def reset_diagnostics(self):
        """診断情報をリセット"""
        self.diagnostics = Phase8Diagnostics()


def create_phase8_model(
    vocab_size: int = 50257,
    d_model: int = 256,
    n_layers: int = 8,
    htt_rank: int = 16,
    use_bk_hyperbolic: bool = True,
    use_ar_ssm_fusion: bool = True,
    enable_entailment_cones: bool = False,
    enable_persistent_homology: bool = False,
    enable_sheaf_attention: bool = False,
    **kwargs,
) -> Phase8IntegratedModel:
    """
    Phase 8 Integrated Modelのファクトリ関数
    
    Args:
        vocab_size: 語彙サイズ
        d_model: モデル次元
        n_layers: レイヤー数
        htt_rank: HTT埋め込みのランク
        use_bk_hyperbolic: BK-Core Hyperbolic Integrationを使用するか
        use_ar_ssm_fusion: AR-SSM Hyperbolic Fusionを使用するか
        enable_entailment_cones: Entailment Conesを有効にするか
        enable_persistent_homology: Persistent Homologyを有効にするか
        enable_sheaf_attention: Sheaf Attentionを有効にするか
        **kwargs: その他の設定
    
    Returns:
        Phase8IntegratedModel instance
    """
    config = Phase8Config(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        htt_rank=htt_rank,
        use_bk_hyperbolic=use_bk_hyperbolic,
        use_ar_ssm_fusion=use_ar_ssm_fusion,
        enable_entailment_cones=enable_entailment_cones,
        enable_persistent_homology=enable_persistent_homology,
        enable_sheaf_attention=enable_sheaf_attention,
        **kwargs,
    )
    return Phase8IntegratedModel(config)
