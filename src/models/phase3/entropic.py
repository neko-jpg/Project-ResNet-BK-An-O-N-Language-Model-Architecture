"""
Entropic Data Selection for Phase 3: Physics Transcendence

このモジュールは、熱力学的サプライズ（損失値）に基づくデータ選別機能を実装します。
学習データを効率的に圧縮し、Phase 3モデルの学習効率を飛躍的に高めます。

Requirements:
    - Requirement 6.1: Surprise Calculation (Loss)
    - Requirement 6.2: Curriculum Warmup
    - Requirement 6.3: Top-k Selection
    - Requirement 6.4: Diversity Guardrail
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import math

class EntropicSelector:
    """
    Entropic Data Selector

    データセットの各サンプルの「熱力学的サプライズ（損失）」を計算し、
    情報量の多い（＝サプライズの大きい）サンプルのみを選別します。

    Args:
        model: 損失計算に使用するモデル（通常はPhase 2モデル）
        selection_rate: データ保持率（例: 0.1 = 10%保持）
        warmup_epochs: 全データを使用するウォームアップ期間
        diversity_lambda: 多様性ガードレールの強度（0.0 = 無効）
    """
    def __init__(
        self,
        model: nn.Module,
        selection_rate: float = 0.1,
        warmup_epochs: int = 5,
        diversity_lambda: float = 0.0
    ):
        self.model = model
        self.selection_rate = selection_rate
        self.warmup_epochs = warmup_epochs
        self.diversity_lambda = diversity_lambda
        self.current_epoch = 0

    def step_epoch(self, epoch: int):
        """エポックの更新"""
        self.current_epoch = epoch

    def select_data(
        self,
        dataset_iterator,
        criterion: nn.Module = nn.CrossEntropyLoss(reduction='none')
    ) -> List[Any]:
        """
        データの選別を実行

        Args:
            dataset_iterator: データセットイテレータ (batch単位で返すことを想定)
            criterion: 損失関数（reduction='none'必須）

        Returns:
            selected_indices: 選別されたデータのインデックスリスト

        Note:
            実際のデータローダーパイプラインでは、
            1. 小さなバッチで損失計算（推論のみ、高速）
            2. スコア上位のサンプルのみを学習用バッチに組み込む
            という流れになります。
            ここでは、スコア計算と選別のロジックを提供します。
        """
        # ウォームアップ期間は全データを使用
        if self.current_epoch < self.warmup_epochs:
            return None # None means "use all data"

        # ここでは単純化のため、バッチごとの処理例を示すメソッドを提供する形にする
        # 実際にはDataLoaderのSamplerとして実装するのが一般的だが、
        # モデル依存の動的なSamplingが必要なため、Trainer側で呼び出すヘルパーとする
        pass

    def compute_surprise(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module = nn.CrossEntropyLoss(reduction='none')
    ) -> torch.Tensor:
        """
        サプライズ（損失）の計算

        Args:
            input_ids: (B, N)
            targets: (B, N)

        Returns:
            surprise_scores: (B,) 各サンプルの平均損失
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_ids)

            # logits: (B, N, V), targets: (B, N)
            # CrossEntropyLoss expects (B, V, N) or (B, C)
            # Flattening is usually done: (B*N, V) vs (B*N)
            # But we want per-sample score.

            # Permute logits to (B, V, N) for loss calculation per token
            if logits.dim() == 3:
                loss = criterion(logits.permute(0, 2, 1), targets) # (B, N)
            else:
                # Fallback for specific outputs
                loss = criterion(logits, targets)

            # サンプルごとの平均損失（サプライズ）
            # パディングトークンを無視するなどの処理が必要だが、
            # ここでは単純平均とする
            surprise_scores = loss.mean(dim=-1) # (B,)

        return surprise_scores

    def filter_batch(
        self,
        batch: Dict[str, torch.Tensor],
        criterion: nn.Module = nn.CrossEntropyLoss(reduction='none')
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        バッチ内のデータを選別

        Args:
            batch: {'input_ids': ..., 'labels': ...}

        Returns:
            filtered_batch: 選別されたバッチ
            stats: 統計情報
        """
        # ウォームアップ期間はそのまま返す
        if self.current_epoch < self.warmup_epochs:
            return batch, {'kept_ratio': 1.0}

        input_ids = batch['input_ids']
        labels = batch.get('labels', input_ids) # ラベルがない場合は入力と同じと仮定（事前学習）

        # サプライズ計算
        surprise_scores = self.compute_surprise(input_ids, labels, criterion)

        # 選別数
        batch_size = input_ids.size(0)
        k = max(1, int(batch_size * self.selection_rate))

        # Diversity Guardrail (Requirement 6.4)
        # 単純なトップk選別だと分布が偏るため、
        # 本来はクラスタリングやトピックモデルと組み合わせるが、
        # ここではランダムサンプリングを混ぜることで簡易的に実装
        if self.diversity_lambda > 0:
            # スコアにランダムノイズを加えて多様性を確保
            noise = torch.randn_like(surprise_scores) * (surprise_scores.std() * self.diversity_lambda)
            adjusted_scores = surprise_scores + noise
            _, top_indices = torch.topk(adjusted_scores, k)
        else:
            _, top_indices = torch.topk(surprise_scores, k)

        # データのフィルタリング
        filtered_batch = {
            key: val[top_indices] for key, val in batch.items() if torch.is_tensor(val) and val.size(0) == batch_size
        }

        stats = {
            'kept_ratio': k / batch_size,
            'mean_surprise_kept': surprise_scores[top_indices].mean().item(),
            'mean_surprise_all': surprise_scores.mean().item()
        }

        return filtered_batch, stats
