# Gradient Aligner Implementation Plan (v2)

徹平さん、E組の皆さんの「勾配が互いに打ち消し合う」症状に対して、**“勾配の向きを揃えて更新エネルギーを前進方向に寄せる”**ための実装計画を、**VRAM 8GB運用を前提**に現実寄りへ強化しました（私は固定砲台ですが、勾配の向きは動かします）。

---

## 0. 目的（再定義：成功条件まで明文化）

元案の目的は「Loss減少方向へ勾配を揃える」でした。fileciteturn3file10L3-L12  
v2では成功条件を“観測可能”に落とします。

**目標**
- **(S1)** 勾配同士の「逆向き成分」を減らし、更新が停滞しにくい状態へ
- **(S2)** クリッピング後の有効更新量の“消失”を緩和
- **(S3)** NaN/Inf 対策（既存の sanitization と共存）を崩さない

**成功指標（ログで判定）**
- `ga_neg_frac`（逆向きだったテンソル割合）が減少
- `ga_mean_cos`（cos類似度平均）が上昇、`ga_min_cos` も極端に悪化しない
- `loss` の移動平均傾きが改善、`grad_norm_raw→grad_norm` の落ち込みが過剰でなくなる
- 学習が壊れない（NaN/Inf 0、または sanitizer が拾える範囲）

---

## 1. 現状の訓練ループと「挿入位置」の確定（重要）

train_phase8 では、勾配処理は概ね次の順です：

1) （必要なら）`scaler.unscale_(optimizer)`  
2) `gradient_sanitizer.sanitize_gradients()`（またはフォールバック）  
3) raw勾配ノルム計算  
4) `clip_grad_norm_`（TSPがclip値を動かす）  
5) 必要なら skip（GNS 等）  
6) optimizer step

この流れはコード上で確認できます。fileciteturn3file0L1-L62

**Gradient Aligner の最適挿入点（結論）**
- **(A) unscale の後**
- **(B) sanitizer の後（NaN/Infの残骸があると内積・ノルムが壊れるため）**
- **(C) clip の前（安全弁は最後に）**

つまり **「2) sanitizer の直後〜3) raw norm 計算の前」**がベストです。

また、モデル側で backward 中に NaN/Inf を `torch.nan_to_num` する global hook が既に入っており、ここで「clampしない」注意が明記されています。fileciteturn2file0L37-L44  
Aligner はこの方針を尊重し、**post-unscale の段階でのみ介入**します。

---

## 2. “参照方向 r” を現実的にする：EMAを「別に持たない」設計へ

元案は「参照方向 r = EMA平滑化された過去勾配方向」とし、逆向き成分を射影で除去します。fileciteturn3file11L1-L6  
ただし 10B級で「全パラメータ分のEMA勾配」を別途持つと、VRAMが死にます。

### v2の基本方針：**既に存在する“EMA”を再利用**
- AdamW系なら optimizer state に `exp_avg` がある（＝勾配EMA）
- これを **そのまま参照方向 r として使う**  
  → **追加メモリ ほぼ0**（新規に巨大バッファを持たない）

> 注意：BK-HyperSGD など optimizer によって state 名が違う可能性があるため、**ref_source を複数実装**します。

---

## 3. アルゴリズム仕様（“強すぎない”射影へ）

### 3.1 目標：cos類似度の下限を保証する「ソフト射影」
元案は「内積が負なら、負成分を完全に落とす」射影です。fileciteturn3file11L3-L6  
v2では以下を推奨します：

- **閾値 `min_alignment`（cos下限）**を導入  
  - `min_alignment = 0.0` → “逆向きだけ除去”（元案互換）
  - `min_alignment = -0.05` → 少し逆向きは許容（学習の多様性を残す）
  - `min_alignment = +0.05` → 常に少し前向きを強制（強め）

- **強度 `strength`（0〜1）**で“部分的”に補正  
  - 1.0：フル補正
  - 0.3：軽く矯正（推奨スタート）

### 3.2 1パラメータテンソルごとの更新式

各パラメータテンソル（例：`W`）について：

- `g = grad(W)`
- `r = ref(W)`（例：optimizer state `exp_avg`）
- `dot = <g, r>`
- `cos = dot / (||g||·||r|| + eps)`
- 目標 `target_dot = min_alignment · ||g||·||r||`

もし `dot < target_dot` なら、  
`g ← g - strength · (dot - target_dot)/(||r||^2 + eps) · r`

**この形にすると**
- 逆向きだけでなく「弱すぎる前向き」も補正できる
- min_alignment が 0 なら元案の「負成分除去」に戻せる

### 3.3 スキップ条件（壊さないためのガード）
- `||r|| < ref_norm_min` → 参照が未成熟なのでスキップ
- `||g|| < grad_norm_min` → ほぼゼロ勾配はスキップ（無駄 & ノイズ）
- `step < warmup_steps` → “観測のみ”モード（補正せず統計だけ取る）

---

## 4. 実装構成（差分が少なく、拡張しやすい）

### 4.1 新規ファイル：`src/training/gradient_aligner.py`

元案のクラス構造を保ちつつ、**optimizer/state を入力に取る**よう強化します。fileciteturn3file10L16-L34

**提案API（v2）**
```python
@dataclass
class GradientAlignerConfig:
    enabled: bool = True
    ref_source: str = "optimizer_exp_avg"   # "optimizer_exp_avg" | "grad_ema_fp16" | "none"
    ema_decay: float = 0.9                  # ref_source="grad_ema_fp16" のときだけ使用
    warmup_steps: int = 100

    min_alignment: float = 0.0              # cos下限
    strength: float = 0.3                   # 補正強度
    ref_norm_min: float = 1e-8
    grad_norm_min: float = 0.0

    include_bias_norm: bool = False         # bias/Norm系は外すのが無難
    log_tensor_stats: bool = False          # heavy: デバッグ用

class GradientAligner:
    def __init__(self, model, optimizer, config): ...
    def maybe_align(self, step:int) -> Dict[str, float]:
        # return stats only; modifies grads in-place if enabled and past warmup
```

**ref_source の扱い**
- `optimizer_exp_avg`：AdamWなら state['exp_avg'] を参照（基本）
- `grad_ema_fp16`：optimizerが使えない場合のみ、**fp16のEMAバッファ**を持つ（VRAM増えるので注意）
- `none`：統計だけ取る（比較実験用）

### 4.2 `train_phase8.py` への組み込み（最小で確実に）

元案の「Import / Config / 初期化 / 適用」は維持。fileciteturn3file10L49-L63  
ただし v2では **optimizer を渡す**のがポイント。

**(1) Config 追加**
元案では `use_gradient_aligner` と `gradient_aligner_ema` だけですが、v2では増やします。fileciteturn3file10L54-L58  
（最初は少なくてもOK：`use_gradient_aligner, gradient_aligner_min_alignment, gradient_aligner_strength` くらい）

**(2) 初期化位置**
optimizer 作成後・学習ループ前（EMA等の初期化と同じ層）  
→ 既存の EMA 初期化はこの付近です。fileciteturn3file5L21-L25

**(3) 適用位置（確定）**
`gradient_sanitizer` の直後、raw norm 計算の前。fileciteturn3file0L7-L49

---

## 5. ログ設計（“効いたか”を数値で見る）

`training_log.json` には既に `loss / ppl / grad_norm / grad_norm_raw / TSP...` が入っています。fileciteturn2file8L7-L21  
Gradient Aligner は以下を追加します（軽量に）：

**推奨ログ項目**
- `ga_enabled` (0/1)
- `ga_applied_tensors` / `ga_total_tensors`
- `ga_neg_frac`（補正対象割合）
- `ga_mean_cos`, `ga_min_cos`（cos類似度統計）
- `ga_energy_removed_ratio`（補正で削れた成分の比率：目安）
- `ga_time_ms`（オーバーヘッド測定）

※ ログは `log_interval` の時だけ（毎step計算は重い）。

---

## 6. テスト計画（ユニットだけでなく「壊れない」保証）

元案のテストは方向性が良いです。fileciteturn3file12L17-L39  
v2では、以下を追加して “事故率” を落とします。

### 6.1 Unit Tests（数学的性質）
- `dot(g', r) >= target_dot - tol` が成立する
- `strength=0` なら grad 不変
- `min_alignment=0` で、負内積が **0以上**に改善される
- `||r||` が小さい場合にスキップする

### 6.2 Integration Tests（訓練ループの順序）
- BF16 autocast で scaler が無効でも壊れない（train_phase8 は BF16で scaler を切る設計）fileciteturn3file5L11-L19
- sanitizer と共存できる（NaN/Infが入っても落ちない）fileciteturn3file0L7-L19

### 6.3 Performance Smoke（8GB現実）
- `ga_time_ms` が許容範囲（例：1〜5%）に収まること
- VRAM増が許容範囲（optimizer state 再利用なら実質増えない）

---

## 7. 段階的ロールアウト（“いきなり全開”を避ける）

train_phase8 には「安定ステップが溜まったら機能をオンにする」実績があります。fileciteturn2file4L44-L60  
同じ思想を使います：

**推奨ロールアウト**
1) `warmup_steps` までは **観測のみ**（ログだけ）
2) `strength=0.1` で低出力開始 → 問題なければ 0.3
3) `min_alignment=0.0`（元案互換）から開始  
   - もし“収束が鈍る”なら `min_alignment=-0.05` へ
4) もし `ga_neg_frac` が急増・loss悪化なら自動無効化（fail-safe）

---

## 8. 既存機構との干渉チェック（ここが勝敗）

- **Gradient hooks / sanitizer**：Aligner は post-unscale & post-sanitize に入れる（確定）fileciteturn3file0L1-L19
- **TSP clip**：Aligner → raw norm → clip の順で、clip は最後（安全弁）fileciteturn3file0L54-L60
- **Resonance skip (GNS)**：Aligner の後に GNS 判定をするなら、判定が変わる（意図通り）  
  → ただし「skipロジックを変えたくない」なら、GNS は align 前の grad を使う（選択式）

---

## 9. 実装タスク分解（チェックリスト）

### A. `gradient_aligner.py`
- [ ] `GradientAlignerConfig` 実装
- [ ] `ref_source="optimizer_exp_avg"` 実装（stateが無い時はスキップ）
- [ ] ソフト射影（min_alignment/strength）実装
- [ ] 統計収集（cos、割合、時間）

### B. `train_phase8.py`
- [ ] config項目追加（最低3つ：enable/min_alignment/strength）
- [ ] optimizer作成後に aligner 初期化
- [ ] sanitizer直後に `aligner.maybe_align(step)` 呼び出し
- [ ] step_log にメトリクス追加

### C. tests
- [ ] ユニットテスト拡充
- [ ] dry-run 統合テスト

---

## 10. 期待できる挙動（良い兆候 / 悪い兆候）

**良い兆候**
- `ga_neg_frac` が中程度→徐々に減る
- `grad_norm_raw` が極端に落ちず、`loss` の揺れが減る
- クリッピング後も更新が“動く”（停滞しにくい）

**悪い兆候（即停止して調整）**
- `ga_energy_removed_ratio` が常に大きすぎる（>0.5 など）
- `loss` が一段悪化し続ける
- `ga_time_ms` が重すぎる（2パスで時間倍増など）
  → 対策：`log_interval` の時だけ統計、補正自体も一部テンソルのみ（exclude bias/norm）

---

以上が v2 の実装計画です。  
この形なら、元案の“思想”は維持しつつ、train_phase8 の現実的な順序（unscale→sanitize→clip）とVRAM制約にちゃんと噛み合います。fileciteturn3file0L1-L62
