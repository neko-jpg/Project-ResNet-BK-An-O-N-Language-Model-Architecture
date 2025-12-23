# Phase 8 (10B Japanese) — Lossが減少しない理由のコードレビュー（調査メモ）

対象:
- ログ: `checkpoints/phase8_10b_japanese/training_log.json`（直近≈1000 step分）
- 学習スクリプト: `scripts/train_phase8.py`
- モデル: `src/models/phase8/integrated_model.py` / `src/models/phase7/*` / `src/models/resnet_bk.py`
- データ: `src/utils/data_utils.py` + `configs/dataset_mixing.yaml` + `data/*`
- TSP: `src/training/tsp_path_optimizer.py`

前提:
- SGD系の学習では **Lossが「単調減少」することは基本的に期待できません**（ミニバッチの難易度やノイズで上下します）。
- ただし今回のログ窓（約1000 step）では、平滑化しても「下がる方向」が見えにくく、**更新量が小さすぎる/設定が噛み合っていない**可能性が高いです。

---

## 1. `training_log.json`（直近1000 step）の事実整理

ログから読み取れる重要点（数値は当環境で解析）:

- 記録件数: **62件**
  - `step` 範囲: **16880 → 17856**（差分 976 ≒「直近1000 step」）
  - `optimizer_step`: **1 → 62**
  - つまり、このログは「1000回のoptimizer更新」ではなく、**≈62回のoptimizer更新**の窓です。
    - さらに、ログの `step` 間隔が 16 なので、実際の `grad_accum_steps` は **16相当**に見えます（後述: 設定読み込みの罠があります）。

- Loss統計:
  - 平均: **9.1535** / 標準偏差: **0.2481**
  - 最小: **8.5195**（step 17488）
  - 最大: **9.6299**（step 17760）
  - 初回→最終: **9.1096 → 9.3710**（この窓では改善していない）
  - 移動平均（例: 10点）でも **増加傾向**が見えます（= “下がる方向”が見えない）。

- 勾配ノルムとクリップ:
  - `grad_clip` は全点 **1.0**
  - `grad_norm > grad_clip`（クリップが実際に効く状態）が **約79%**（49/62）
  - つまり更新前に **勾配がかなりの頻度で縮小**されています。

- LR:
  - ログの `lr` は **≈0.05**（`scripts/train_phase8.py` の scheduler の値）
  - 同時に `tsp_effective_lr` は **0.005 固定**（`tsp_lr_scale=0.1`）
  - 重要: `scripts/train_phase8.py` は **optimizerに実際に設定したLR（TSP適用後）ではなく、schedulerのLRをログに書いています**。  
    そのため「lr=0.05で回しているのに…」に見えても、実際は `0.005` で更新されている可能性が高いです。

---

## 2. 「Lossが単調減少しない」は正常（ただし今回は“下がらない”寄り）

まず一般論:
- ミニバッチ学習の loss は、**毎ステップのバッチ難易度が違う**ため上下して当然です。
- `batch_size=1`（+ 混合データ + 長さ512）だと、バッチ間の難易度差が大きく、**見かけの loss が荒れます**。
- `label_smoothing=0.1`（`scripts/train_phase8.py`）は、損失に一定の揺らぎと下限を入れるので、より「単調減少」しにくくします。

ただし今回のログ窓は:
- 1000 step（≒62 optimizer update）程度の短い窓でも、**移動平均がほぼ下がっていない**ので、
  - 「単調ではない」よりも **「更新が弱すぎる/設定が噛み合っていない」**側の疑いが濃いです。

---

## 3. コードレビュー（Lossが下がらない/下がりにくい“強い”要因）

### 3.1 最重要: GradScaler + “勾配フックでの常時clamp” が実質的に更新量を潰す

該当箇所:
- `scripts/train_phase8.py` の `create_model()` 内で、全パラメータに `param.register_hook(...)` を仕込んでいます。
  - NaN/Inf対策だけでなく、**常に** `torch.clamp(grad, -10, 10)` が走ります。
- 学習ループ側は `scaler.scale(loss).backward()`（AMP+GradScaler）→ `scaler.unscale_(optimizer)` の順です。

ここが致命的になり得ます:
- GradScalerは loss をスケール `S`（デフォルト初期は **65536**）で拡大します。
- その結果、各パラメータの勾配も `S`倍になった状態でフックに到達します。
- フックが **±10 にclamp**してしまうと、スケール済み勾配は最大でも 10 に固定されます。
- その後 `unscale_` で `S` で割られるので、最終的な勾配は最大でも `10/S`（例: 10/65536 ≈ **1.5e-4**）になります。

つまり:
- **GradScalerの存在と、フックでの固定clampが組み合わさると、意図せず“極小勾配”になります。**
- 大規模パラメータ数（数億）では、`grad_norm` 自体は 1〜3 程度に見えても、1要素あたりは ~1e-4 オーダーになり得て、**実際の重み更新がほぼ進まない**状況が起きます。

これは今回のログとも整合します:
- `grad_norm` は 1〜3 台（大きくはない）
- `grad_clip=1.0` が高頻度で発動（さらに縮む）
- さらに TSP で `tsp_effective_lr=0.005`（ログ上確認）  
  → **総合すると更新がかなり小さい**可能性が高いです。

### 3.2 クリップ/安全装置が多段で入り、更新が“常に抑制”されやすい

更新を抑制する仕組みが複数あります:
- `scripts/train_phase8.py`
  - `clip_grad_norm_(..., grad_clip_train or tsp_city.clip_value)`（ログ上ほぼ 1.0）
- `src/optimizers/bk_hyper_sgd.py`
  - `max_grad_norm` によるパラメータ毎クリップ
- モデル内部
  - `src/models/bitnet.py` の `tanh` でのsoft clamp
  - `src/models/resnet_bk.py` の `torch.clamp / nan_to_num` 多用

個々は安全装置として理解できますが、積み重なると **「安定だが進まない」**典型になります。

### 3.3 LRが“ログと実態でズレる”（TSPが最終LRを上書き）

該当箇所:
- `scripts/train_phase8.py` で scheduler のあとに TSP が `group['lr']=tsp_effective_lr` を上書きします。
- 一方でログの `lr` は scheduler の `get_last_lr()` を保存しており、**TSP適用後の実LRではありません**。

結果:
- loss曲線を見て「lrは0.05で十分大きい」と判断しやすいが、実態は `tsp_effective_lr`（今回ログ上 0.005）になっている可能性が高いです。

### 3.4 設定読み込みの罠: `gradient_accumulation_steps` が YAMLから反映されない可能性

該当箇所:
- `scripts/train_phase8.py` の `parse_args()` 内、`get_val()` の仕様と
  - `grad_accum_steps=get_val('grad_accum_steps', get_val('gradient_accumulation_steps', 16))`

この形は、YAML側で `gradient_accumulation_steps` を変えても、CLI側のデフォルト（16）が “defaultと違う” 扱いになって上書きされ、**結局16になる**挙動を生みます。

今回ログの `step` 間隔が 16 なので、まさにこの影響を受けている可能性があります。

影響:
- 想定より optimizer update が少ない（= lossが下がるのが遅く見える）
- “effective batch size” の想定が外れ、LR設計が噛み合わない

### 3.5 TSPの評価周期が “optimizer step” ではなく “micro step” と混ざっている

` に **globalの `step`**（= micro step）を渡しています。
一方で `record()` は optimizer update ごとにしか呼ばれないため、`tsp_eval_interval` の意味が `grad_accum_steps` と混ざって変質します。

これにより:
- 意図より早く “収束(city)” 側へ行く
- `lr_scale` が保守的になり、更新がさらに弱くなる  
…などが起こり得ます。

---

## 4. データ/設定レビュー（Lossが高止まりしやすい要因）

### 4.1 「Japanese設定」だが、実データは GPT-2語彙（≈50k token id）

`data/*/train.bin` をサンプル計測すると、各データセットの `max token id` は概ね **5024x〜5025x** でした。
これは語彙が `~50257` の GPT-2 系 token id であることを示唆します。

一方で `configs/phase8_10b_japanese.yaml` には:
- `tokenizer_name: "rinna/japanese-gpt-neox-3.6b"`
- `vocab_size: 32000`

しかし:
- `scripts/train_phase8.py` は `tokenizer_name` を参照していません（= 設定として置かれているだけ）。
- `src/utils/data_utils.py` は token id の最大値に合わせて `vocab_size` を拡張し得ます。

結果として:
- “32k Japanese tokenizerで学習している” つもりでも、実態は “50k GPT-2 token id” の可能性が高いです。
  - 日本語を GPT-2 BPE で扱うと、一般に token 列が長くなりやすく、学習が難しくなり、lossも下がりにくい傾向があります。

### 4.2 データ混合が日本語特化ではない

`configs/dataset_mixing.yaml` は英語/コード系（例: `evol_instruct_code`）を大きく混ぜます。
特に `evol_instruct_code` の重みが 0.33 と大きく、文体/分布がばらつきやすい構成です。

この場合:
- ステップごとの難易度差が増え、lossが “ギザギザ” になる
- 日本語に特化した低lossへ下がるには時間がかかる  
…が起きやすいです。

---

## 5. 追加で確認すると切り分けが早い（コード修正なしで可能）

1) **GradScalerのscale値を確認**  
`scaler.get_scale()` をログ出しすると、`create_model()` の勾配clampとの相互作用が起きているか判断しやすいです。

2) **“重みが本当に動いているか” を確認**  
`scripts/train_phase8.py` 内にコメントアウトされた `weight_before/after` のデバッグがあります（復活させて数stepだけ確認）。

3) **固定バッチでのloss（evaluation）**  
データのばらつきを除き、「平均lossが下がるか」を見るのが最短です。

4) **実際のgrad_accum_stepsの確認**  
ログの `step` 間隔（=16）と、起動時に表示される `Grad Accum:` を突き合わせると、YAML反映の問題が分かります。

---

## 6. 結論（優先度順）

今回の “lossが減少方向に向かわない” の最有力要因は次の組み合わせです:

1. **GradScaler使用中に、全パラメータへ「常時±10 clamp」の勾配フックが入っている**  
   → unscale後の勾配が極端に小さくなり、更新が進まない可能性が高い。
2. **勾配クリップが高頻度で発動（clip=1.0）**  
   → さらに更新が縮む。
3. **TSPにより実効LRが 0.005（ログ上）に下げられている**  
   → “思ったより小さいLR”で進む。
4. **`gradient_accumulation_steps` が YAMLから反映されず 16 になる可能性**  
   → optimizer update 数が想定より減り、曲線が鈍く見える。

上記が解消されない限り、学習は「安定だが進まない」方向に寄りやすく、lossは単調減少しません（むしろ停滞・微増すらあり得ます）。

