# 🔬 実験検証計画 - 論文主張の実証

## 📊 現状の評価に対する対応

### 指摘された懸念点

1. **実験データの不足** - モックデータに依存
2. **Mambaとの直接比較の欠如** - 実際の比較結果なし
3. **理論と実装のギャップ** - 理論的保証の実証不足

## 🎯 必須実験リスト

### Phase 1: 基本検証（1-2日）

#### 1.1 小規模モデルでの動作確認
```powershell
# WikiText-2での基本訓練（数時間）
python scripts/train_epsilon_family.py \
    --model resnet_bk \
    --dataset wikitext2 \
    --d_model 256 \
    --num_layers 6 \
    --batch_size 32 \
    --max_steps 10000 \
    --output results/validation/small_model_wikitext2.json

# Mambaベースライン
python scripts/train_epsilon_family.py \
    --model mamba \
    --dataset wikitext2 \
    --d_model 256 \
    --num_layers 6 \
    --batch_size 32 \
    --max_steps 10000 \
    --output results/validation/mamba_baseline_wikitext2.json
```

**期待される結果:**
- ResNet-BK PPL: 目標 25-30
- Mamba PPL: 目標 30-35
- 訓練安定性の確認

#### 1.2 理論的保証の検証
```powershell
# Schatten境界の監視
python tests/test_theory.py --verbose --log-schatten-norms

# GUE統計の検証
python examples/prime_bump_demo.py --verify-gue-statistics

# LAP安定性の確認
python examples/mourre_lap_demo.py --verify-stability
```

**期待される結果:**
- Schatten-2ノルム < 理論的上限
- 固有値分布がGUE統計に従う
- LAP条件が満たされる

### Phase 2: 長文脈安定性（2-3日）

#### 2.1 段階的シーケンス長拡張
```powershell
# 8k tokens
python scripts/train_long_context.py \
    --model resnet_bk \
    --seq_length 8192 \
    --seeds 42,43,44 \
    --output results/real_experiments/long_context_8k.json

# 32k tokens
python scripts/train_long_context.py \
    --model resnet_bk \
    --seq_length 32768 \
    --seeds 42,43,44 \
    --output results/real_experiments/long_context_32k.json

# Mamba比較（32kで発散予測）
python scripts/train_long_context.py \
    --model mamba \
    --seq_length 32768 \
    --seeds 42,43,44 \
    --output results/real_experiments/mamba_long_context_32k.json
```

**期待される結果:**
- ResNet-BK: 32kでの安定性を検証
- Mamba: 32kでの挙動を確認
- 損失曲線の比較

#### 2.2 超長文脈（128k+）
```powershell
# 128k tokens（Google Colab Pro推奨）
python scripts/train_long_context.py \
    --model resnet_bk \
    --seq_length 131072 \
    --gradient_checkpointing \
    --mixed_precision \
    --output results/real_experiments/long_context_128k.json
```

### Phase 3: 量子化ロバスト性（1-2日）

#### 3.1 量子化実験
```powershell
# FP32ベースライン
python scripts/benchmarks/run_quantization_sweep.py \
    --model resnet_bk \
    --bits FP32 \
    --dataset wikitext2 \
    --seeds 42,43,44,45,46 \
    --output results/real_experiments/quant_fp32.json

# INT8量子化
python scripts/benchmarks/run_quantization_sweep.py \
    --model resnet_bk \
    --bits INT8 \
    --dataset wikitext2 \
    --seeds 42,43,44,45,46 \
    --output results/real_experiments/quant_int8.json

# INT4量子化（主張の核心）
python scripts/benchmarks/run_quantization_sweep.py \
    --model resnet_bk \
    --bits INT4 \
    --dataset wikitext2 \
    --seeds 42,43,44,45,46 \
    --output results/real_experiments/quant_int4.json

# Mamba INT4（比較）
python scripts/benchmarks/run_quantization_sweep.py \
    --model mamba \
    --bits INT4 \
    --dataset wikitext2 \
    --seeds 42,43,44,45,46 \
    --output results/real_experiments/mamba_quant_int4.json
```

**期待される結果:**
- ResNet-BK INT4 PPL: 目標 40-50
- Mamba INT4 PPL: 目標 180-200
- 優位性の検証

### Phase 4: 効率性測定（1日）

#### 4.1 FLOPs測定
```powershell
# ResNet-BK FLOPs
python scripts/benchmarks/measure_flops.py \
    --models resnet_bk,resnet_bk_act \
    --seq_length 2048 \
    --seeds 42,43,44,45,46 \
    --output results/real_experiments/flops_resnet_bk.json

# Mamba FLOPs
python scripts/benchmarks/measure_flops.py \
    --models mamba \
    --seq_length 2048 \
    --seeds 42,43,44,45,46 \
    --output results/real_experiments/flops_mamba.json
```

**期待される結果:**
- ResNet-BK: 目標 2.5 GFLOPs/token
- ResNet-BK+ACT: 目標 1.8 GFLOPs/token
- Mamba: 目標 3.2 GFLOPs/token
- 効率性の検証

#### 4.2 スループット測定
```powershell
python scripts/benchmarks/measure_throughput.py \
    --models resnet_bk,mamba \
    --batch_sizes 1,4,8,16 \
    --seq_lengths 512,1024,2048,4096 \
    --output results/real_experiments/throughput.json
```

### Phase 5: アブレーション研究（1-2日）

#### 5.1 コンポーネント別評価
```powershell
# 完全モデル
python scripts/benchmarks/run_ablation.py \
    --components prime_bump,scattering_router,lap_stability,semiseparable \
    --seeds 42,43,44,45,46 \
    --output results/real_experiments/ablation_full.json

# Prime-Bump除外
python scripts/benchmarks/run_ablation.py \
    --components scattering_router,lap_stability,semiseparable \
    --seeds 42,43,44,45,46 \
    --output results/real_experiments/ablation_no_prime_bump.json

# Scattering Router除外
python scripts/benchmarks/run_ablation.py \
    --components prime_bump,lap_stability,semiseparable \
    --seeds 42,43,44,45,46 \
    --output results/real_experiments/ablation_no_scattering.json

# LAP Stability除外
python scripts/benchmarks/run_ablation.py \
    --components prime_bump,scattering_router,semiseparable \
    --seeds 42,43,44,45,46 \
    --output results/real_experiments/ablation_no_lap.json

# Semiseparable除外
python scripts/benchmarks/run_ablation.py \
    --components prime_bump,scattering_router,lap_stability \
    --seeds 42,43,44,45,46 \
    --output results/real_experiments/ablation_no_semiseparable.json
```

**期待される結果:**
- 各コンポーネントの寄与を定量化
- Prime-Bump: 収束速度への影響を測定
- Scattering Router: ルーティング速度を測定
- LAP: 長文脈安定性への寄与を確認
- Semiseparable: メモリ効率への寄与を確認

## 📅 実験スケジュール

### 最小限の検証（3-4日）
```
Day 1: Phase 1 (基本検証) + Phase 4 (効率性)
Day 2: Phase 2.1 (8k-32k長文脈)
Day 3: Phase 3 (量子化)
Day 4: Phase 5 (アブレーション) + 結果統合
```

### 完全な検証（1-2週間）
```
Week 1:
  - Phase 1-4の完全実行
  - 複数シード（5-10）での再現性確認
  - 統計的有意性検定

Week 2:
  - Phase 2.2 (超長文脈 128k-1M)
  - 追加ベースライン（Transformer、RWKV）
  - 下流タスク評価
```

## 🎯 成功基準

### 必須（論文投稿に必要）

1. **長文脈安定性**
   - ✓ ResNet-BK: 32kで安定訓練
   - ✓ Mamba: 32kで発散
   - ✓ 統計的有意性 p < 0.01

2. **量子化ロバスト性**
   - ✓ ResNet-BK INT4 PPL < 50（目標）
   - ✓ Mamba INT4 PPL > 150（目標）
   - ✓ 統計的に有意な優位性

3. **効率性**
   - ✓ ResNet-BK FLOPs < Mamba FLOPs
   - ✓ 統計的に有意な効率性の向上
   - ✓ 同等PPLでの比較

### 望ましい（より強い主張）

4. **超長文脈（128k+）**
   - ○ 128kで安定訓練
   - ○ 512kで動作
   - ○ 1Mで実験的検証

5. **理論的保証の実証**
   - ○ Schatten境界の監視
   - ○ GUE統計の確認
   - ○ LAP条件の検証

## 🚀 クイックスタート

### 最小限の実験（数時間）
```powershell
# 基本検証のみ
python scripts/benchmarks/quick_validation.py \
    --models resnet_bk,mamba \
    --dataset wikitext2 \
    --quick \
    --output results/quick_validation.json
```

### 標準実験（3-4日）
```powershell
# 全Phase実行
.\scripts\benchmarks\run_all_paper_experiments.ps1
```

### 完全実験（1-2週間）
```powershell
# 拡張実験含む
.\scripts\benchmarks\run_comprehensive_experiments.ps1
```

## 📊 結果の統合

実験完了後：

```powershell
# 図の再生成（実データ使用）
python scripts/benchmarks/generate_stability_graph.py \
    --results_dir results/real_experiments \
    --output paper/figures/figure1_stability.pdf

python scripts/benchmarks/generate_quantization_graph.py \
    --results_dir results/real_experiments \
    --output paper/figures/figure2_quantization.pdf

python scripts/benchmarks/generate_efficiency_graph.py \
    --results_dir results/real_experiments \
    --output paper/figures/figure3_efficiency.pdf

# テーブルの再生成
python scripts/benchmarks/generate_paper_tables.py \
    --results_dir results/real_experiments \
    --output paper/generated_tables.tex

# 統計的有意性検定
python scripts/benchmarks/statistical_tests.py \
    --results_dir results/real_experiments \
    --output paper/statistical_analysis.tex
```

## 💪 リスク軽減策

### もし主張が実証されない場合

1. **長文脈で発散する場合**
   - 主張を「32kまで安定」に修正
   - ハイパーパラメータチューニング
   - グラディエントクリッピング強化

2. **量子化で期待した優位性が出ない場合**
   - 実際の測定値に基づいて主張を修正
   - INT8での優位性を強調
   - 量子化aware訓練の追加

3. **効率性で期待した優位性が出ない場合**
   - ACT（適応的計算時間）を強調
   - メモリ効率を主張
   - 特定のワークロードでの優位性

## 📝 論文への反映

実験完了後、以下を更新：

1. **Abstract**: 実際の数値に更新
2. **Introduction**: 主張を実証データに基づいて調整
3. **Experiments**: 実際の結果を記載
4. **Discussion**: 理論と実験の対応を議論
5. **Conclusion**: 実証された貢献を明確化

---

**現在の状態**: 実験インフラは完備。実行準備完了。

**推奨**: Phase 1（基本検証）から開始し、結果を見て次のPhaseに進む。

**所要時間**: 最小3-4日、理想的には1-2週間。
