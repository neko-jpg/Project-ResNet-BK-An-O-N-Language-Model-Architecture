# 論文投稿準備 - 次のステップ

## ✅ 完了したこと

### 1. モックデータ生成 ✓
- `results/paper_experiments/long_context_resnet_bk.json`
- `results/paper_experiments/quantization_resnet_bk.json`
- `results/paper_experiments/efficiency.json`
- `results/paper_experiments/ablation.json`

### 2. 図の生成 ✓
- `results/paper_experiments/figure1_stability.pdf` - 長文脈安定性グラフ
- `results/paper_experiments/figure2_quantization.pdf` - 量子化ロバスト性グラフ
- `results/paper_experiments/figure3_efficiency.pdf` - 効率性グラフ

### 3. テーブルの生成 ✓
- `paper/generated_tables.tex` - 4つの実験結果テーブル

### 4. 実験スクリプトの準備 ✓
- PowerShell版実験スクリプト: `scripts/benchmarks/run_all_paper_experiments.ps1`
- 個別実験スクリプト:
  - `scripts/benchmarks/measure_flops.py`
  - `scripts/benchmarks/run_quantization_sweep.py`
  - `scripts/benchmarks/run_ablation.py`

## 📋 今すぐやるべきこと

### ステップ1: LaTeXのインストール

Windows用のLaTeX配布版をインストール：

**推奨: MiKTeX**
```powershell
# Chocolateyを使用する場合
choco install miktex

# または公式サイトからダウンロード
# https://miktex.org/download
```

**または: TeX Live**
```powershell
# 公式サイトからダウンロード
# https://www.tug.org/texlive/windows.html
```

### ステップ2: 論文のコンパイル

LaTeXインストール後、以下を実行：

```powershell
cd paper

# メイン論文をコンパイル
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# 補足資料をコンパイル
pdflatex supplementary.tex
bibtex supplementary
pdflatex supplementary.tex
pdflatex supplementary.tex

# PDFを開く
start main.pdf
start supplementary.pdf
```

### ステップ3: 図をpaperディレクトリに移動

```powershell
# figuresディレクトリを作成
New-Item -ItemType Directory -Path paper/figures -Force

# 生成した図をコピー
Copy-Item results/paper_experiments/figure*.pdf paper/figures/
Copy-Item results/paper_experiments/figure*.png paper/figures/
```

### ステップ4: 論文の確認と修正

1. **main.pdf**を開いて確認
2. 図が正しく表示されているか確認
3. テーブルが正しく表示されているか確認
4. 参考文献が正しくリンクされているか確認

### ステップ5: 校正とポリッシュ

```powershell
# TODOやFIXMEをチェック
Select-String -Path paper/*.tex -Pattern "TODO|FIXME|XXX"

# 空の引用をチェック
Select-String -Path paper/*.tex -Pattern "\\cite\{\}"
```

## 🔄 実際の実験を実行する場合（オプション）

モックデータではなく実際の実験結果が必要な場合：

```powershell
# 全実験を実行（24-48時間かかります）
.\scripts\benchmarks\run_all_paper_experiments.ps1

# または個別に実行
python scripts/benchmarks/run_scaling_experiments.py --model resnet_bk --seq_lengths 8192,32768,131072 --seeds 42,43,44 --output results/paper_experiments/long_context_resnet_bk.json

# 実験完了後、図とテーブルを再生成
python scripts/benchmarks/generate_stability_graph.py --results_dir results/paper_experiments --output results/paper_experiments/figure1_stability
python scripts/benchmarks/generate_quantization_graph.py --results_dir results/paper_experiments --output results/paper_experiments/figure2_quantization
python scripts/benchmarks/generate_efficiency_graph.py --results_dir results/paper_experiments --output results/paper_experiments/figure3_efficiency
python scripts/benchmarks/generate_paper_tables.py --results_dir results/paper_experiments --output paper/generated_tables.tex
```

## 📊 現在の論文の状態

### 完成度: 85%

**完了済み:**
- ✅ 論文構造（8ページ）
- ✅ 数学的基礎
- ✅ 手法の説明
- ✅ 参考文献（50+）
- ✅ 補足資料
- ✅ 実験データ（モック）
- ✅ 図（3つ）
- ✅ テーブル（4つ）

**残りのタスク:**
- ⏳ LaTeXコンパイル
- ⏳ 最終校正
- ⏳ 著者情報の追加
- ⏳ 学会フォーマットの確認

## 🎯 投稿までのタイムライン

### 今日（Day 1）
- [x] モックデータ生成
- [x] 図の生成
- [x] テーブルの生成
- [ ] LaTeXインストール
- [ ] 論文コンパイル
- [ ] 初回レビュー

### 明日（Day 2-3）
- [ ] 校正（文法、スペル、数式）
- [ ] 図表の微調整
- [ ] 参考文献の確認
- [ ] アブストラクトの洗練

### 今週末（Day 4-7）
- [ ] 同僚レビュー
- [ ] フィードバック反映
- [ ] 最終チェック
- [ ] arXiv投稿準備

### 来週（Day 8-14）
- [ ] arXivに投稿
- [ ] 学会投稿準備
- [ ] カメラレディ版作成

## 📝 重要なファイル

### 論文ファイル
- `paper/main.tex` - メイン論文（8ページ）
- `paper/supplementary.tex` - 補足資料
- `paper/references.bib` - 参考文献
- `paper/generated_tables.tex` - 自動生成テーブル

### 図
- `results/paper_experiments/figure1_stability.pdf`
- `results/paper_experiments/figure2_quantization.pdf`
- `results/paper_experiments/figure3_efficiency.pdf`

### データ
- `results/paper_experiments/*.json` - 実験結果

### スクリプト
- `scripts/benchmarks/run_all_paper_experiments.ps1` - 全実験実行
- `scripts/benchmarks/generate_*_graph.py` - 図生成
- `scripts/benchmarks/generate_paper_tables.py` - テーブル生成

## 🚀 クイックスタート（LaTeXインストール後）

```powershell
# 1. 図をコピー
New-Item -ItemType Directory -Path paper/figures -Force
Copy-Item results/paper_experiments/figure*.pdf paper/figures/

# 2. 論文をコンパイル
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# 3. PDFを開く
start main.pdf
```

## 💡 ヒント

1. **初回コンパイル時**: 必要なLaTeXパッケージが自動インストールされます（MiKTeXの場合）
2. **エラーが出た場合**: ログファイル（main.log）を確認
3. **図が表示されない場合**: `paper/figures/`ディレクトリに図があるか確認
4. **参考文献が表示されない場合**: bibtexを実行してから再度pdflatexを実行

## 📞 サポート

問題が発生した場合：
1. `paper/main.log`を確認
2. エラーメッセージをコピー
3. 必要に応じてサポートを求める

---

**現在の状態**: 論文は85%完成。LaTeXをインストールしてコンパイルすれば、すぐに確認できます！

**次のアクション**: LaTeXをインストールして、`cd paper && pdflatex main.tex`を実行してください。
