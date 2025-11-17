# 📄 論文コンパイル手順

## ✅ 準備完了

- [x] 論文構造（main.tex、supplementary.tex）
- [x] 参考文献（references.bib）
- [x] 図（paper/figures/*.pdf）
- [x] テーブル（generated_tables.tex）
- [x] LaTeXインストール中

## 🚀 コンパイル手順

### 方法1: コマンドライン（推奨）

```powershell
# paperディレクトリに移動
cd paper

# メイン論文をコンパイル
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# PDFを開く
start main.pdf
```

### 方法2: 補足資料も含めて

```powershell
cd paper

# メイン論文
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# 補足資料
pdflatex supplementary.tex
bibtex supplementary
pdflatex supplementary.tex
pdflatex supplementary.tex

# 両方を開く
start main.pdf
start supplementary.pdf
```

### 方法3: PowerShellスクリプト

```powershell
# 自動コンパイルスクリプトを作成
cd paper

# 以下をcompile.ps1として保存して実行
.\compile.ps1
```

## 🔧 トラブルシューティング

### エラー: "pdflatex: command not found"

**原因**: LaTeXがPATHに追加されていない

**解決策**:
```powershell
# MiKTeXのパスを確認
where.exe pdflatex

# パスが見つからない場合、手動で追加
$env:Path += ";C:\Program Files\MiKTeX\miktex\bin\x64"

# または、MiKTeX Consoleから"Refresh FNDB"を実行
```

### エラー: "File `neurips_2024.sty' not found"

**原因**: 必要なLaTeXパッケージがインストールされていない

**解決策**:
```powershell
# MiKTeX Package Managerを開く
# または、自動インストールを有効化（MiKTeX Consoleで設定）

# コマンドラインから
mpm --install=neurips_2024
```

### エラー: "! LaTeX Error: File `figure1_stability.pdf' not found"

**原因**: 図がpaper/figuresディレクトリにない

**解決策**:
```powershell
# 図を再コピー
New-Item -ItemType Directory -Path "paper/figures" -Force
Copy-Item "results/paper_experiments/figure*.pdf" "paper/figures/" -Force
```

### 警告: "Citation 'xxx' undefined"

**原因**: bibtexを実行していない

**解決策**:
```powershell
# bibtexを実行してから再度pdflatexを実行
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## 📋 コンパイル後のチェックリスト

### 1. PDFが生成されたか確認
```powershell
dir paper/main.pdf
dir paper/supplementary.pdf
```

### 2. 図が表示されているか確認
- Figure 1: 長文脈安定性グラフ
- Figure 2: 量子化ロバスト性グラフ
- Figure 3: 動的効率性グラフ

### 3. テーブルが表示されているか確認
- Table 1: 長文脈安定性比較
- Table 2: 量子化ロバスト性比較
- Table 3: 効率性比較
- Table 4: アブレーション研究

### 4. 参考文献が正しくリンクされているか確認
- 引用番号が表示されている
- 参考文献リストが最後にある
- ハイパーリンクが機能している

### 5. ページ数を確認
- メイン論文: 8ページ以内
- 補足資料: 制限なし

## 🎨 PDFの品質確認

### 図の解像度
```powershell
# 図のサイズを確認
dir paper/figures/*.pdf | ForEach-Object { 
    Write-Host "$($_.Name): $($_.Length / 1KB) KB" 
}
```

期待値:
- figure1_stability.pdf: ~90 KB
- figure2_quantization.pdf: ~40 KB
- figure3_efficiency.pdf: ~60 KB

### フォントの埋め込み確認
```powershell
# PDFのプロパティを確認（Adobe Readerなどで）
# すべてのフォントが埋め込まれているか確認
```

## 📤 次のステップ

### 1. 論文のレビュー
- [ ] 全体を通読
- [ ] 図表の確認
- [ ] 参考文献の確認
- [ ] 数式の確認

### 2. 校正
- [ ] スペルチェック
- [ ] 文法チェック
- [ ] 一貫性チェック

### 3. フィードバック
- [ ] 同僚レビュー
- [ ] 指導教員レビュー
- [ ] 修正反映

### 4. 投稿準備
- [ ] 著者情報の追加
- [ ] 学会フォーマットの確認
- [ ] arXivバージョンの作成

## 🔄 再コンパイル

変更を加えた後：

```powershell
cd paper

# クリーンビルド
Remove-Item *.aux, *.bbl, *.blg, *.log, *.out -ErrorAction SilentlyContinue
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## 📊 コンパイル統計

期待される処理時間:
- 初回コンパイル: 2-5分（パッケージダウンロード含む）
- 2回目以降: 30秒-1分
- bibtex処理: 5-10秒

生成されるファイル:
- main.pdf (~500-800 KB)
- supplementary.pdf (~300-500 KB)
- 補助ファイル (.aux, .bbl, .blg, .log, .out)

## 🎯 成功の確認

コンパイルが成功したら：

```powershell
Write-Host "✓ 論文コンパイル成功！" -ForegroundColor Green
Write-Host "  - main.pdf: $(if (Test-Path 'paper/main.pdf') {'✓'} else {'✗'})" -ForegroundColor $(if (Test-Path 'paper/main.pdf') {'Green'} else {'Red'})
Write-Host "  - supplementary.pdf: $(if (Test-Path 'paper/supplementary.pdf') {'✓'} else {'✗'})" -ForegroundColor $(if (Test-Path 'paper/supplementary.pdf') {'Green'} else {'Red'})
Write-Host "`n次のステップ: PDFを開いて内容を確認してください"
```

---

**準備完了**: LaTeXインストール完了後、すぐにコンパイル可能です！

**推定時間**: 初回5分、以降1分以内
