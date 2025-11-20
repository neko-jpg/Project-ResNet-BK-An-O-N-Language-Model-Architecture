# Phase 2 CI/CD Quick Reference

**最終更新**: 2025年11月20日

---

## 🚀 クイックスタート

### ワークフローファイル
- **場所**: `.github/workflows/phase2_tests.yml`
- **トリガー**: Push, PR, Schedule (毎日02:00 UTC), Manual

### 手動実行
```bash
# GitHub UIから
1. Actions タブ → "Phase 2 Tests" → "Run workflow"
2. テストスイート選択: all/unit/integration/benchmarks/long_context
```

---

## 📋 テストジョブ一覧

### Priority 0: 基盤の修復
| ジョブ名 | テスト対象 | タイムアウト |
|---------|-----------|------------|
| `test-bk-triton` | BK-Core Tritonカーネル | 5分 |
| `test-complex-gradient` | 複素勾配安全性 | 5分 |

### Priority 1: コアアルゴリズム
| ジョブ名 | テスト対象 | タイムアウト |
|---------|-----------|------------|
| `test-non-hermitian` | Non-Hermitian Forgetting | 5分 |
| `test-dissipative-hebbian` | Dissipative Hebbian | 5分 |
| `test-memory-selection` | SNRベース記憶選択 | 5分 |
| `test-memory-resonance` | Memory Resonance | 5分 |
| `test-zeta-init` | Zeta初期化 | 5分 |

### Priority 2: 統合モデル
| ジョブ名 | テスト対象 | タイムアウト |
|---------|-----------|------------|
| `test-phase2-block` | Phase2Block | 5分 |
| `test-phase2-integrated` | Phase2統合モデル | 10分 |
| `test-phase2-factory` | Phase2ファクトリー | 5分 |
| `test-phase2-integration` | Phase2完全統合 | 15分 |

### Priority 3: ベンチマークと検証
| ジョブ名 | テスト対象 | タイムアウト | 条件 |
|---------|-----------|------------|------|
| `test-phase2-benchmarks` | ベンチマーク | 20分 | Push/Manual |
| `test-long-context` | 長期依存関係 | 30分 | Push/Manual |

### Priority 4: 例とドキュメント
| ジョブ名 | テスト対象 | タイムアウト |
|---------|-----------|------------|
| `test-examples` | Phase2使用例 | 各5分 |
| `verify-docstrings` | Docstring検証 | 5分 |

---

## 🎯 カバレッジフラグ

各テストジョブは個別のカバレッジフラグを使用:

```yaml
flags:
  - phase2-bk-triton
  - phase2-complex-grad
  - phase2-non-hermitian
  - phase2-dissipative-hebbian
  - phase2-memory-selection
  - phase2-memory-resonance
  - phase2-zeta-init
  - phase2-block
  - phase2-integrated
  - phase2-factory
  - phase2-integration
  - phase2-benchmarks
```

---

## 📦 アーティファクト

### 自動保存されるファイル
1. **ベンチマーク結果**
   - `results/benchmarks/*.json`
   - `results/benchmarks/PHASE2_BENCHMARK_REPORT.md`

2. **統合テストレポート**
   - `results/benchmarks/PHASE2_INTEGRATION_TEST_REPORT.md`

3. **長期コンテキストテスト結果**
   - `results/benchmarks/long_context_*.json`
   - `results/benchmarks/LONG_CONTEXT_TEST_IMPLEMENTATION_REPORT.md`

4. **Docstring検証レポート**
   - `results/benchmarks/TASK17_DOCSTRING_COMPLETION_REPORT.md`

5. **テストサマリー**
   - `phase2_summary.md`

### ダウンロード方法
```bash
# GitHub UIから
1. Actions → ワークフロー実行 → Artifacts
2. 各アーティファクトをダウンロード
```

---

## 🔧 トラブルシューティング

### テスト失敗時
1. **ログ確認**
   ```
   Actions → ワークフロー実行 → 失敗したジョブ → ログ表示
   ```

2. **ローカル再現**
   ```bash
   # 該当テストをローカルで実行
   pytest tests/test_phase2_xxx.py -v
   ```

3. **Issue自動作成**
   - テスト失敗時、自動的にGitHub Issueが作成されます
   - ラベル: `bug`, `phase2`, `ci-failure`

### タイムアウト発生時
```yaml
# タイムアウト時間の調整（必要に応じて）
- name: Run tests
  run: pytest tests/test_xxx.py -v
  timeout-minutes: 10  # デフォルトは5分
```

### キャッシュクリア
```bash
# GitHub UIから
Settings → Actions → Caches → 該当キャッシュを削除
```

---

## 📊 ワークフロー監視

### ステータスバッジ
```markdown
![Phase 2 Tests](https://github.com/YOUR_ORG/Project-MUSE/workflows/Phase%202%20Tests/badge.svg)
```

### 実行履歴
```bash
# GitHub CLIで確認
gh run list --workflow=phase2_tests.yml
```

### 最新実行の詳細
```bash
gh run view --workflow=phase2_tests.yml
```

---

## 🎨 カスタマイズ

### テストスイート選択
```yaml
# 手動実行時に選択可能
workflow_dispatch:
  inputs:
    test_suite:
      type: choice
      options:
        - all          # 全テスト
        - unit         # 単体テストのみ
        - integration  # 統合テストのみ
        - benchmarks   # ベンチマークのみ
        - long_context # 長期コンテキストのみ
```

### 実行条件の変更
```yaml
# 特定のパスの変更時のみ実行
on:
  push:
    paths:
      - 'src/models/phase2/**'
      - 'tests/test_phase2_*.py'
```

---

## 📈 パフォーマンス最適化

### キャッシング戦略
```yaml
- uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-phase2-${{ hashFiles('requirements.txt') }}
```

### 並列実行
```yaml
strategy:
  matrix:
    python-version: ['3.9', '3.10', '3.11']
  # 全バージョンで並列実行
```

### 条件付き実行
```yaml
# 重いテストはpush時のみ
if: github.event_name == 'push'
```

---

## 🔐 セキュリティ

### シークレット管理
```yaml
# 必要に応じてシークレットを追加
env:
  API_KEY: ${{ secrets.API_KEY }}
```

### 権限設定
```yaml
permissions:
  contents: read
  issues: write  # Issue作成用
  pull-requests: write  # PRコメント用
```

---

## 📝 ベストプラクティス

### 1. テスト追加時
```yaml
# 新しいテストジョブを追加
test-new-feature:
  name: New Feature Tests
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v3
    - name: Run tests
      run: pytest tests/test_new_feature.py -v
```

### 2. 依存関係の管理
```yaml
# 特定のジョブ完了後に実行
needs: [test-bk-triton, test-complex-gradient]
```

### 3. タイムアウト設定
```yaml
# 長時間実行されるテストには必ずタイムアウトを設定
timeout-minutes: 30
```

---

## 🚨 緊急時の対応

### ワークフロー無効化
```bash
# GitHub UIから
Actions → "Phase 2 Tests" → "..." → "Disable workflow"
```

### 特定ジョブのスキップ
```yaml
# コミットメッセージに追加
git commit -m "fix: update code [skip ci]"
```

---

## 📞 サポート

### ドキュメント
- **詳細レポート**: `results/benchmarks/TASK20_CI_CD_IMPLEMENTATION_REPORT.md`
- **ワークフローファイル**: `.github/workflows/phase2_tests.yml`

### 問題報告
1. GitHub Issueを作成
2. ラベル: `ci-cd`, `phase2`
3. ワークフロー実行URLを添付

---

**最終更新**: 2025年11月20日  
**バージョン**: 1.0.0  
**メンテナー**: Project MUSE Team
