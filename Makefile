# ResNet-BK Phase 8 訓練用 Makefile
# WSL ubuntu内で使用: make help で全コマンドを確認
#
# 使用方法:
#   make train      - 新規訓練を開始
#   make resume     - 最新チェックポイントから再開
#   make verify     - 訓練前検証を実行
#   make test       - ユニットテストを実行
#   make help       - ヘルプを表示

.PHONY: train resume verify test test-data status clean help

# 設定
SHELL := /bin/bash
VENV := source venv_ubuntu/bin/activate &&
CONFIG := configs/phase8_300m_japanese_chat.yaml
TRAIN_SCRIPT := scripts/train_phase8_stable.py
CHECKPOINT_DIR := checkpoints/phase8_300m_japanese_chat

# デフォルト: ヘルプ表示
.DEFAULT_GOAL := help

#==============================================================================
# ヘルプ
#==============================================================================
help:
	@echo ""
	@echo "==================================="
	@echo " ResNet-BK Phase 8 訓練コマンド"
	@echo "==================================="
	@echo ""
	@echo " 訓練:"
	@echo "   make train     - 新規訓練を開始（ステップ0から）"
	@echo "   make resume    - 最新チェックポイントから再開"
	@echo ""
	@echo " データセット:"
	@echo "   make regenerate-data - rinnaトークナイザーでデータ再生成"
	@echo ""
	@echo " 検証:"
	@echo "   make verify    - 訓練前の包括的検証を実行"
	@echo "   make test      - ユニットテストを実行"
	@echo "   make test-data - データサンプリングテスト"
	@echo ""
	@echo " ユーティリティ:"
	@echo "   make status    - 訓練ステータスを確認"
	@echo "   make clean     - キャッシュファイルを削除"
	@echo "   make help      - このヘルプを表示"
	@echo ""

#==============================================================================
# 訓練
#==============================================================================
train:
	@echo "🚀 新規訓練を開始..."
	$(VENV) python $(TRAIN_SCRIPT) --config $(CONFIG) --dataset configs/dataset_japanese_chat_optimized.yaml

resume:
	@echo "🔄 チェックポイントから再開..."
	$(VENV) python $(TRAIN_SCRIPT) --config $(CONFIG) --dataset configs/dataset_japanese_chat_optimized.yaml --resume

#==============================================================================
# データセット再生成
#==============================================================================
regenerate-data:
	@echo "📦 rinnaトークナイザーでデータセット再生成..."
	chmod +x scripts/regenerate_datasets.sh
	bash scripts/regenerate_datasets.sh

#==============================================================================
# 検証
#==============================================================================
verify:
	@echo "🔍 訓練前検証を実行..."
	$(VENV) python scripts/verify_pretrain.py

test:
	@echo "🧪 ユニットテストを実行..."
	$(VENV) python -m pytest tests/test_data_validation.py -v

test-data:
	@echo "📊 データサンプリングテスト..."
	$(VENV) python scripts/test_sampling.py

#==============================================================================
# ユーティリティ
#==============================================================================
status:
	@echo "📈 訓練ステータス..."
	@if [ -f "$(CHECKPOINT_DIR)/latest.pt" ]; then \
		echo "最新チェックポイント: $(CHECKPOINT_DIR)/latest.pt"; \
		ls -lh $(CHECKPOINT_DIR)/latest.pt; \
		echo ""; \
		echo "全チェックポイント:"; \
		ls -lh $(CHECKPOINT_DIR)/*.pt 2>/dev/null || echo "  (なし)"; \
	else \
		echo "チェックポイントが見つかりません"; \
	fi

clean:
	@echo "🧹 キャッシュを削除..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "完了"
