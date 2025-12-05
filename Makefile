.PHONY: help setup install clean recipe chat start-japanese train-japanese prepare-japanese-data dry-run-japanese resume-japanese resume list-checkpoints test benchmark export-model

# Default shell
SHELL := /bin/bash
VENV := venv_ubuntu
export PYTHONPATH := .
PYTHON := $(shell if [ -f $(VENV)/bin/python ]; then echo $(VENV)/bin/python; else echo python3; fi)
PIP := $(shell if [ -f $(VENV)/bin/pip ]; then echo $(VENV)/bin/pip; else echo pip; fi)

help:
	@echo "=============================================="
	@echo "🧠 MUSE - 10B Japanese LLM (Phase 8)"
	@echo "=============================================="
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make start-japanese      🎯 Full: Download data + Train 10B Japanese"
	@echo "  make dry-run-japanese    Test model config (no training)"
	@echo ""
	@echo "💾 Resume & Checkpoints:"
	@echo "  make resume-japanese     Resume from latest checkpoint"
	@echo "  make resume CHECKPOINT=path  Resume from specific file"
	@echo "  make list-checkpoints    Show saved checkpoints"
	@echo ""
	@echo "💬 Chat & Export:"
	@echo "  make chat                Chat with trained model"
	@echo "  make export-model        Export model for deployment"
	@echo ""
	@echo "🔧 Setup & Utils:"
	@echo "  make setup               Install dependencies"
	@echo "  make recipe              Configure training wizard"
	@echo "  make test                Run tests"
	@echo "  make benchmark           Run speed benchmark"
	@echo "  make clean               Clean caches"

# ==========================================
# 🔧 Setup
# ==========================================

setup:
	@if [ -f scripts/easy_setup.sh ]; then \
		chmod +x scripts/easy_setup.sh && ./scripts/easy_setup.sh; \
	else \
		$(MAKE) install; \
	fi

install:
	test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

clean:
	rm -rf build dist *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Cleaned (venv preserved)"

# ==========================================
# 🇯🇵 Japanese LLM Training
# ==========================================

prepare-japanese-data:
	@echo "🇯🇵 Downloading Japanese Datasets..."
	$(PYTHON) scripts/prepare_japanese_data.py --max-pretrain 100000 --max-instruct 20000

train-japanese:
	@echo "🇯🇵 Training Japanese 10B Model..."
	$(PYTHON) scripts/train_phase8.py --config configs/phase8_10b_japanese.yaml --compile

start-japanese:
	@echo "=========================================="
	@echo "🇯🇵 Japanese 10B LLM - Full Pipeline"
	@echo "=========================================="
	$(MAKE) prepare-japanese-data
	$(MAKE) train-japanese

dry-run-japanese:
	@echo "🧪 Dry Run: Japanese 10B Model..."
	$(PYTHON) scripts/train_phase8.py --config configs/phase8_10b_japanese.yaml --dry-run --compile

# ==========================================
# 💾 Checkpoint Management
# ==========================================

resume-japanese:
	@echo "🔄 Resuming Japanese Training..."
	@LATEST=$$(ls -t checkpoints/phase8_10b_japanese/step_*.pt 2>/dev/null | head -1); \
	if [ -n "$$LATEST" ]; then \
		echo "Found: $$LATEST"; \
		$(PYTHON) scripts/train_phase8.py --config configs/phase8_10b_japanese.yaml --resume-from "$$LATEST" --compile; \
	else \
		echo "❌ No checkpoint. Run 'make start-japanese' first."; \
	fi

resume:
ifdef CHECKPOINT
	$(PYTHON) scripts/train_phase8.py --config configs/phase8_10b_japanese.yaml --resume-from $(CHECKPOINT) --compile
else
	@echo "Usage: make resume CHECKPOINT=checkpoints/phase8_10b_japanese/step_500.pt"
endif

list-checkpoints:
	@echo "💾 Checkpoints:"
	@echo "Japanese:" && ls -lh checkpoints/phase8_10b_japanese/*.pt 2>/dev/null || echo "  (none)"
	@echo "English:"  && ls -lh checkpoints/phase8_10b_rtx3080/*.pt 2>/dev/null || echo "  (none)"

# ==========================================
# 💬 Chat & Export
# ==========================================

chat:
ifdef CHECKPOINT
	$(PYTHON) scripts/chat_inference.py --checkpoint $(CHECKPOINT)
else
	$(PYTHON) scripts/chat_inference.py
endif

export-model:
	@echo "📦 Exporting model for deployment..."
	@LATEST=$$(ls -t checkpoints/phase8_10b_japanese/*.pt 2>/dev/null | head -1); \
	if [ -n "$$LATEST" ]; then \
		mkdir -p exports && \
		cp "$$LATEST" exports/muse_japanese_10b.pt && \
		echo "✓ Exported to exports/muse_japanese_10b.pt"; \
	else \
		echo "❌ No model found. Train first."; \
	fi

# ==========================================
# 🧪 Testing & Benchmarks
# ==========================================

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

benchmark:
	@echo "⚡ Running speed benchmark..."
	$(PYTHON) -c "from scripts.train_phase8 import *; print('Benchmark not implemented yet')"

recipe:
	$(PYTHON) scripts/configure_recipe.py
