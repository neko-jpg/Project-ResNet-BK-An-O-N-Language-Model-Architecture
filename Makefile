.PHONY: help setup install data data-lite clean doctor recipe start-10b-local compress-10b train-10b chat dashboard start-japanese train-japanese prepare-japanese-data dry-run-japanese

# Default shell
SHELL := /bin/bash
VENV := venv_ubuntu
export PYTHONPATH := .
# Try to detect if we are in the venv or need to use the path
PYTHON := $(shell if [ -f $(VENV)/bin/python ]; then echo $(VENV)/bin/python; else echo python3; fi)
PIP := $(shell if [ -f $(VENV)/bin/pip ]; then echo $(VENV)/bin/pip; else echo pip; fi)

# Optional CLI overrides for training
TRAIN_OVERRIDES :=
ifdef N_SEQ
TRAIN_OVERRIDES += --n-seq $(N_SEQ)
endif
ifdef D_MODEL
TRAIN_OVERRIDES += --d-model $(D_MODEL)
endif
ifdef N_LAYERS
TRAIN_OVERRIDES += --n-layers $(N_LAYERS)
endif
ifdef BATCH_SIZE
TRAIN_OVERRIDES += --batch-size $(BATCH_SIZE)
endif
ifdef EPOCHS
TRAIN_OVERRIDES += --epochs $(EPOCHS)
endif

help:
	@echo "=============================================="
	@echo "🧠 MUSE (ResNet-BK Phase 8) - 10B LLM Training"
	@echo "=============================================="
	@echo ""
	@echo "🚀 Quick Start (English):"
	@echo "  make start-10b-local     Auto-setup & Train 10B English Model"
	@echo "  make dry-run-10b         Test run without training"
	@echo ""
	@echo "🇯🇵 Japanese LLM (Recommended):"
	@echo "  make start-japanese      🎯 Full Pipeline: Download data + Train Japanese 10B"
	@echo "  make prepare-japanese-data   Download Japanese datasets only"
	@echo "  make train-japanese      Train with existing Japanese data"
	@echo "  make dry-run-japanese    Test Japanese model config"
	@echo ""
	@echo "� Checkpoints & Resume:"
	@echo "  make resume-japanese     Resume training from latest checkpoint"
	@echo "  make resume CHECKPOINT=path/to/step_500.pt  Resume from specific checkpoint"
	@echo "  make list-checkpoints    Show available checkpoints"
	@echo "  (Checkpoints auto-saved every 500 steps)"
	@echo ""
	@echo "💬 Chat & Inference:"
	@echo "  make chat                Start interactive chat with trained model"
	@echo "  make chat CHECKPOINT=path  Chat with specific checkpoint"
	@echo ""
	@echo "🔧 Setup & Utilities:"
	@echo "  make setup               Install dependencies"
	@echo "  make recipe              Configure training (language, VRAM, etc.)"
	@echo "  make dashboard           Start Training Dashboard"
	@echo "  make clean               Clean artifacts and caches"
	@echo ""
	@echo "📊 Data Management:"
	@echo "  make data-lite           Download small English dataset"
	@echo "  make data                Download full English datasets"
	@echo ""
	@echo "⚙️  Advanced:"
	@echo "  make compress-10b        Initialize/Compress 10B Model"
	@echo "  make train-10b-8gb       Train with Extreme Optimization (8GB VRAM)"

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

data-lite:
	$(PYTHON) scripts/prepare_datasets.py --datasets cosmopedia --max_samples 1000

data:
	$(PYTHON) scripts/prepare_datasets.py

clean:
	rm -rf $(VENV)
	rm -rf build dist *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete

doctor:
	$(PYTHON) scripts/doctor.py

recipe:
	$(PYTHON) scripts/configure_recipe.py

chat:
	PYTHONPATH=. $(VENV)/bin/streamlit run app.py

dashboard:
	PYTHONPATH=. $(VENV)/bin/streamlit run app.py

# ============================================================================
# Phase 8: 10B Parameter Workflow
# ============================================================================

start-10b-local:
	@echo "=========================================="
	@echo "🚀 Starting 10B All-in-One Local Setup & Training"
	@echo "=========================================="
	$(PYTHON) scripts/easy_start_10b.py

compress-10b:
	@echo "=========================================="
	@echo "🗜️  Compressing 10B (100.1 Billion) Parameter Model"
	@echo "=========================================="
	$(PYTHON) scripts/compress_model.py --output_dir checkpoints/compressed_10b_start --d_model 5120 --n_layers 31

train-10b:
	@if [ ! -f checkpoints/compressed_10b_start/compressed_model.pt ]; then \
		echo "Error: Compressed model not found. Please run 'make compress-10b' first."; \
		exit 1; \
	fi
	@echo "=========================================="
	@echo "🚀 Starting Training on 10B Compressed Model (RTX 3080 Ready)"
	@echo "=========================================="
	$(PYTHON) scripts/train_phase8.py --config configs/phase8_10b.yaml --resume-from checkpoints/compressed_10b_start/compressed_model.pt --dataset configs/dataset_mixing.yaml $(TRAIN_OVERRIDES)

train-10b-8gb:
	@echo "=========================================="
	@echo "🚀 Starting Extreme Optimization Training (RTX 3080 8GB)"
	@echo "=========================================="
	$(PYTHON) scripts/train_phase8.py --config configs/phase8_10b_rtx3080.yaml --extreme-compression --compile --dataset configs/dataset_mixing.yaml $(TRAIN_OVERRIDES)

# Dry run for testing (no dataset download)
dry-run-10b:
	@echo "=========================================="
	@echo "🧪 Dry Run: Testing 10B Configuration"
	@echo "=========================================="
	$(PYTHON) scripts/train_phase8.py --config configs/phase8_10b_rtx3080.yaml --dry-run --compile

# WSL Quick Start - runs everything in one command
wsl-start-10b:
	@echo "=========================================="
	@echo "🐧 WSL Ubuntu: Starting 10B Training"
	@echo "=========================================="
	wsl -d ubuntu -e bash -c "cd /mnt/c/dev/Project-ResNet-BK-An-O-N-Language-Model-Architecture && source venv_ubuntu/bin/activate && make start-10b-local"

# ==========================================
# 🇯🇵 Japanese LLM Training
# ==========================================

# Step 1: Download Japanese datasets
prepare-japanese-data:
	@echo "=========================================="
	@echo "🇯🇵 Downloading Japanese Datasets..."
	@echo "=========================================="
	$(PYTHON) scripts/prepare_japanese_data.py --max-pretrain 100000 --max-instruct 20000

# Step 2: Train Japanese model
train-japanese:
	@echo "=========================================="
	@echo "🇯🇵 Training Japanese 10B Model"
	@echo "=========================================="
	$(PYTHON) scripts/train_phase8.py --config configs/phase8_10b_japanese.yaml --dataset configs/dataset_japanese.yaml --compile

# All-in-one Japanese training
start-japanese:
	@echo "=========================================="
	@echo "🇯🇵 Japanese LLM Full Pipeline"
	@echo "=========================================="
	make prepare-japanese-data
	make train-japanese

# Dry run for Japanese model
dry-run-japanese:
	@echo "=========================================="
	@echo "🧪 Dry Run: Japanese 10B Model"
	@echo "=========================================="
	$(PYTHON) scripts/train_phase8.py --config configs/phase8_10b_japanese.yaml --dry-run --compile

# ==========================================
# 💾 Checkpoint Management
# ==========================================

# Resume from latest Japanese checkpoint
resume-japanese:
	@echo "=========================================="
	@echo "🔄 Resuming Japanese Training"
	@echo "=========================================="
	@LATEST=$$(ls -t checkpoints/phase8_10b_japanese/step_*.pt 2>/dev/null | head -1); \
	if [ -n "$$LATEST" ]; then \
		echo "Found checkpoint: $$LATEST"; \
		$(PYTHON) scripts/train_phase8.py --config configs/phase8_10b_japanese.yaml --resume-from "$$LATEST" --compile; \
	else \
		echo "❌ No checkpoint found. Run 'make start-japanese' first."; \
	fi

# Resume from specific checkpoint
resume:
	@echo "=========================================="
	@echo "🔄 Resuming from: $(CHECKPOINT)"
	@echo "=========================================="
ifdef CHECKPOINT
	$(PYTHON) scripts/train_phase8.py --config configs/phase8_10b_japanese.yaml --resume-from $(CHECKPOINT) --compile
else
	@echo "❌ Usage: make resume CHECKPOINT=path/to/step_500.pt"
endif

# List available checkpoints
list-checkpoints:
	@echo "=========================================="
	@echo "💾 Available Checkpoints"
	@echo "=========================================="
	@echo ""
	@echo "Japanese (phase8_10b_japanese):"
	@ls -lh checkpoints/phase8_10b_japanese/*.pt 2>/dev/null || echo "  (none)"
	@echo ""
	@echo "English (phase8_10b_rtx3080):"
	@ls -lh checkpoints/phase8_10b_rtx3080/*.pt 2>/dev/null || echo "  (none)"
	@echo ""
	@echo "General (phase8):"
	@ls -lh checkpoints/phase8/*.pt 2>/dev/null || echo "  (none)"

# Chat with model
chat:
	@echo "=========================================="
	@echo "💬 Starting Chat Interface"
	@echo "=========================================="
ifdef CHECKPOINT
	$(PYTHON) scripts/chat_inference.py --checkpoint $(CHECKPOINT)
else
	$(PYTHON) scripts/chat_inference.py
endif
