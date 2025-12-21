.PHONY: help setup install clean recipe chat start-japanese train-japanese prepare-japanese-data dry-run-japanese resume-japanese resume list-checkpoints test benchmark export-model

# WSL Configuration
# All training commands run inside WSL Ubuntu
WSL := wsl -d ubuntu
WSL_BASH := $(WSL) bash -c

# Virtual environment paths (inside WSL)
VENV := venv_ubuntu
export PYTHONPATH := .

# For local (non-WSL) commands
SHELL := /bin/bash
PYTHON := $(shell if [ -f $(VENV)/bin/python ]; then echo $(VENV)/bin/python; else echo python3; fi)
PIP := $(shell if [ -f $(VENV)/bin/pip ]; then echo $(VENV)/bin/pip; else echo pip; fi)

help:
	@echo "=============================================="
	@echo "🧠 MUSE - 10B Japanese LLM (Phase 8)"
	@echo "=============================================="
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make start-japanese      🎯 Full: Download data + Train 10B Japanese"
	@echo "  make train-japanese      🚀 Fast: Skip data download (Train only)"
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
	@echo ""
	@echo "📝 WSL Ubuntu Setup:"
	@echo "  wsl -d ubuntu"
	@echo "  source venv_ubuntu/bin/activate"

# ==========================================
# 🔧 Setup
# ==========================================

setup:
	@echo "=========================================="
	@echo "🔧 MUSE 10B - Complete Setup"
	@echo "=========================================="
	@echo ""
	@echo "Step 1/5: Creating virtual environment..."
	test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	@echo ""
	@echo "Step 2/5: Installing Python dependencies..."
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	$(PIP) install datasets huggingface_hub triton ninja matplotlib
	@echo ""
	@echo "Step 3/5: Building CUDA C++ extensions..."
	@if command -v nvcc > /dev/null 2>&1 || [ -f /usr/local/cuda/bin/nvcc ]; then \
		echo "  CUDA found, building holographic kernel..."; \
		export CUDA_HOME=$${CUDA_HOME:-/usr/local/cuda}; \
		cd src/cuda && $(PYTHON) setup.py build_ext --inplace 2>&1 | tail -5 || echo "  (CUDA build skipped - will use PyTorch fallback)"; \
	else \
		echo "  CUDA not found, skipping kernel build (will use PyTorch fallback)"; \
	fi
	@echo ""
	@echo "Step 4/5: Verifying installation..."
	$(PYTHON) -c "import torch; print(f'  PyTorch: {torch.__version__}')"
	$(PYTHON) -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}')"
	$(PYTHON) -c "import datasets; print('  datasets: OK')" 2>/dev/null || echo "  datasets: install manually if needed"
	@echo ""
	@echo "Step 5/5: Testing revolutionary algorithms..."
	$(PYTHON) -c "from src.training.revolutionary_trainer import RevolutionaryTrainer; print('  RevolutionaryTrainer: OK')" 2>/dev/null || echo "  RevolutionaryTrainer: will initialize on first run"
	@echo ""
	@echo "=========================================="
	@echo "✅ Setup Complete!"
	@echo ""
	@echo "Next: Run 'make start-japanese' to begin training"
	@echo "=========================================="

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
	@echo "🇯🇵 Training Japanese 10B Model (Stable Phase 8)..."
	@bash -c '\
		cleanup() { pkill -f "checkpoint-saver" 2>/dev/null; echo "🦀 Checkpoint saver stopped"; }; \
		trap cleanup EXIT; \
		echo "🦀 Starting Checkpoint Saver Daemon..."; \
		(source ~/.cargo/env 2>/dev/null && cd checkpoint-saver && cargo run --release -q -- --config config.toml &) || echo "⚠ Checkpoint saver not available"; \
		$(PYTHON) scripts/train_phase8_stable.py --config configs/phase8_10b_japanese.yaml; \
	'

start-japanese:
	@echo "=========================================="
	@echo "🇯🇵 Japanese 10B LLM - Full Pipeline"
	@echo "=========================================="
	@echo "Step 1: Installing dependencies..."
	$(PIP) install -q datasets huggingface_hub || true
	@echo "Step 2: Verifying environment..."
	$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
	@echo "Step 3: Running quick benchmark..."
	$(PYTHON) scripts/benchmark_simple.py 2>&1 | head -30 || true
	@echo "Step 4: Downloading data..."
	$(MAKE) prepare-japanese-data
	@bash -c '\
		cleanup() { pkill -f "checkpoint-saver" 2>/dev/null; echo "🦀 Checkpoint saver stopped"; }; \
		trap cleanup EXIT; \
		echo "Step 5: Starting Checkpoint Saver Daemon..."; \
		(source ~/.cargo/env 2>/dev/null && cd checkpoint-saver && cargo run --release -q -- --config config.toml &) || echo "⚠ Checkpoint saver not available"; \
		echo "Step 6: Starting training with revolutionary algorithms..."; \
		$(PYTHON) scripts/train_phase8_stable.py --config configs/phase8_10b_japanese.yaml ; \
	'

dry-run-japanese:
	@echo "==========================================="
	@echo "🧪 Dry Run: Japanese 10B Model (Stable Phase 8)"
	@echo "==========================================="
	@echo ""
	@echo "📋 Expected Results (Stability Check):"
	@echo "   ✅ Vocab Size = 32768 (Cubic)"
	@echo "   ✅ Resonance Warmup: Active"
	@echo "   ✅ Loss decreasing from initial"
	@echo ""
	@echo "-------------------------------------------"
	@bash -c '\
		$(PYTHON) scripts/train_phase8_stable.py --config configs/phase8_10b_japanese.yaml --dry-run; \
	'
	@echo ""
	@echo "==========================================="
	@echo "🔍 Check the output above for:"
	@echo "   - No 'NaN/Inf' warnings"
	@echo "   - Loss decreasing each step"
	@echo "==========================================="

# ==========================================
# 💾 Checkpoint Management
# ==========================================

resume-japanese:
	@echo "🔄 Resuming Japanese Training..."
	@bash -c '\
		cleanup() { pkill -f "checkpoint-saver" 2>/dev/null; echo "🦀 Checkpoint saver stopped"; }; \
		trap cleanup EXIT; \
		echo "🦀 Starting Checkpoint Saver Daemon..."; \
		(source ~/.cargo/env 2>/dev/null && cd checkpoint-saver && cargo run --release -q -- --config config.toml &) || echo "⚠ Checkpoint saver not available"; \
		LATEST=$$(ls -t checkpoints/phase8_10b_japanese/step_*.pt 2>/dev/null | head -1); \
		if [ -n "$$LATEST" ]; then \
			echo "Found: $$LATEST"; \
			$(PYTHON) scripts/train_phase8_stable.py --config configs/phase8_10b_japanese.yaml --resume-from "$$LATEST" ; \
		else \
			echo "❌ No checkpoint. Run make start-japanese first."; \
		fi; \
	'

resume:
ifdef CHECKPOINT
	$(PYTHON) scripts/train_phase8_stable.py --config configs/phase8_10b_japanese.yaml --resume-from $(CHECKPOINT) 
else
	@echo "Usage: make resume CHECKPOINT=checkpoints/phase8_10b_japanese/step_500.pt"
endif

list-checkpoints:
	@echo "💾 Checkpoints:"
	@echo "Japanese:" && ls -lh checkpoints/phase8_10b_japanese/*.pt 2>/dev/null || echo "  (none)"
	@echo "English:"  && ls -lh checkpoints/phase8_10b_rtx3080/*.pt 2>/dev/null || echo "  (none)"
	@echo "300M Scaling:" && ls -lh checkpoints/phase8_300m_scaling/*.pt 2>/dev/null || echo "  (none)"

# ==========================================
# 📊 300M Scaling Law Experiment
# ==========================================

train-300m:
	@echo "==========================================="
	@echo "🔬 Training 300M Model for Scaling Laws"
	@echo "==========================================="
	@echo ""
	@echo "📋 Configuration:"
	@echo "   Model: 300M (d=1024, layers=24)"
	@echo "   Batch: 4 x 8 = 32 effective"
	@echo "   Save: checkpoints/phase8_300m_scaling/"
	@echo ""
	$(PYTHON) scripts/train_phase8_stable.py --config configs/phase8_300m_scaling.yaml

resume-300m:
	@echo "==========================================="
	@echo "🔬 Resuming 300M Scaling Experiment"
	@echo "==========================================="
	@LATEST=$$(ls -t checkpoints/phase8_300m_scaling/step_*.pt 2>/dev/null | head -1); \
	if [ -n "$$LATEST" ]; then \
		echo "Found: $$LATEST"; \
		$(PYTHON) scripts/train_phase8_stable.py --config configs/phase8_300m_scaling.yaml --resume-from "$$LATEST"; \
	else \
		echo "❌ No checkpoint found. Run make train-300m first."; \
	fi

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
	@echo "⚡ Running Phase 8 Kernel Benchmark..."
	$(PYTHON) scripts/benchmark_simple.py

profile-checkpoint:
	@echo "🔬 Profiling Checkpoint Slowdown..."
	@echo "This will identify exact source of speed degradation after checkpoint saves."
	$(PYTHON) scripts/profile_checkpoint_slowdown.py

recipe:
	$(PYTHON) scripts/configure_recipe.py

# ==========================================
# 🦀 Rust Checkpoint Saver Daemon
# ==========================================

build-saver:
	@echo "🦀 Building Checkpoint Saver Daemon..."
	cd checkpoint-saver && cargo build --release
	@echo "✅ Built: checkpoint-saver/target/release/checkpoint-saver"

run-saver:
	@echo "🦀 Starting Checkpoint Saver Daemon..."
	@echo "   Press Ctrl+C to stop"
	cd checkpoint-saver && cargo run --release -- --config config.toml

run-saver-bg:
	@echo "🦀 Starting Checkpoint Saver Daemon (background)..."
	cd checkpoint-saver && cargo run --release -- --config config.toml &
	@echo "✅ Daemon started in background"

stop-saver:
	@pkill -f "checkpoint-saver" || echo "No daemon running"
