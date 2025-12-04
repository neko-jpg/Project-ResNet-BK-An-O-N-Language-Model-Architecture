.PHONY: help setup install data data-lite clean doctor recipe start-10b-local compress-10b train-10b chat dashboard

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
	@echo "MUSE (ResNet-BK Phase 8) Development Commands"
	@echo "=============================================="
	@echo "Quick Start:"
	@echo "  make start-10b-local  - 🚀 Auto-setup & Train 10B Model (Recommended)"
	@echo ""
	@echo "Core Commands:"
	@echo "  make setup            - Install dependencies & Prepare environment"
	@echo "  make chat             - Start Chat Interface with trained model"
	@echo "  make dashboard        - Start Training Dashboard"
	@echo "  make clean            - Clean artifacts and caches"
	@echo ""
	@echo "Data & Configuration:"
	@echo "  make recipe           - Configure dataset mixing recipe"
	@echo "  make data-lite        - Download small test dataset (Cosmopedia)"
	@echo "  make data             - Download full datasets"
	@echo ""
	@echo "Manual Training Steps (Advanced):"
	@echo "  make compress-10b     - Initialize/Compress 10B Model"
	@echo "  make train-10b        - Train 10B Model (Requires compressed checkpoint)"
	@echo "  make train-10b-8gb    - Train with Extreme Optimization (8GB VRAM)"

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
	$(PYTHON) scripts/train_phase8.py --d-model 4096 --n-layers 48 --extreme-compression --dataset configs/dataset_mixing.yaml $(TRAIN_OVERRIDES)
