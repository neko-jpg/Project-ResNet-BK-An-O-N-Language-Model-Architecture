.PHONY: help setup install data data-lite data-ja data-ja-lite test demo clean up down doctor import recipe train-user phase4

# Default shell
SHELL := /bin/bash
VENV := venv_ubuntu
export PYTHONPATH := .
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest

# Optional CLI overrides for training (set via `make train-user N_SEQ=512 BATCH_SIZE=8 ...`)
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
	@bash -c 'source .muse_config 2>/dev/null || true; \
	if [ "$$MUSE_LANG" = "2" ]; then \
		echo "MUSE (ResNet-BK) 開発コマンド"; \
		echo "======================================"; \
		echo "make setup      - 完全セットアップ (仮想環境, 依存関係, Liteデータ)"; \
		echo "make install    - 依存関係のみインストール"; \
		echo "make doctor     - システム診断とトラブルシューティング"; \
		echo "make import     - 独自データのインポート (data/import/ から)"; \
		echo "make recipe     - 学習データの配合設定 (Phase 3/7 モデル選択)"; \
		echo "make phase4     - Phase 4最強設定(BitNet/Symplectic)を現在の設定に適用"; \
		echo "make train-user - 設定したレシピで学習開始"; \
		echo "make train-resume - 学習の再開 (Usage: make train-resume CHECKPOINT=...)"; \
		echo "make reborn       - Reborn Ritual (強化学習的転生)"; \
		echo "make merge        - モデルのマージ"; \
		echo "make data-lite  - テスト用データセットのダウンロード"; \
		echo "make data       - 全データセットのダウンロード"; \
		echo "make test       - テストの実行"; \
		echo "make demo       - MUSEデモの実行"; \
		echo "make clean      - 仮想環境とキャッシュの削除"; \
		echo "make scale-up   - ハードウェアに合わせた最適設定の自動生成"; \
		echo "make chat       - MUSE Creative Studio (Chat & Merge)"; \
		echo "make dashboard  - 学習状況の可視化 (Streamlit)"; \
		echo "make clean-safe - ゴミファイルと古いチェックポイントの掃除"; \
		echo "make deploy     - Hugging Faceへデプロイ"; \
		echo "make pack       - 配布用Zipの作成"; \
		echo "make restore    - 現在の状態をバックアップ"; \
		echo "make up         - Docker環境の起動"; \
		echo "make down       - Docker環境の停止"; \
		echo ""; \
		echo "Phase 7 (ハイブリッド双曲アテンション):"; \
		echo "make train-phase7       - Phase 7モデルの学習 (CUDA+Triton必須)"; \
		echo "make train-phase7-small - テスト用小規模設定で学習"; \
		echo "make test-phase7        - Phase 7統合テスト実行"; \
		echo "make triton-attn        - Tritonカーネル動作確認"; \
	else \
		echo "MUSE (ResNet-BK) Development Commands"; \
		echo "======================================"; \
		echo "make setup      - Full setup (venv, deps, lite data)"; \
		echo "make install    - Install dependencies only"; \
		echo "make doctor     - Run system diagnostics"; \
		echo "make import     - Import user data from data/import/"; \
		echo "make recipe     - Configure dataset mixing recipe (select Phase 3/7 model)"; \
		echo "make phase4     - Merge Phase 4 Config into current recipe"; \
		echo "make train-user - Start training with user recipe"; \
		echo "make train-resume - Resume training (Usage: make train-resume CHECKPOINT=...)"; \
		echo "make reborn       - Reborn Ritual (Usage: make reborn CHECKPOINT=...)"; \
		echo "make merge        - Merge Models"; \
		echo "make data-lite  - Download small test dataset"; \
		echo "make data       - Download ALL datasets"; \
		echo "make test       - Run tests"; \
		echo "make demo       - Run MUSE capabilities demo"; \
		echo "make clean      - Remove venv and artifacts"; \
		echo "make scale-up   - Auto-configure for hardware"; \
		echo "make chat       - MUSE Creative Studio (Chat & Merge)"; \
		echo "make dashboard  - Visualize Training (Streamlit)"; \
		echo "make clean-safe - Clean garbage and old checkpoints"; \
		echo "make deploy     - Deploy to Hugging Face"; \
		echo "make pack       - Create distribution Zip"; \
		echo "make restore    - Backup current state"; \
		echo "make up         - Start Docker environment"; \
		echo "make down       - Stop Docker environment"; \
		echo ""; \
		echo "Phase 7 (Hybrid Hyperbolic Attention):"; \
		echo "make train-phase7       - Train Phase 7 model (CUDA+Triton required)"; \
		echo "make train-phase7-small - Train with small config for testing"; \
		echo "make test-phase7        - Run Phase 7 integration tests"; \
		echo "make triton-attn        - Verify Triton kernel"; \
	fi'

setup:
	@chmod +x scripts/easy_setup.sh
	@./scripts/easy_setup.sh

install:
	test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

data-lite:
	$(PYTHON) scripts/prepare_datasets.py --datasets cosmopedia --max_samples 1000

data-ja-lite:
	$(PYTHON) scripts/prepare_datasets.py --datasets mc4_ja --max_samples 1000

data-ja:
	$(PYTHON) scripts/prepare_datasets.py --datasets mc4_ja

data:
	$(PYTHON) scripts/prepare_datasets.py

test:
	$(PYTEST) tests/

ci:
	$(PYTHON) scripts/run_ci.py

demo:
	$(PYTHON) scripts/demo_muse_full.py

clean:
	rm -rf $(VENV)
	rm -rf build dist *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete

up:
	docker-compose up -d

down:
	docker-compose down

doctor:
	$(PYTHON) scripts/doctor.py

import:
	$(PYTHON) scripts/import_user_data.py

recipe:
	$(PYTHON) scripts/configure_recipe.py

phase4:
	$(PYTHON) scripts/apply_phase4_config.py

train-user:
	@if [ ! -f configs/dataset_mixing.yaml ]; then \
		echo "Error: Recipe not found. Please run 'make recipe' first."; \
		exit 1; \
	fi
	@if [ -f configs/auto_optimized.yaml ]; then \
		echo "Using auto-optimized config..."; \
		cmd="$(PYTHON) scripts/train.py --dataset configs/dataset_mixing.yaml --config configs/auto_optimized.yaml $(TRAIN_OVERRIDES)"; \
		echo "$$cmd"; \
		$$cmd; \
	elif [ -f configs/user_train_config.yaml ]; then \
		echo "Using user_train_config.yaml (Phase 4 / Manual)..."; \
		cmd="$(PYTHON) scripts/train.py --dataset configs/dataset_mixing.yaml --config configs/user_train_config.yaml $(TRAIN_OVERRIDES)"; \
		echo "$$cmd"; \
		$$cmd; \
	else \
		echo "User config not found. Running with default preset 'small'."; \
		cmd="$(PYTHON) scripts/train.py --dataset configs/dataset_mixing.yaml --config-preset small $(TRAIN_OVERRIDES)"; \
		echo "$$cmd"; \
		$$cmd; \
	fi

train-resume:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "Error: Please specify CHECKPOINT=path/to/model.pt"; \
		exit 1; \
	fi
	$(PYTHON) scripts/train.py --dataset configs/dataset_mixing.yaml --resume-from $(CHECKPOINT)

reborn:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "Error: Please specify CHECKPOINT=path/to/elder.pt"; \
		exit 1; \
	fi
	$(PYTHON) scripts/reborn.py --checkpoint $(CHECKPOINT)

merge:
	$(PYTHON) scripts/merge_models.py --help

scale-up:
	$(PYTHON) scripts/auto_scale.py

chat:
	PYTHONPATH=. $(VENV)/bin/streamlit run app.py

dashboard:
	PYTHONPATH=. $(VENV)/bin/streamlit run app.py

clean-safe:
	$(PYTHON) scripts/muse_utils.py clean-safe

deploy:
	$(PYTHON) scripts/deploy_interactive.py

restore:
	$(PYTHON) scripts/muse_utils.py restore-point

pack:
	$(PYTHON) scripts/muse_utils.py pack

check-update:
	$(PYTHON) scripts/muse_utils.py version-guardian

notify:
	$(PYTHON) scripts/muse_utils.py notify

# Phase 7 Hyperbolic Attention Triton smoke test
triton-attn:
	$(PYTHON) scripts/check_hyperbolic_triton.py --use-triton --use-mask --kernel fast --json results/triton_attention_check.json

# Phase 7 Hyperbolic Attention Triton benchmark (compare all kernels)
triton-bench:
	$(PYTHON) scripts/benchmark_hyperbolic_triton.py --batch 4 --seq-len 512 --d-model 256 --heads 8 --json results/benchmarks/hyperbolic_triton_benchmark.json

# Phase 7 Hyperbolic Attention - fast kernel only
triton-fast:
	$(PYTHON) scripts/check_hyperbolic_triton.py --use-triton --use-mask --kernel fast --seq-len 512 --d-model 256 --heads 8 --json results/triton_attention_check.json

# ============================================================================
# Phase 7 Training Commands
# ============================================================================

# Phase 7 Training - Default configuration (RTX 3080 optimized)
train-phase7:
	@if [ ! -f configs/dataset_mixing.yaml ]; then \
		echo "Error: Recipe not found. Please run 'make recipe' first."; \
		exit 1; \
	fi
	$(PYTHON) scripts/train_phase7.py --dataset configs/dataset_mixing.yaml $(TRAIN_OVERRIDES)

# Phase 7 Training - Small configuration for testing
train-phase7-small:
	$(PYTHON) scripts/train_phase7.py --d-model 256 --n-layers 4 --n-seq 256 --batch-size 8 --epochs 1 $(TRAIN_OVERRIDES)

# Phase 7 Training - Large configuration (requires 24GB+ VRAM)
train-phase7-large:
	@if [ ! -f configs/dataset_mixing.yaml ]; then \
		echo "Error: Recipe not found. Please run 'make recipe' first."; \
		exit 1; \
	fi
	$(PYTHON) scripts/train_phase7.py --d-model 768 --n-layers 12 --n-seq 1024 --batch-size 2 --dataset configs/dataset_mixing.yaml $(TRAIN_OVERRIDES)

# Phase 7 Training - Resume from checkpoint
train-phase7-resume:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "Error: Please specify CHECKPOINT=path/to/model.pt"; \
		exit 1; \
	fi
	$(PYTHON) scripts/train_phase7.py --resume-from $(CHECKPOINT) --dataset configs/dataset_mixing.yaml $(TRAIN_OVERRIDES)

# Phase 7 Validation - Run integration tests
test-phase7:
	$(PYTEST) tests/test_phase7_integration.py -v

# Phase 7 Benchmark - Full validation suite
bench-phase7:
	$(PYTHON) benchmarks/phase7_validation.py

# ============================================================================
# Phase 7 Maximum Parameters (1.8B Monster)
# ============================================================================

# Phase 7 Max - 1.8B parameters training (d=4096, L=32)
train-phase7-max:
	@if [ ! -f configs/dataset_mixing.yaml ]; then \
		echo "Warning: Recipe not found. Using dry-run mode."; \
		$(PYTHON) scripts/train_phase7_max.py --config configs/phase7_max_push.yaml --dry-run; \
	else \
		$(PYTHON) scripts/train_phase7_max.py --config configs/phase7_max_push.yaml --dataset configs/dataset_mixing.yaml; \
	fi

# Phase 7 Max - Dry run (test with dummy data)
train-phase7-max-test:
	$(PYTHON) scripts/train_phase7_max.py --config configs/phase7_max_push.yaml --dry-run

# Phase 7 Max - Resume training
train-phase7-max-resume:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "Error: Please specify CHECKPOINT=path/to/model.pt"; \
		exit 1; \
	fi
	$(PYTHON) scripts/train_phase7_max.py --config configs/phase7_max_push.yaml --dataset configs/dataset_mixing.yaml --resume-from $(CHECKPOINT)

# GPU Benchmark - Find maximum parameters for your GPU
gpu-benchmark:
	$(PYTHON) scripts/gpu_benchmark_standalone.py

# ============================================================================
# Phase 7 Chat AI Training (1.8B Monster - Quick Start)
# ============================================================================

# 🚀 チャットAI訓練開始 (最大設定: d=4096, L=32, ~1.8B params)
train-chat:
	@echo "=========================================="
	@echo "🚀 Phase 7 Chat AI Training (1.8B Monster)"
	@echo "=========================================="
	@echo "Config: d_model=4096, n_layers=32, seq=512"
	@echo "VRAM: ~6.89GB (batch=1, gradient_accum=16)"
	@echo ""
	$(PYTHON) scripts/train_phase7_max.py --config configs/phase7_max_push.yaml --dataset configs/dataset_mixing.yaml

# 🧪 ダミーデータでテスト (データセットなしで動作確認)
train-chat-test:
	@echo "=========================================="
	@echo "🧪 Phase 7 Chat AI - Dry Run Test"
	@echo "=========================================="
	$(PYTHON) scripts/train_phase7_max.py --config configs/phase7_max_push.yaml --dry-run

# 📊 GPU性能ベンチマーク (最大パラメータ数を測定)
bench-chat:
	@echo "=========================================="
	@echo "📊 GPU Maximum Parameters Benchmark"
	@echo "=========================================="
	$(PYTHON) scripts/gpu_benchmark_standalone.py

# 🔄 訓練再開
train-chat-resume:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "Error: Please specify CHECKPOINT=path/to/model.pt"; \
		echo "Example: make train-chat-resume CHECKPOINT=checkpoints/phase7_max_push/step_2000.pt"; \
		exit 1; \
	fi
	$(PYTHON) scripts/train_phase7_max.py --config configs/phase7_max_push.yaml --dataset configs/dataset_mixing.yaml --resume-from $(CHECKPOINT)

# ✅ 環境チェック (訓練前に実行推奨)
verify-phase7:
	@echo "=========================================="
	@echo "✅ Phase 7 Environment Verification"
	@echo "=========================================="
	$(PYTHON) scripts/verify_phase7_ready.py

# 🔧 Tritonカーネル動作確認
verify-triton:
	$(PYTHON) scripts/check_hyperbolic_triton.py --use-triton --kernel fast
