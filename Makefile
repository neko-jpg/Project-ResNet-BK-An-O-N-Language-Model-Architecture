.PHONY: help setup install data data-lite data-ja data-ja-lite test demo clean up down doctor import recipe train-user phase4 build-rust bench-optimization

# Default shell
SHELL := /bin/bash
VENV := venv_ubuntu
export PYTHONPATH := .
# Try to detect if we are in the venv or need to use the path
PYTHON := $(shell if [ -f $(VENV)/bin/python ]; then echo $(VENV)/bin/python; else echo python3; fi)
PIP := $(shell if [ -f $(VENV)/bin/pip ]; then echo $(VENV)/bin/pip; else echo pip; fi)
PYTEST := $(shell if [ -f $(VENV)/bin/pytest ]; then echo $(VENV)/bin/pytest; else echo pytest; fi)

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
		echo "make setup      - 完全セットアップ (仮想環境, 依存関係, Liteデータ, Rustビルド)"; \
		echo "make build-rust - Rustデータローダーのビルド"; \
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
		echo "make compress-10b - 🚀 100億(10B)パラメータモデルの初期化と圧縮"; \
		echo "make train-10b    - 🚀 10B圧縮モデルでの訓練開始 (RTX 3080動作)"; \
		echo ""; \
		echo "Phase 7 (ハイブリッド双曲アテンション - Triton必須):"; \
		echo "make check-phase7-env       - Phase 7環境チェック (CUDA+Triton確認)"; \
		echo "make train-phase7-1.5b      - 🚀 1.5Bパラメータ訓練 (10GB+ VRAM)"; \
		echo "make train-phase7-1.5b-8gb  - 🚀 1.2Bパラメータ訓練 (8GB VRAM最適化)"; \
		echo "make train-phase7-1.5b-test - 🧪 1.5Bモデル動作確認 (ダミーデータ)"; \
		echo "make train-phase7-1.5b-resume CHECKPOINT=... - 🔄 訓練再開"; \
		echo "make bench-phase7-1.5b      - 📊 GPUベンチマーク"; \
		echo "make chat-phase7-1.5b CHECKPOINT=... - 💬 チャット推論"; \
		echo "make train-phase7           - Phase 7モデルの学習 (デフォルト設定)"; \
		echo "make train-phase7-small     - テスト用小規模設定で学習"; \
		echo "make train-phase7-large     - 大規模設定で学習 (24GB+ VRAM)"; \
		echo "make train-phase7-config CONFIG=path/to/config.yaml - カスタム設定で学習"; \
		echo "make train-phase7-resume CHECKPOINT=path/to/model.pt - 訓練再開"; \
		echo "make test-phase7            - Phase 7統合テスト実行"; \
		echo "make triton-attn            - Tritonカーネル動作確認"; \
		echo ""; \
		echo "Phase 8 (双曲超越 - O(N)複雑度):"; \
		echo "make train-phase8       - Phase 8モデルの学習 (O(N)線形アテンション)"; \
		echo "make train-phase8-small - テスト用小規模設定で学習"; \
		echo "make train-phase8-max   - 最大設定で学習 (3B params, 8GB VRAM)"; \
		echo "make train-phase8-test  - ダミーデータでテスト"; \
		echo "make bench-phase8-vs-phase7 - Phase 7とPhase 8の性能比較"; \
		echo "make bench-optimization     - 今回実装した最適化のベンチマーク"; \
	else \
		echo "MUSE (ResNet-BK) Development Commands"; \
		echo "======================================"; \
		echo "make setup      - Full setup (venv, deps, lite data, rust build)"; \
		echo "make build-rust - Build Rust data loader"; \
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
		echo "make compress-10b - 🚀 Initialize and Compress 10B Parameter Model"; \
		echo "make train-10b    - 🚀 Train 10B Compressed Model (RTX 3080 Ready)"; \
		echo ""; \
		echo "Phase 7 (Hybrid Hyperbolic Attention - Triton Required):"; \
		echo "make check-phase7-env       - Check Phase 7 environment (CUDA+Triton)"; \
		echo "make train-phase7-1.5b      - 🚀 Train 1.5B model (10GB+ VRAM)"; \
		echo "make train-phase7-1.5b-8gb  - 🚀 Train 1.2B model (8GB VRAM optimized)"; \
		echo "make train-phase7-1.5b-test - 🧪 Test 1.5B model (dummy data)"; \
		echo "make train-phase7-1.5b-resume CHECKPOINT=... - 🔄 Resume training"; \
		echo "make bench-phase7-1.5b      - 📊 Benchmark GPU"; \
		echo "make chat-phase7-1.5b CHECKPOINT=... - 💬 Chat inference"; \
		echo "make train-phase7           - Train Phase 7 model (default config)"; \
		echo "make train-phase7-small     - Train with small config for testing"; \
		echo "make train-phase7-large     - Train with large config (24GB+ VRAM)"; \
		echo "make train-phase7-config CONFIG=path/to/config.yaml - Train with custom config"; \
		echo "make train-phase7-resume CHECKPOINT=path/to/model.pt - Resume training"; \
		echo "make test-phase7            - Run Phase 7 integration tests"; \
		echo "make triton-attn            - Verify Triton kernel"; \
		echo ""; \
		echo "Phase 8 (Hyperbolic Transcendence - O(N) Complexity):"; \
		echo "make train-phase8       - Train Phase 8 model (O(N) linear attention)"; \
		echo "make train-phase8-small - Train with small config for testing"; \
		echo "make train-phase8-max   - Train with maximum config (3B params, 8GB VRAM)"; \
		echo "make train-phase8-test  - Test with dummy data"; \
		echo "make bench-phase8-vs-phase7 - Benchmark Phase 7 vs Phase 8"; \
		echo "make bench-optimization     - Benchmark new optimizations"; \
	fi'

setup:
	@if [ -f scripts/easy_setup.sh ]; then \
		chmod +x scripts/easy_setup.sh && ./scripts/easy_setup.sh; \
	else \
		$(MAKE) install; \
	fi
	$(MAKE) build-rust

install:
	test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

build-rust:
	cd rust_loader && maturin develop --release

bench-optimization:
	$(PYTHON) src/benchmarks/optimization_benchmark.py

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
	cd rust_loader && cargo clean

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
# Phase 7 Training Commands (Triton必須 - CUDA+Triton Required)
# ============================================================================

# Phase 7環境チェック (Triton必須確認)
check-phase7-env:
	@echo "=========================================="
	@echo "🔍 Phase 7 環境チェック (Triton必須)"
	@echo "=========================================="
	@$(PYTHON) -c "import torch; print('✓ PyTorch:', torch.__version__)" || (echo "❌ PyTorch not found"; exit 1)
	@$(PYTHON) -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('✓ CUDA:', torch.version.cuda)" || (echo "❌ CUDA not available"; exit 1)
	@$(PYTHON) -c "import triton; print('✓ Triton:', triton.__version__)" || (echo "❌ Triton not found. Install: pip install triton"; exit 1)
	@$(PYTHON) -c "from src.kernels.hyperbolic_attention_fast import fast_hyperbolic_attention; print('✓ Hyperbolic Triton kernel loaded')" || (echo "❌ Triton kernel load failed"; exit 1)
	@echo "=========================================="
	@echo "✅ Phase 7環境OK - 訓練可能です"
	@echo "=========================================="

# Phase 7 Training - Default configuration (RTX 3080 optimized, Triton必須)
train-phase7: check-phase7-env
	@if [ ! -f configs/dataset_mixing.yaml ]; then \
		echo "Error: Recipe not found. Please run 'make recipe' first."; \
		exit 1; \
	fi
	@echo "🚀 Phase 7訓練開始 (Tritonカーネル使用)"
	$(PYTHON) scripts/train_phase7.py --dataset configs/dataset_mixing.yaml $(TRAIN_OVERRIDES)

# Phase 7 Training - Small configuration for testing (Triton必須)
train-phase7-small: check-phase7-env
	@echo "🧪 Phase 7小規模テスト訓練 (Tritonカーネル使用)"
	$(PYTHON) scripts/train_phase7.py --d-model 256 --n-layers 4 --n-seq 256 --batch-size 8 --epochs 1 $(TRAIN_OVERRIDES)

# Phase 7 Training - Large configuration (requires 24GB+ VRAM, Triton必須)
train-phase7-large: check-phase7-env
	@if [ ! -f configs/dataset_mixing.yaml ]; then \
		echo "Error: Recipe not found. Please run 'make recipe' first."; \
		exit 1; \
	fi
	@echo "🔥 Phase 7大規模訓練 (Tritonカーネル使用)"
	$(PYTHON) scripts/train_phase7.py --d-model 768 --n-layers 12 --n-seq 1024 --batch-size 2 --dataset configs/dataset_mixing.yaml $(TRAIN_OVERRIDES)

# Phase 7 Training - Resume from checkpoint (Triton必須)
train-phase7-resume: check-phase7-env
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "Error: Please specify CHECKPOINT=path/to/model.pt"; \
		exit 1; \
	fi
	@echo "🔄 Phase 7訓練再開 (Tritonカーネル使用)"
	$(PYTHON) scripts/train_phase7.py --resume-from $(CHECKPOINT) --dataset configs/dataset_mixing.yaml $(TRAIN_OVERRIDES)

# Phase 7 Training - Custom config file (Triton必須)
train-phase7-config: check-phase7-env
	@if [ -z "$(CONFIG)" ]; then \
		echo "Error: Please specify CONFIG=path/to/config.yaml"; \
		echo "Example: make train-phase7-config CONFIG=configs/phase7_optimized.yaml"; \
		exit 1; \
	fi
	@if [ ! -f $(CONFIG) ]; then \
		echo "Error: Config file not found: $(CONFIG)"; \
		exit 1; \
	fi
	@echo "⚙️  Phase 7訓練 (カスタム設定: $(CONFIG))"
	$(PYTHON) scripts/train_phase7.py --config $(CONFIG) $(TRAIN_OVERRIDES)

# Phase 7 Validation - Run integration tests
test-phase7:
	$(PYTEST) tests/test_phase7_integration.py -v

# Phase 7 Benchmark - Full validation suite
bench-phase7:
	$(PYTHON) benchmarks/phase7_validation.py

# ============================================================================
# Phase 8 Training Commands (Hyperbolic Transcendence)
# ============================================================================

# Phase 8 Training - Default configuration (RTX 3080 optimized)
train-phase8:
	@if [ ! -f configs/dataset_mixing.yaml ]; then \
		echo "Error: Recipe not found. Please run 'make recipe' first."; \
		exit 1; \
	fi
	$(PYTHON) scripts/train_phase8.py --config configs/phase8_optimized.yaml --dataset configs/dataset_mixing.yaml $(TRAIN_OVERRIDES)

# Phase 8 Training - Small configuration for testing
train-phase8-small:
	$(PYTHON) scripts/train_phase8.py --d-model 256 --n-layers 4 --n-seq 256 --batch-size 8 --epochs 1 --dry-run $(TRAIN_OVERRIDES)

# Phase 8 Training - Maximum configuration (3B params, 8GB VRAM)
train-phase8-max:
	@if [ ! -f configs/dataset_mixing.yaml ]; then \
		echo "Warning: Recipe not found. Using dry-run mode."; \
		$(PYTHON) scripts/train_phase8.py --config configs/phase8_max_push.yaml --dry-run; \
	else \
		$(PYTHON) scripts/train_phase8.py --config configs/phase8_max_push.yaml --dataset configs/dataset_mixing.yaml; \
	fi

# Phase 8 Training - Maximum with SSM (heavier, experimental)
train-phase8-max-ssm:
	@if [ ! -f configs/dataset_mixing.yaml ]; then \
		echo "Warning: Recipe not found. Using dry-run mode."; \
		$(PYTHON) scripts/train_phase8.py --config configs/phase8_max_push.yaml --use-ssm --dry-run; \
	else \
		$(PYTHON) scripts/train_phase8.py --config configs/phase8_max_push.yaml --use-ssm --dataset configs/dataset_mixing.yaml; \
	fi

# Phase 8 Training - Dry run test
train-phase8-test:
	$(PYTHON) scripts/train_phase8.py --config configs/phase8_max_push.yaml --dry-run

# Phase 8 Training - Resume from checkpoint
train-phase8-resume:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "Error: Please specify CHECKPOINT=path/to/model.pt"; \
		exit 1; \
	fi
	$(PYTHON) scripts/train_phase8.py --config configs/phase8_optimized.yaml --dataset configs/dataset_mixing.yaml --resume-from $(CHECKPOINT) $(TRAIN_OVERRIDES)

# Phase 8 vs Phase 7 Benchmark
bench-phase8-vs-phase7:
	$(PYTHON) scripts/benchmark_phase7_vs_phase8.py

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
# Phase 7 - 1.5B Parameters Training (Triton必須 - 全最適化ON)
# ============================================================================

# 🚀 Phase 7 - 1.5Bパラメータ訓練開始 (Triton必須)
train-phase7-1.5b: check-phase7-env
	@echo "=========================================="
	@echo "🚀 Phase 7 - 1.5B Parameters Training"
	@echo "=========================================="
	@echo "Config: d_model=2048, n_layers=24, seq=1024"
	@echo "Parameters: ~1.5B (1,500,000,000)"
	@echo "VRAM: ~8-10GB (batch=1, gradient_accum=16)"
	@echo "Triton: 必須 (全最適化ON)"
	@echo ""
	@if [ ! -f configs/dataset_mixing.yaml ]; then \
		echo "⚠️  Warning: Recipe not found. Please run 'make recipe' first."; \
		echo "Using dry-run mode for testing..."; \
		$(PYTHON) scripts/train_phase7.py --config configs/phase7_1.5b_triton.yaml --dry-run; \
	else \
		$(PYTHON) scripts/train_phase7.py --config configs/phase7_1.5b_triton.yaml --dataset configs/dataset_mixing.yaml $(TRAIN_OVERRIDES); \
	fi

# 🚀 Phase 7 - 1.5Bパラメータ訓練 (8GB VRAM版)
train-phase7-1.5b-8gb: check-phase7-env
	@echo "=========================================="
	@echo "🚀 Phase 7 - 1.2B Parameters (8GB VRAM)"
	@echo "=========================================="
	@echo "Config: d_model=1792, n_layers=24, seq=512"
	@echo "Parameters: ~1.2B (optimized for 8GB GPU)"
	@echo "VRAM: ~7-8GB (batch=1, gradient_accum=16)"
	@echo "Optimizer: AdamW 8bit (memory efficient)"
	@echo ""
	@if [ ! -f configs/dataset_mixing.yaml ]; then \
		echo "⚠️  Warning: Recipe not found. Using dry-run mode..."; \
		$(PYTHON) scripts/train_phase7.py --config configs/phase7_1.5b_triton_8gb.yaml --dry-run; \
	else \
		$(PYTHON) scripts/train_phase7.py --config configs/phase7_1.5b_triton_8gb.yaml --dataset configs/dataset_mixing.yaml $(TRAIN_OVERRIDES); \
	fi

# 🧪 1.5Bモデル - ダミーデータでテスト
train-phase7-1.5b-test: check-phase7-env
	@echo "=========================================="
	@echo "🧪 Phase 7 - 1.5B Dry Run Test"
	@echo "=========================================="
	$(PYTHON) scripts/train_phase7.py --config configs/phase7_1.5b_triton.yaml --dry-run

# 🔄 1.5Bモデル - 訓練再開
train-phase7-1.5b-resume: check-phase7-env
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "Error: Please specify CHECKPOINT=path/to/model.pt"; \
		echo "Example: make train-phase7-1.5b-resume CHECKPOINT=checkpoints/phase7_1.5b_triton/step_2000.pt"; \
		exit 1; \
	fi
	@echo "🔄 Phase 7 - 1.5B Training Resume"
	$(PYTHON) scripts/train_phase7.py --config configs/phase7_1.5b_triton.yaml --dataset configs/dataset_mixing.yaml --resume-from $(CHECKPOINT) $(TRAIN_OVERRIDES)

# 📊 1.5Bモデル - GPU性能ベンチマーク
bench-phase7-1.5b:
	@echo "=========================================="
	@echo "📊 Phase 7 - 1.5B GPU Benchmark"
	@echo "=========================================="
	$(PYTHON) scripts/gpu_benchmark_phase7.py --config configs/phase7_1.5b_triton.yaml

# 💬 1.5Bモデル - チャット推論
chat-phase7-1.5b:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "========================================"; \
		echo "💬 Phase 7 - 1.5B Chat (Auto-detect)"; \
		echo "========================================"; \
		$(PYTHON) scripts/chat_inference.py --config configs/phase7_1.5b_triton.yaml; \
	else \
		echo "========================================"; \
		echo "💬 Phase 7 - 1.5B Chat"; \
		echo "========================================"; \
		$(PYTHON) scripts/chat_inference.py --config configs/phase7_1.5b_triton.yaml --checkpoint $(CHECKPOINT); \
	fi

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

# 💬 訓練済みモデルでチャット
chat-ai:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "========================================"; \
		echo "💬 MUSE Chat AI (Auto-detect checkpoint)"; \
		echo "========================================"; \
		$(PYTHON) scripts/chat_inference.py; \
	else \
		echo "========================================"; \
		echo "💬 MUSE Chat AI"; \
		echo "========================================"; \
		$(PYTHON) scripts/chat_inference.py --checkpoint $(CHECKPOINT); \
	fi

# ============================================================================
# Phase 8 Extreme Compression (1B -> 10B)
# ============================================================================

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
<<<<<<< HEAD

train-10b-8gb:
	@echo "=========================================="
	@echo "🚀 Starting Extreme Optimization Training (RTX 3080 8GB)"
	@echo "=========================================="
	$(PYTHON) scripts/train_phase8.py --d-model 4096 --n-layers 48 --extreme-compression --dataset configs/dataset_mixing.yaml $(TRAIN_OVERRIDES)
=======
>>>>>>> 99f3f4c6dcba04bfb1d5e20a9f802278fe6d055a
