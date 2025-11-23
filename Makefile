.PHONY: help setup install data data-lite test demo clean up down doctor import recipe train-user

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
		echo "make recipe     - 学習データの配合設定"; \
		echo "make train-user - 設定したレシピで学習開始"; \
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
	else \
		echo "MUSE (ResNet-BK) Development Commands"; \
		echo "======================================"; \
		echo "make setup      - Full setup (venv, deps, lite data)"; \
		echo "make install    - Install dependencies only"; \
		echo "make doctor     - Run system diagnostics"; \
		echo "make import     - Import user data from data/import/"; \
		echo "make recipe     - Configure dataset mixing recipe"; \
		echo "make train-user - Start training with user recipe"; \
		echo "make train-resume - Resume training (Usage: make train-resume CHECKPOINT=...)"; \
		echo "make reborn       - Reborn Ritual (Usage: make reborn CHECKPOINT=...)"; \
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

train-user:
	@if [ -f configs/auto_optimized.yaml ]; then \
		echo "Using auto-optimized config..."; \
		cmd="$(PYTHON) scripts/train.py --dataset configs/dataset_mixing.yaml --config configs/auto_optimized.yaml $(TRAIN_OVERRIDES)"; \
		echo "$$cmd"; \
		$$cmd; \
	elif [ -f configs/user_train_config.yaml ]; then \
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
