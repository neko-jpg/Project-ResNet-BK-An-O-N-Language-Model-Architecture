.PHONY: help setup install data data-lite test demo clean up down doctor import recipe train-user

# Default shell
SHELL := /bin/bash
VENV := venv_ubuntu
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest

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
		echo "make data-lite  - Download small test dataset"; \
		echo "make data       - Download ALL datasets"; \
		echo "make test       - Run tests"; \
		echo "make demo       - Run MUSE capabilities demo"; \
		echo "make clean      - Remove venv and artifacts"; \
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
	$(PYTHON) scripts/train.py --dataset configs/dataset_mixing.yaml --config-preset small
