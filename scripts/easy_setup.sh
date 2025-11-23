#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# --- Language Selection ---
clear
echo -e "${BLUE}=======================================${NC}"
echo -e "Select Language / 言語を選択してください"
echo -e "1) English (Default)"
echo -e "2) Japanese (日本語)"
echo -e "${BLUE}=======================================${NC}"

read -p "Choice (1/2) [1]: " LANG_CHOICE
LANG_CHOICE=${LANG_CHOICE:-1}

# Save configuration for Makefile
echo "MUSE_LANG=$LANG_CHOICE" > .muse_config

# Define Messages
if [ "$LANG_CHOICE" -eq 2 ]; then
    # Japanese
    MSG_HEADER="MUSE (ResNet-BK) 環境構築スクリプト"
    MSG_STEP1="[Step 1] システムチェック..."
    MSG_PY_FAIL="Error: python3 が見つかりません。インストールしてください。"
    MSG_PY_VER_FAIL="Error: Python 3.9以上が必要です"
    MSG_GPU_OK="✔ NVIDIA GPU detected."
    MSG_GPU_WARN="Warning: NVIDIA GPU が検出されませんでした。CPUモード (Triton無効) で動作します。"
    MSG_APT_ASK="システムライブラリのインストール (sudo権限が必要) を実行しますか？ (y/N): "
    MSG_STEP2="[Step 2] 仮想環境 (venv_ubuntu) のセットアップ..."
    MSG_VENV_CREATING="仮想環境を作成中..."
    MSG_VENV_DONE="✔ 作成完了"
    MSG_VENV_EXIST="✔ 既存の仮想環境を使用します（更新のみ）。"
    MSG_VENV_RECREATE="既存の仮想環境が見つかりました。\n作り直しますか？ (y/N) [N=更新のみ]: "
    MSG_VENV_RECREATING="仮想環境を削除して再作成中..."
    MSG_PIP_UPGRADE="pipをアップグレード中..."
    MSG_STEP3="[Step 3] 依存ライブラリのインストール..."
    MSG_WAIT="これには時間がかかる場合があります (Coffee time ☕)"
    MSG_DEPS_OK="✔ requirements.txt installed."
    MSG_DEPS_FAIL="Error: 依存関係のインストールに失敗しました。"
    MSG_INSTALL_PROJ="プロジェクトをEditableモードでインストール中..."
    MSG_STEP4="[Step 4] データセットの準備"
    MSG_DATA_ASK="1) Lite (テスト用: Cosmopedia 1000件) - 推奨 (数秒)\n2) Full (本番用: 全データ) - (時間がかかります)\n3) Skip (後で行う)\n選択してください (1/2/3) [1]: "
    MSG_DATA_LITE="Liteモードでダウンロード中..."
    MSG_DATA_FULL="Fullモードでダウンロード中..."
    MSG_DATA_SKIP="スキップしました。後で 'make data' または 'make data-lite' で実行できます。"
    MSG_COMPLETE="セットアップ完了！ (MUSE Setup Complete)"
    MSG_NEXT_DEMO="以下のコマンドでデモを起動してみましょう："
    MSG_NEXT_DEV="開発を始めるには："
    MSG_GIT_WARN="[警告] gitの状態がクリーンではありません。Windows側で変更がある場合は git pull を推奨します。"
    MSG_MNT_WARN="[警告] /mnt/c (Windows領域) で実行しています。パフォーマンスが低下する可能性があります。"
else
    # English
    MSG_HEADER="MUSE (ResNet-BK) Setup Script"
    MSG_STEP1="[Step 1] System Check..."
    MSG_PY_FAIL="Error: python3 not found. Please install it."
    MSG_PY_VER_FAIL="Error: Python 3.9+ required"
    MSG_GPU_OK="✔ NVIDIA GPU detected."
    MSG_GPU_WARN="Warning: No NVIDIA GPU detected. Running in CPU mode (Triton disabled)."
    MSG_APT_ASK="Do you want to install system libraries (requires sudo)? (y/N): "
    MSG_STEP2="[Step 2] Virtual Environment Setup (venv_ubuntu)..."
    MSG_VENV_CREATING="Creating virtual environment..."
    MSG_VENV_DONE="✔ Created"
    MSG_VENV_EXIST="✔ Using existing virtual environment (Update only)."
    MSG_VENV_RECREATE="Existing virtual environment found.\nRecreate it? (y/N) [N=Update only]: "
    MSG_VENV_RECREATING="Recreating virtual environment..."
    MSG_PIP_UPGRADE="Upgrading pip..."
    MSG_STEP3="[Step 3] Installing Dependencies..."
    MSG_WAIT="This may take a while (Coffee time ☕)"
    MSG_DEPS_OK="✔ requirements.txt installed."
    MSG_DEPS_FAIL="Error: Failed to install dependencies."
    MSG_INSTALL_PROJ="Installing project in Editable mode..."
    MSG_STEP4="[Step 4] Dataset Preparation"
    MSG_DATA_ASK="1) Lite (Test: Cosmopedia 1k samples) - Recommended\n2) Full (Production: All data)\n3) Skip\nChoice (1/2/3) [1]: "
    MSG_DATA_LITE="Downloading Lite dataset..."
    MSG_DATA_FULL="Downloading Full dataset..."
    MSG_DATA_SKIP="Skipped. You can run 'make data' or 'make data-lite' later."
    MSG_COMPLETE="Setup Complete!"
    MSG_NEXT_DEMO="Run the demo with:"
    MSG_NEXT_DEV="To start developing:"
    MSG_GIT_WARN="[Warning] git status is not clean. Ensure you pulled latest changes."
    MSG_MNT_WARN="[Warning] Running in /mnt/c (Windows mount). Performance may be slow."
fi

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}   $MSG_HEADER   ${NC}"
echo -e "${BLUE}=======================================${NC}"

# --- 0. Pre-Checks ---
if [[ "$PWD" == *"/mnt/c"* ]] || [[ "$PWD" == *"/mnt/d"* ]]; then
    echo -e "${YELLOW}$MSG_MNT_WARN${NC}"
fi

if command -v git &> /dev/null; then
    if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
        echo -e "${YELLOW}$MSG_GIT_WARN${NC}"
    fi
fi

# --- 1. System Check ---
echo -e "\n${YELLOW}$MSG_STEP1${NC}"

# Python Version Check
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}$MSG_PY_FAIL${NC}"
    exit 1
fi

PY_VER=$(python3 -c "import sys; print('%d.%d' % (sys.version_info.major, sys.version_info.minor))")
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]); then
    echo -e "${RED}$MSG_PY_VER_FAIL (Current: $PY_VER)${NC}"
    exit 1
else
    echo -e "${GREEN}✔ Python $PY_VER detected.${NC}"
fi

# GPU Check
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}$MSG_GPU_OK${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}$MSG_GPU_WARN${NC}"
fi

# System Dependencies (Ubuntu/Debian only)
if [ -f /etc/debian_version ] && command -v sudo &> /dev/null; then
    echo -e "\n${YELLOW}[Optional] apt install${NC}"
    read -p "$MSG_APT_ASK" APT_CHOICE
    if [[ "$APT_CHOICE" =~ ^[Yy]$ ]]; then
        sudo apt-get update
        sudo apt-get install -y python3-venv python3-dev build-essential
    fi
fi

# --- 2. Virtual Environment ---
VENV_NAME="venv_ubuntu"
echo -e "\n${YELLOW}$MSG_STEP2${NC}"

if [ -d "$VENV_NAME" ]; then
    read -p "$MSG_VENV_RECREATE" VENV_CHOICE
    if [[ "$VENV_CHOICE" =~ ^[Yy]$ ]]; then
        echo -e "$MSG_VENV_RECREATING"
        rm -rf "$VENV_NAME"
        python3 -m venv "$VENV_NAME"
        echo -e "${GREEN}$MSG_VENV_DONE${NC}"
    else
        echo -e "${GREEN}$MSG_VENV_EXIST${NC}"
    fi
else
    echo -e "$MSG_VENV_CREATING"
    python3 -m venv "$VENV_NAME"
    echo -e "${GREEN}$MSG_VENV_DONE: $VENV_NAME${NC}"
fi

# Activate
source "$VENV_NAME/bin/activate"

# Upgrade Pip
echo -e "$MSG_PIP_UPGRADE"
pip install --upgrade pip setuptools wheel | tail -n 1

# --- 3. Dependencies ---
echo -e "\n${YELLOW}$MSG_STEP3${NC}"
echo -e "$MSG_WAIT"

# Install requirements
if pip install -r requirements.txt; then
    echo -e "${GREEN}$MSG_DEPS_OK${NC}"
else
    echo -e "${RED}$MSG_DEPS_FAIL${NC}"
    exit 1
fi

# Install Project
echo -e "$MSG_INSTALL_PROJ"
pip install -e .

# --- 4. Dataset Setup ---
echo -e "\n${YELLOW}$MSG_STEP4${NC}"
echo -e "$MSG_DATA_ASK"

read -p "" DATA_CHOICE
DATA_CHOICE=${DATA_CHOICE:-1}

case $DATA_CHOICE in
    1)
        echo -e "${BLUE}$MSG_DATA_LITE${NC}"
        python scripts/prepare_datasets.py --datasets cosmopedia --max_samples 1000
        ;;
    2)
        echo -e "${BLUE}$MSG_DATA_FULL${NC}"
        python scripts/prepare_datasets.py --all
        ;;
    *)
        echo -e "$MSG_DATA_SKIP"
        ;;
esac

# --- 5. Finish ---
echo -e "\n${GREEN}=======================================${NC}"
echo -e "${GREEN}   $MSG_COMPLETE  ${NC}"
echo -e "${GREEN}=======================================${NC}"
echo -e "$MSG_NEXT_DEMO"
echo -e "  ${YELLOW}make demo${NC}"
echo -e ""
echo -e "$MSG_NEXT_DEV"
echo -e "  ${YELLOW}source $VENV_NAME/bin/activate${NC}"
