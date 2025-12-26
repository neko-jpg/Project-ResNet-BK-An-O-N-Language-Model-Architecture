#!/bin/bash
# rinnaトークナイザーでデータセットを再生成するスクリプト
# 使用方法: ./scripts/regenerate_datasets.sh

set -e

cd "$(dirname "$0")/.."

echo "=================================="
echo "rinnaトークナイザーでデータセット再生成"
echo "=================================="

CONFIG_PATH="${CONFIG_PATH:-configs/phase8_300m_japanese_chat.yaml}"
DATASET_CONFIG="${DATASET_CONFIG:-configs/dataset_japanese_chat_optimized.yaml}"
TOKENIZER_NAME="${TOKENIZER_NAME:-rinna/japanese-gpt-neox-3.6b-instruction-sft-v2}"
TOKENS_PER_PARAM="${TOKENS_PER_PARAM:-3}"
VAL_RATIO="${VAL_RATIO:-0.05}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

# venv有効化
source venv_ubuntu/bin/activate

# HuggingFace Token (環境変数で設定してください)
if [ -z "$HF_TOKEN" ]; then
    echo "エラー: HF_TOKEN 環境変数を設定してください"
    echo "例: export HF_TOKEN=your_huggingface_token"
    exit 1
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

echo "HF_TOKEN: ${HF_TOKEN:0:10}..."

# Build token plan from model config + dataset mix
PLAN_FILE="$(mktemp)"
python - <<'PY' > "$PLAN_FILE"
import os
import re
import sys
import yaml

config_path = os.environ.get("CONFIG_PATH", "configs/phase8_300m_japanese_chat.yaml")
dataset_path = os.environ.get("DATASET_CONFIG", "configs/dataset_japanese_chat_optimized.yaml")
tokens_per_param = float(os.environ.get("TOKENS_PER_PARAM", "3"))

if not os.path.exists(config_path):
    print(f"Missing config: {config_path}", file=sys.stderr)
    sys.exit(1)
if not os.path.exists(dataset_path):
    print(f"Missing dataset config: {dataset_path}", file=sys.stderr)
    sys.exit(1)

cfg = yaml.safe_load(open(config_path, "r"))
d_model = int(cfg.get("d_model", 0))
n_layers = int(cfg.get("n_layers", 0))
n_seq = int(cfg.get("n_seq", 0))
vocab_size = int(cfg.get("vocab_size", 0))

if not all([d_model, n_layers, n_seq, vocab_size]):
    print("Config is missing required model fields", file=sys.stderr)
    sys.exit(1)

# Dense-style estimate: embedding + layers (12*d^2 + ln) + output head
embedding_params = vocab_size * d_model + n_seq * d_model
layer_params = n_layers * (12 * d_model * d_model + 4 * d_model)
output_params = d_model * vocab_size
total_params = embedding_params + layer_params + output_params

target_tokens = int(total_params * tokens_per_param)

ds_cfg = yaml.safe_load(open(dataset_path, "r"))
weights = {k: float(v.get("weight", 0.0)) for k, v in (ds_cfg.get("datasets") or {}).items()}
weight_sum = sum(weights.values()) or 1.0

print(f"TOTAL_PARAMS={total_params}")
print(f"TARGET_TOKENS={target_tokens}")
for name, weight in weights.items():
    norm = weight / weight_sum
    token_target = int(target_tokens * norm)
    key = re.sub(r"[^A-Za-z0-9]+", "_", name).upper()
    print(f"TOKENS_{key}={token_target}")
PY

# shellcheck disable=SC1090
source "$PLAN_FILE"
rm -f "$PLAN_FILE"

echo ""
echo "Config: $CONFIG_PATH"
echo "Dataset config: $DATASET_CONFIG"
echo "Estimated params: ${TOTAL_PARAMS}"
echo "Target tokens: ${TARGET_TOKENS} (tokens/param=${TOKENS_PER_PARAM})"
echo "Token plan:"
echo "  japanese_instruct: ${TOKENS_JAPANESE_INSTRUCT:-0}"
echo "  dolly_ja:          ${TOKENS_DOLLY_JA:-0}"
echo "  wiki_ja:           ${TOKENS_WIKI_JA:-0}"
echo "  mc4_ja:            ${TOKENS_MC4_JA:-0}"

# 既存データをバックアップ
echo ""
echo "既存データをバックアップ中..."
for ds in japanese_instruct dolly_ja wiki_ja mc4_ja; do
    if [ -d "data/$ds" ]; then
        mv "data/$ds" "data/${ds}_backup_$(date +%Y%m%d%H%M%S)" 2>/dev/null || true
    fi
done

# データセット再生成
echo ""
echo "Rebuilding japanese_instruct..."
python scripts/prepare_datasets.py \
    --datasets japanese_instruct \
    --tokenizer "$TOKENIZER_NAME" \
    --token "$HF_TOKEN" \
    --max_samples "$MAX_SAMPLES" \
    --max_tokens "${TOKENS_JAPANESE_INSTRUCT:-0}" \
    --val_ratio "$VAL_RATIO"

echo ""
echo "Rebuilding dolly_ja..."
python scripts/prepare_datasets.py \
    --datasets dolly_ja \
    --tokenizer "$TOKENIZER_NAME" \
    --token "$HF_TOKEN" \
    --max_samples "$MAX_SAMPLES" \
    --max_tokens "${TOKENS_DOLLY_JA:-0}" \
    --val_ratio "$VAL_RATIO"

echo ""
echo "Rebuilding wiki_ja..."
python scripts/prepare_datasets.py \
    --datasets wiki_ja \
    --tokenizer "$TOKENIZER_NAME" \
    --token "$HF_TOKEN" \
    --max_samples "$MAX_SAMPLES" \
    --max_tokens "${TOKENS_WIKI_JA:-0}" \
    --val_ratio "$VAL_RATIO"

echo ""
echo "Rebuilding mc4_ja..."
python scripts/prepare_datasets.py \
    --datasets mc4_ja \
    --tokenizer "$TOKENIZER_NAME" \
    --token "$HF_TOKEN" \
    --max_samples "$MAX_SAMPLES" \
    --max_tokens "${TOKENS_MC4_JA:-0}" \
    --val_ratio "$VAL_RATIO"

echo ""
echo "=================================="
echo "完了！"
echo "=================================="

# 検証
echo ""
echo "トークナイザー整合性を検証..."
python scripts/verify_tokenizer_compat.py
