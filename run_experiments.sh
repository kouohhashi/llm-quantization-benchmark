#!/usr/bin/env bash
# run_experiments.sh — GGUF本実験ランナー (Machine A)
# セクション7の実験条件マトリクスに従い、有効な (データ × コンテキスト長) の
# 全組み合わせ × 2タスク を benchmark.py で実行し results.csv に蓄積する。
#
# 有効な (data, context_len):
#   T1: 8192, 16384, 32768   (短: 全コンテキスト)
#   T2: 16384, 32768         (中: 16K以上)
#   T3: 32768                (長: 32Kのみ)
# → 6 条件 × 2 タスク(summary/json_extract) = 12 試行 / モデル
#
# 使い方:
#   ./run_experiments.sh <model_type> <model_path> [output_csv]
# 例:
#   ./run_experiments.sh gguf models/gguf/Qwen_Qwen3-8B-Q4_K_M.gguf results/results.csv

set -u
PY="${PY:-.venv/bin/python}"
MODEL_TYPE="${1:?model_type (gguf|awq) を指定}"
MODEL_PATH="${2:?model_path を指定}"
OUT="${3:-results/results.csv}"

# 8GB VRAM(RTX4060 Laptop)で各コンテキスト長が「載る」GPUオフロード層数。
# モデルごとにVRAMフットプリントが違うため NGL_OVERRIDE で上書き可能。
#   - Qwen3-8B(36層,重5.0GB): フル(-1)では16K/32KでKVキャッシュ超過→llama_context
#     生成失敗。実測の安全上限 16K=33/32K=18 を既定にする(34/20は限界ギリのため1段下げ)。
#   - Gemma4-E4B(42層,重5.4GBだが実効4B級でKVも小): 32K最長入力でも全層GPUで6.5GB→
#     `NGL_OVERRIDE=-1` を指定して全コンテキスト フルオフロード。
ngl_for_ctx() {
  if [ -n "${NGL_OVERRIDE:-}" ]; then echo "$NGL_OVERRIDE"; return; fi
  case "$1" in
    8192)  echo -1 ;;   # 36/36 全層GPU (6.6GB)
    16384) echo 33 ;;   # 33/36     (7.63GB)
    32768) echo 18 ;;   # 18/36     (7.32GB)
    *)     echo -1 ;;
  esac
}

# (data_id, context_len) の有効な組み合わせ
declare -a COMBOS=(
  "T1 8192"
  "T1 16384"
  "T1 32768"
  "T2 16384"
  "T2 32768"
  "T3 32768"
)
TASKS=("summary" "json_extract")

mkdir -p "$(dirname "$OUT")"
total=$(( ${#COMBOS[@]} * ${#TASKS[@]} ))
i=0
for combo in "${COMBOS[@]}"; do
  read -r DID CTX <<< "$combo"
  INPUT="data/test_data_${DID}.txt"
  NGL="$(ngl_for_ctx "$CTX")"
  for TASK in "${TASKS[@]}"; do
    i=$((i+1))
    echo "===================================================================="
    echo "[$i/$total] model=$MODEL_TYPE data=$DID ctx=$CTX task=$TASK ngl=$NGL"
    echo "===================================================================="
    "$PY" benchmark.py \
      --model_type "$MODEL_TYPE" \
      --model_path "$MODEL_PATH" \
      --task "$TASK" \
      --input_file "$INPUT" \
      --context_len "$CTX" \
      --n_gpu_layers "$NGL" \
      --output_file "$OUT"
    echo
  done
done
echo "完了: $total 試行を $OUT に記録しました。"
