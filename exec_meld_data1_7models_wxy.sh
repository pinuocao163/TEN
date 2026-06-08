#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="/data/zzb/BaseLine/ten/SDT"
DATA_DIR="${PROJECT_DIR}/data_1"
RESULT_DIR="${PROJECT_DIR}/result"
DATE_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${RESULT_DIR}/meld_data1_7models_wxy_${DATE_TAG}.txt}"
TRAIN_SCRIPT="${PROJECT_DIR}/train_1.py"
PYTHON_BIN="${PYTHON_BIN:-/home/zzb/anaconda3/envs/wxy/bin/python}"
RUNS="${RUNS:-10}"
EPOCHS="${EPOCHS:-150}"
GPU_ID="${GPU_ID:-1}"

MODEL_NAMES=(
  "Qwen2.5-1.5B-Instruct"
  "Qwen2.5-3B-Instruct"
  "Qwen2.5-7B-Instruct"
  "Qwen2.5-14B-Instruct"
  "Llama-3.1-8B-Instruct"
  "Mistral-7B-Instruct-v0.3"
  "InternLM3-8B-Instruct"
)

CACHE_FILES=(
  "${DATA_DIR}/meld_Qwen2.5-1.5B-Instruct.jsonl"
  "${DATA_DIR}/meld_Qwen2.5-3B-Instruct.jsonl"
  "${DATA_DIR}/meld_Qwen2.5-7B-Instruct.jsonl"
  "${DATA_DIR}/meld_Qwen2.5-14B-Instruct.jsonl"
  "${DATA_DIR}/meld_llama31_8b_instruct.jsonl"
  "${DATA_DIR}/meld_mistral7b_instruct_v03.jsonl"
  "${DATA_DIR}/meld_internlm3_8b_instruct.jsonl"
)

mkdir -p "${RESULT_DIR}"

if [[ -e "${LOG_FILE}" ]]; then
  base="${LOG_FILE%.*}"
  ext="${LOG_FILE##*.}"
  if [[ "${base}" == "${LOG_FILE}" ]]; then
    ext=""
  else
    ext=".${ext}"
  fi
  suffix=1
  while [[ -e "${base}_${suffix}${ext}" ]]; do
    suffix=$((suffix + 1))
  done
  LOG_FILE="${base}_${suffix}${ext}"
fi

cat >> "${LOG_FILE}" <<EOF
Start MELD data_1 7-LLM-cache runs: $(date)
Log file: ${LOG_FILE}
Project dir: ${PROJECT_DIR}
Data dir: ${DATA_DIR}
Train script: ${TRAIN_SCRIPT}
Python: ${PYTHON_BIN}
GPU ID: ${GPU_ID}
Runs per cache: ${RUNS}
Epochs: ${EPOCHS}

Run configuration:
  Dataset: MELD
  batch-size: 8
  temp: 8
  lr: 0.000005
  use-llm-reasoning: true
  llm-loss-weight: 0.00001
  llm-reliability-weight: 0.00001
  vad-contrast-weight: 0.00003
  llm-residual-init: 0.003
  llm-min-quality: 0.65
  llm-min-confidence: 0.90

EOF

cd "${RESULT_DIR}" || exit 1

for idx in "${!MODEL_NAMES[@]}"; do
  MODEL_NAME="${MODEL_NAMES[$idx]}"
  LLM_CACHE="${CACHE_FILES[$idx]}"

  if [[ ! -f "${LLM_CACHE}" ]]; then
    echo "[ERROR] Missing cache for ${MODEL_NAME}: ${LLM_CACHE}" | tee -a "${LOG_FILE}"
    exit 1
  fi

  echo "=======================" >> "${LOG_FILE}"
  echo "Model cache: ${MODEL_NAME}" >> "${LOG_FILE}"
  echo "LLM cache: ${LLM_CACHE}" >> "${LOG_FILE}"

  for iter in $(seq 1 "${RUNS}"); do
    echo "--- MELD ${MODEL_NAME} run ${iter}/${RUNS}: $(date) ---" >> "${LOG_FILE}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u "${TRAIN_SCRIPT}" \
      --Dataset MELD \
      --epochs "${EPOCHS}" \
      --batch-size 8 \
      --temp 8 \
      --lr 0.000005 \
      --llm-cache "${LLM_CACHE}" \
      --use-llm-reasoning \
      --llm-loss-weight 0.00001 \
      --llm-reliability-weight 0.00001 \
      --vad-contrast-weight 0.00003 \
      --llm-residual-init 0.003 \
      --llm-min-quality 0.65 \
      --llm-min-confidence 0.90 \
      >> "${LOG_FILE}" 2>&1
  done
done

echo "Finished MELD data_1 7-LLM-cache runs: $(date)" >> "${LOG_FILE}"
