#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="/data/zzb/BaseLine/ten/SDT"
DATA_DIR="${PROJECT_DIR}/data_1"
RESULT_DIR="${PROJECT_DIR}/result"
DATE_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${RESULT_DIR}/iemocap_ablation_wxy_${DATE_TAG}.txt}"
TRAIN_SCRIPT="${PROJECT_DIR}/train.py"
PYTHON_BIN="${PYTHON_BIN:-/home/zzb/anaconda3/envs/wxy/bin/python}"
LLM_CACHE="${LLM_CACHE:-${DATA_DIR}/iemocap_Qwen2.5-7B-Instruct.jsonl}"
RUNS="${RUNS:-10}"
EPOCHS="${EPOCHS:-150}"
GPU_ID="${GPU_ID:-0}"

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
Start IEMOCAP ablation runs: $(date)
Log file: ${LOG_FILE}
Project dir: ${PROJECT_DIR}
Train script: ${TRAIN_SCRIPT}
Python: ${PYTHON_BIN}
LLM cache: ${LLM_CACHE}
GPU ID: ${GPU_ID}
Runs per setting: ${RUNS}
Epochs: ${EPOCHS}

Base configuration:
  Dataset: IEMOCAP
  batch-size: 16
  temp: 1
  lr: 0.0001
  full llm-loss-weight: 0.00005
  full llm-reliability-weight: 0.00001
  full vad-contrast-weight: 0.00002
  full llm-residual-init: 0.00001
  full llm-min-quality: 0.60
  full llm-min-confidence: 0.85
  grad-clip: 0.0

EOF

if [[ ! -f "${LLM_CACHE}" ]]; then
  echo "[ERROR] Missing LLM cache: ${LLM_CACHE}" | tee -a "${LOG_FILE}"
  exit 1
fi

cd "${RESULT_DIR}" || exit 1

run_setting() {
  local setting_name="$1"
  local setting_desc="$2"
  shift 2

  echo "=======================" >> "${LOG_FILE}"
  echo "Ablation setting: ${setting_name}" >> "${LOG_FILE}"
  echo "Description: ${setting_desc}" >> "${LOG_FILE}"
  echo "Extra args: $*" >> "${LOG_FILE}"

  for iter in $(seq 1 "${RUNS}"); do
    echo "--- IEMOCAP ${setting_name} run ${iter}/${RUNS}: $(date) ---" >> "${LOG_FILE}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u "${TRAIN_SCRIPT}" \
      --Dataset IEMOCAP \
      --epochs "${EPOCHS}" \
      --batch-size 16 \
      --temp 1 \
      --lr 0.0001 \
      --grad-clip 0.0 \
      "$@" \
      >> "${LOG_FILE}" 2>&1
  done
}

run_setting \
  "A0_TEN_Baseline" \
  "No LLM cache, no RAG/VAD/LLM reasoning branch."

run_setting \
  "A1_w_o_Structured_Cognition" \
  "Remove structured cognition variables; equivalent to multimodal backbone without LLM reasoning."

run_setting \
  "A2_w_o_LLM_Distribution_Alignment" \
  "Use cognitive fusion but remove LLM soft-label KL alignment." \
  --llm-cache "${LLM_CACHE}" \
  --use-llm-reasoning \
  --llm-loss-weight 0 \
  --llm-reliability-weight 0.00001 \
  --vad-contrast-weight 0.00002 \
  --llm-residual-init 0.00001 \
  --llm-min-quality 0.60 \
  --llm-min-confidence 0.85

run_setting \
  "A3_w_o_Reliability_Alignment" \
  "Use cognitive fusion but remove reliability supervision loss." \
  --llm-cache "${LLM_CACHE}" \
  --use-llm-reasoning \
  --llm-loss-weight 0.00005 \
  --llm-reliability-weight 0 \
  --vad-contrast-weight 0.00002 \
  --llm-residual-init 0.00001 \
  --llm-min-quality 0.60 \
  --llm-min-confidence 0.85

run_setting \
  "A4_w_o_Affective_Geometry" \
  "Use cognitive fusion but remove VAD-aware contrastive loss." \
  --llm-cache "${LLM_CACHE}" \
  --use-llm-reasoning \
  --llm-loss-weight 0.00005 \
  --llm-reliability-weight 0.00001 \
  --vad-contrast-weight 0 \
  --llm-residual-init 0.00001 \
  --llm-min-quality 0.60 \
  --llm-min-confidence 0.85

run_setting \
  "A5_w_o_Quality_Gate" \
  "Use full losses but remove quality and confidence filtering." \
  --llm-cache "${LLM_CACHE}" \
  --use-llm-reasoning \
  --llm-loss-weight 0.00005 \
  --llm-reliability-weight 0.00001 \
  --vad-contrast-weight 0.00002 \
  --llm-residual-init 0.00001 \
  --llm-min-quality 0.0 \
  --llm-min-confidence 0.0

run_setting \
  "A6_w_o_Dynamic_Calibration" \
  "Weaken cognitive calibration by using the minimum residual initialization." \
  --llm-cache "${LLM_CACHE}" \
  --use-llm-reasoning \
  --llm-loss-weight 0.00005 \
  --llm-reliability-weight 0.00001 \
  --vad-contrast-weight 0.00002 \
  --llm-residual-init 0.0001 \
  --llm-min-quality 0.60 \
  --llm-min-confidence 0.85

run_setting \
  "A7_Full_EACC" \
  "Full EACC with RAG-LLM cognition, reliability alignment, VAD contrast, quality gate, and calibration." \
  --llm-cache "${LLM_CACHE}" \
  --use-llm-reasoning \
  --llm-loss-weight 0.00005 \
  --llm-reliability-weight 0.00001 \
  --vad-contrast-weight 0.00002 \
  --llm-residual-init 0.00001 \
  --llm-min-quality 0.60 \
  --llm-min-confidence 0.85

echo "Finished IEMOCAP ablation runs: $(date)" >> "${LOG_FILE}"

