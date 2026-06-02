#!/usr/bin/env bash

PROJECT_DIR="/data/zzb/BaseLine/ten/SDT"
RESULT_DIR="${PROJECT_DIR}/result"
DATE_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${RESULT_DIR}/meld_best_wxy_rag_vad_llm_100runs_1_${DATE_TAG}.txt}"
TRAIN_SCRIPT="${PROJECT_DIR}/train_1.py"
LLM_CACHE="${PROJECT_DIR}/data/meld_llm_reasoning_mmrag.jsonl"
RUNS="${RUNS:-100}"
EPOCHS="${EPOCHS:-150}"
GPU_ID="${GPU_ID:-1}"

mkdir -p "${RESULT_DIR}"
cd "${RESULT_DIR}" || exit 1

cat > "${LOG_FILE}" <<EOF
Start MELD best-parameter ${RUNS} runs, epochs=${EPOCHS}: $(date)
Log file: ${LOG_FILE}
Project dir: ${PROJECT_DIR}
Train script: ${TRAIN_SCRIPT}
LLM cache: ${LLM_CACHE}
GPU ID: ${GPU_ID}

Run configuration:
  Dataset: MELD
  epochs: ${EPOCHS}
  batch-size: 8
  temp: 8
  lr: 0.000005
  use-llm-reasoning: true
  llm-loss-weight: 0.000001
  llm-reliability-weight: 0.000001
  vad-contrast-weight: 0.000003
  llm-residual-init: 0.0003
  llm-min-quality: 0.65
  llm-min-confidence: 0.90

EOF

for iter in $(seq 1 "${RUNS}")
do
  echo "=======================" >> "${LOG_FILE}"
  echo "--- MELD run ${iter}/${RUNS}: $(date) ---" >> "${LOG_FILE}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" /home/zzb/anaconda3/envs/wxy/bin/python -u "${TRAIN_SCRIPT}" \
    --Dataset MELD \
    --epochs "${EPOCHS}" \
    --batch-size 8 \
    --temp 8 \
    --lr 0.000005 \
    --llm-cache "${LLM_CACHE}" \
    --use-llm-reasoning \
    --llm-loss-weight 0.000001 \
    --llm-reliability-weight 0.000001 \
    --vad-contrast-weight 0.000003 \
    --llm-residual-init 0.0003 \
    --llm-min-quality 0.65 \
    --llm-min-confidence 0.90 \
    >> "${LOG_FILE}" 2>&1
done

echo "Finished MELD best-parameter ${RUNS} runs: $(date)" >> "${LOG_FILE}"
