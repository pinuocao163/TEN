#!/usr/bin/env bash

PROJECT_DIR="/data/zzb/BaseLine/ten/SDT"
RESULT_DIR="${PROJECT_DIR}/result"
DATE_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${RESULT_DIR}/iemocap_best_wxy_rag_vad_llm_100runs_1_${DATE_TAG}.txt}"
TRAIN_SCRIPT="${PROJECT_DIR}/train.py"
LLM_CACHE="${PROJECT_DIR}/data/iemocap_llm_reasoning_mmrag.jsonl"
RUNS="${RUNS:-100}"
EPOCHS="${EPOCHS:-150}"
GPU_ID="${GPU_ID:-1}"

mkdir -p "${RESULT_DIR}"
cd "${RESULT_DIR}" || exit 1

cat > "${LOG_FILE}" <<EOF
Start IEMOCAP best-parameter ${RUNS} runs, epochs=${EPOCHS}: $(date)
Log file: ${LOG_FILE}
Project dir: ${PROJECT_DIR}
Train script: ${TRAIN_SCRIPT}
LLM cache: ${LLM_CACHE}
GPU ID: ${GPU_ID}

Run configuration:
  Dataset: IEMOCAP
  epochs: ${EPOCHS}
  batch-size: 16
  temp: 1
  lr: 0.0001
  use-llm-reasoning: true
  llm-loss-weight: 0.000005
  llm-reliability-weight: 0.000001
  vad-contrast-weight: 0.000002
  llm-residual-init: 0.000001
  llm-min-quality: 0.60
  llm-min-confidence: 0.85
  grad-clip: 0.0

EOF

for iter in $(seq 1 "${RUNS}")
do
  echo "=======================" >> "${LOG_FILE}"
  echo "--- IEMOCAP run ${iter}/${RUNS}: $(date) ---" >> "${LOG_FILE}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" /home/zzb/anaconda3/envs/wxy/bin/python -u "${TRAIN_SCRIPT}" \
    --Dataset IEMOCAP \
    --epochs "${EPOCHS}" \
    --batch-size 16 \
    --temp 1 \
    --lr 0.0001 \
    --llm-cache "${LLM_CACHE}" \
    --use-llm-reasoning \
    --llm-loss-weight 0.000005 \
    --llm-reliability-weight 0.000001 \
    --vad-contrast-weight 0.000002 \
    --llm-residual-init 0.000001 \
    --llm-min-quality 0.60 \
    --llm-min-confidence 0.85 \
    --grad-clip 0.0 \
    >> "${LOG_FILE}" 2>&1
done

echo "Finished IEMOCAP best-parameter ${RUNS} runs: $(date)" >> "${LOG_FILE}"
