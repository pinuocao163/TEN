#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/data/zzb/BaseLine/ten/SDT"
PYTHON_BIN="/home/zzb/anaconda3/envs/wxy/bin/python"
TRAIN_SCRIPT="${PROJECT_DIR}/train_motai.py"
RESULT_DIR="${PROJECT_DIR}/result"

GPU="${GPU:-0}"
RUNS="${RUNS:-10}"
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TEMP="${TEMP:-1}"
LR="${LR:-0.0001}"
GRAD_CLIP="${GRAD_CLIP:-0.0}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${RESULT_DIR}/iemocap_motai_ablation_${RUN_ID}.txt"

mkdir -p "${RESULT_DIR}"
cd "${PROJECT_DIR}"

MODES=(
  "text"
  "audio"
  "visual"
  "text_audio"
  "text_visual"
  "audio_visual"
)

{
  echo "IEMOCAP modality ablation"
  echo "time: $(date '+%F %T')"
  echo "gpu: ${GPU}"
  echo "runs_per_modality: ${RUNS}"
  echo "env: ${PYTHON_BIN}"
  echo "train_script: ${TRAIN_SCRIPT}"
  echo "epochs: ${EPOCHS}"
  echo "batch_size: ${BATCH_SIZE}"
  echo "temp: ${TEMP}"
  echo "lr: ${LR}"
  echo "grad_clip: ${GRAD_CLIP}"
  echo "modalities: ${MODES[*]}"
  echo
} >> "${LOG_FILE}"

for MODE in "${MODES[@]}"; do
  for RUN in $(seq 1 "${RUNS}"); do
    {
      echo "============================================================"
      echo "Dataset=IEMOCAP Modalities=${MODE} Run=${RUN}/${RUNS} Start=$(date '+%F %T')"
      echo "Command:"
      echo "CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON_BIN} -u ${TRAIN_SCRIPT} --Dataset IEMOCAP --modalities ${MODE} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --temp ${TEMP} --lr ${LR} --grad-clip ${GRAD_CLIP}"
    } >> "${LOG_FILE}"

    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u "${TRAIN_SCRIPT}" \
      --Dataset IEMOCAP \
      --modalities "${MODE}" \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --temp "${TEMP}" \
      --lr "${LR}" \
      --grad-clip "${GRAD_CLIP}" \
      >> "${LOG_FILE}" 2>&1

    echo "Dataset=IEMOCAP Modalities=${MODE} Run=${RUN}/${RUNS} End=$(date '+%F %T')" >> "${LOG_FILE}"
  done
done

echo "All IEMOCAP modality ablation runs finished at $(date '+%F %T')" >> "${LOG_FILE}"
echo "Log saved to ${LOG_FILE}"
