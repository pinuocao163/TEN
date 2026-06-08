#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/data/zzb/BaseLine/ten/SDT"
RESULT_DIR="${PROJECT_DIR}/result"
SENS_CACHE_DIR="${PROJECT_DIR}/data/param_sensitivity"
TRAIN_SCRIPT="${PROJECT_DIR}/train_1.py"
GEN_SCRIPT="${PROJECT_DIR}/generate_llm_reasoning_rag.py"
TRAIN_PY="${TRAIN_PY:-/home/zzb/anaconda3/envs/wxy/bin/python}"
GEN_PY="${GEN_PY:-/home/zzb/anaconda3/envs/ten_1/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/LLM/Qwen2.5-7B-Instruct}"
PROMPTS="${PROJECT_DIR}/data/meld_llm_prompts.jsonl"
DEFAULT_CACHE="${DEFAULT_CACHE:-${PROJECT_DIR}/data_1/meld_Qwen2.5-7B-Instruct.jsonl}"

GPU_ID="${GPU_ID:-1}"
GEN_GPU_ID="${GEN_GPU_ID:-${GPU_ID}}"
RUNS="${RUNS:-3}"
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEMP="${TEMP:-8}"
LR="${LR:-0.000005}"
GRAD_CLIP="${GRAD_CLIP:-0.0}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-16}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
SENS_GROUP="${SENS_GROUP:-quality}"  # quality | rag_weight | context | all
FORCE_REGENERATE_DEFAULT="${FORCE_REGENERATE_DEFAULT:-0}"
LIMIT="${LIMIT:-}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${RESULT_DIR}/meld_param_sensitivity_${SENS_GROUP}_${RUN_ID}.txt}"
CACHE_RUN_DIR="${SENS_CACHE_DIR}/meld_${RUN_ID}"

mkdir -p "${RESULT_DIR}" "${CACHE_RUN_DIR}"
cd "${PROJECT_DIR}"

log_header() {
  cat >> "${LOG_FILE}" <<EOF
MELD parameter sensitivity experiments
time: $(date '+%F %T')
sens_group: ${SENS_GROUP}
runs_per_setting: ${RUNS}
epochs: ${EPOCHS}
train_gpu: ${GPU_ID}
generation_gpu: ${GEN_GPU_ID}
train_python: ${TRAIN_PY}
generation_python: ${GEN_PY}
train_script: ${TRAIN_SCRIPT}
generation_script: ${GEN_SCRIPT}
model_path: ${MODEL_PATH}
default_cache: ${DEFAULT_CACHE}
cache_run_dir: ${CACHE_RUN_DIR}
generation_batch_size: ${GEN_BATCH_SIZE}
force_regenerate_default: ${FORCE_REGENERATE_DEFAULT}
limit: ${LIMIT:-full}

EOF
}

generate_cache() {
  local label="$1"
  local text_weight="$2"
  local context_window="$3"
  local output_cache="${CACHE_RUN_DIR}/meld_${label}.jsonl"
  local limit_args=()
  if [[ -n "${LIMIT}" ]]; then
    limit_args=(--limit "${LIMIT}")
  fi

  if [[ -s "${output_cache}" ]]; then
    echo "[cache exists] ${output_cache}" >> "${LOG_FILE}"
    echo "${output_cache}"
    return
  fi

  {
    echo "------------------------------------------------------------"
    echo "Generate cache label=${label}, text_rag_weight=${text_weight}, context_window=${context_window}, start=$(date '+%F %T')"
    echo "Output cache: ${output_cache}"
  } >> "${LOG_FILE}"

  local start_ts
  start_ts=$(date +%s)
  CUDA_VISIBLE_DEVICES="${GEN_GPU_ID}" "${GEN_PY}" -u "${GEN_SCRIPT}" \
    --Dataset MELD \
    --prompts "${PROMPTS}" \
    --output "${output_cache}" \
    --model-path "${MODEL_PATH}" \
    --rag-k 5 \
    --context-window "${context_window}" \
    --text-rag-weight "${text_weight}" \
    --dtype bf16 \
    --max-input-tokens "${MAX_INPUT_TOKENS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --generation-batch-size "${GEN_BATCH_SIZE}" \
    --temperature 0.0 \
    --top-p 0.9 \
    --log-every 50 \
    --resume \
    "${limit_args[@]}" \
    >> "${LOG_FILE}" 2>&1
  local end_ts
  end_ts=$(date +%s)
  echo "Generate cache label=${label}, elapsed_sec=$((end_ts - start_ts)), end=$(date '+%F %T')" >> "${LOG_FILE}"
  echo "${output_cache}"
}

train_setting() {
  local setting_name="$1"
  local cache_file="$2"
  local min_quality="$3"
  local min_confidence="$4"

  for iter in $(seq 1 "${RUNS}"); do
    {
      echo "============================================================"
      echo "Dataset=MELD Setting=${setting_name} Run=${iter}/${RUNS} Start=$(date '+%F %T')"
      echo "cache=${cache_file}"
      echo "llm_min_quality=${min_quality}, llm_min_confidence=${min_confidence}"
    } >> "${LOG_FILE}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${TRAIN_PY}" -u "${TRAIN_SCRIPT}" \
      --Dataset MELD \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --temp "${TEMP}" \
      --lr "${LR}" \
      --llm-cache "${cache_file}" \
      --use-llm-reasoning \
      --llm-loss-weight 0.00001 \
      --llm-reliability-weight 0.00001 \
      --vad-contrast-weight 0.00003 \
      --llm-residual-init 0.003 \
      --llm-min-quality "${min_quality}" \
      --llm-min-confidence "${min_confidence}" \
      --grad-clip "${GRAD_CLIP}" \
      >> "${LOG_FILE}" 2>&1

    echo "Dataset=MELD Setting=${setting_name} Run=${iter}/${RUNS} End=$(date '+%F %T')" >> "${LOG_FILE}"
  done
}

run_quality() {
  local settings=(
    "quality_q055_c085:0.55:0.85"
    "quality_q060_c0875:0.60:0.875"
    "quality_q065_c090:0.65:0.90"
    "quality_q070_c0925:0.70:0.925"
    "quality_q075_c095:0.75:0.95"
  )
  for item in "${settings[@]}"; do
    IFS=':' read -r name q c <<< "${item}"
    train_setting "${name}" "${DEFAULT_CACHE}" "${q}" "${c}"
  done
}

run_rag_weight() {
  local weights=("0.3" "0.5" "0.7" "0.9" "1.0")
  for weight in "${weights[@]}"; do
    local label="ragw_${weight//./}"
    local cache_file
    if [[ "${weight}" == "0.7" && "${FORCE_REGENERATE_DEFAULT}" != "1" ]]; then
      cache_file="${DEFAULT_CACHE}"
      echo "[reuse default cache] ${label}: ${cache_file}" >> "${LOG_FILE}"
    else
      cache_file=$(generate_cache "${label}" "${weight}" "3")
    fi
    train_setting "${label}" "${cache_file}" "0.65" "0.90"
  done
}

run_context() {
  local windows=("1" "2" "3" "5" "7")
  for window in "${windows[@]}"; do
    local label="ctx_${window}"
    local cache_file
    if [[ "${window}" == "3" && "${FORCE_REGENERATE_DEFAULT}" != "1" ]]; then
      cache_file="${DEFAULT_CACHE}"
      echo "[reuse default cache] ${label}: ${cache_file}" >> "${LOG_FILE}"
    else
      cache_file=$(generate_cache "${label}" "0.7" "${window}")
    fi
    train_setting "${label}" "${cache_file}" "0.65" "0.90"
  done
}

log_header

case "${SENS_GROUP}" in
  quality)
    run_quality
    ;;
  rag_weight)
    run_rag_weight
    ;;
  context)
    run_context
    ;;
  all)
    run_quality
    run_rag_weight
    run_context
    ;;
  *)
    echo "Unknown SENS_GROUP=${SENS_GROUP}. Use quality | rag_weight | context | all" >&2
    exit 1
    ;;
esac

echo "Finished MELD parameter sensitivity experiments: $(date '+%F %T')" >> "${LOG_FILE}"
echo "Log saved to ${LOG_FILE}"
