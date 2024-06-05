#!/bin/bash
ROOT_DATASET_DIR="dataset"
RESULTS_DIR="evaluation_results"
MODEL_DIR="csebuetnlp/mT5_m2m_crossSum"

# suffix=$(basename $(realpath $MODEL_DIR/..))
python evaluator.py \
    --dataset_dir "${ROOT_DATASET_DIR}" \
    --output_dir "${RESULTS_DIR}/mT5_m2m_crossSum" \
    --evaluation_type xlingual \
    --data_type "val" \
    --xlingual_summarization_model_name_or_path $MODEL_DIR \