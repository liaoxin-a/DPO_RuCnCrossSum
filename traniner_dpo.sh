#!/bin/bash

ARGPARSE_DESCRIPTION="Trainer utility"
source $(dirname $0)/argparse.bash || exit 1
argparse "$@" <<EOF || exit 1

parser.add_argument('--ngpus', default=2, type=int,
                    help='No. of gpus to use')
parser.add_argument('--training_type', type=str, choices=["m2m", "m2o", "o2m"],
                    required=True, help='Training type (many-to-many/many-to-one/one-to-many)')
parser.add_argument('--exclude_native', action='store_true',
                    default=False, help='Exclude the native-to-native filepairs during training')
parser.add_argument('--copy_last_checkpoint', type=str, default="",
                    help='If provided, copies the last checkpoint to this directory')

EOF

export BASE_DIR=$(realpath .)
export ROOT_DATASET_DIR="${BASE_DIR}/dpo_data"
export ROOT_INPUT_DIR="${BASE_DIR}/input"
export ROOT_OUTPUT_DIR="${BASE_DIR}/output"


export PREFIX="${TRAINING_TYPE}_${PIVOT_LANG}"


export SUFFIX="with_native"
if [[ "$EXCLUDE_NATIVE" = "yes" ]]; then
    SUFFIX="without_native"
fi

export BASENAME="${PREFIX}_${SUFFIX}"
export INPUT_DIR="${ROOT_INPUT_DIR}/${BASENAME}"
export OUTPUT_DIR="${ROOT_OUTPUT_DIR}/${BASENAME}"
export MIN_EXAMPLE_COUNT=30

# conda activate "${BASE_DIR}/env" || source activate "${BASE_DIR}/env"

# for ozstar only; the model must
# be cached if this variable is set
export LINK_CACHE_ONLY=false 

# training settings
export max_steps=25000
export save_steps=25000
export logging_steps=100

# validation settings
export evaluation_strategy="no"

# model settings
export model_name="cache_dir/mT5_m2m_crossSum"

# optimization settings
export learning_rate=1
export warmup_steps=5000
export gradient_accumulation_steps=2
export weight_decay=0.01
export lr_scheduler_type="linear"
export label_smoothing_factor=0.1

# misc. settings
export seed=1234

# input / output settings
export input_dir=$INPUT_DIR
export output_dir=$OUTPUT_DIR

# batch / sequence sizes
export PER_DEVICE_TRAIN_BATCH_SIZE=2
export MAX_SOURCE_LENGTH=512
export MAX_TARGET_LENGTH=84

# cross lingual settings
# export per_lang_batch_size=$PER_LANG_BATCH_SIZE

# logging settings
export WANDB_PROJECT="Crossum"
export WANDB_WATCH=false

python -m torch.distributed.launch \
		--nproc_per_node=${NPROC_PER_NODE:-$NGPUS} \
		--nnodes=${SLURM_JOB_NUM_NODES:-1} \
		--node_rank=${SLURM_PROCID:-0} \
		--master_addr="${PARENT:-127.0.0.1}" --master_port="${MPORT:-29500}" "${BASE_DIR}/dpo_train.py" \
    --model_name_or_path $model_name \
    --data_dir $INPUT_DIR --output_dir $OUTPUT_DIR \
    --learning_rate=$learning_rate --warmup_steps $warmup_steps --gradient_accumulation_steps $gradient_accumulation_steps \
    --weight_decay $weight_decay --lr_scheduler_type $lr_scheduler_type --adafactor --label_smoothing_factor $label_smoothing_factor \
    --per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE --logging_steps $logging_steps \
    --max_source_length $MAX_SOURCE_LENGTH --max_target_length $MAX_TARGET_LENGTH \
    # --per_lang_batch_size $per_lang_batch_size \
    --seed $seed --overwrite_output_dir \
    --max_steps $max_steps --save_steps $save_steps \
    --evaluation_strategy $evaluation_strategy  \
    --logging_first_step \
    --cache_dir "${BASE_DIR}/cache_dir" \
    --run_name $BASENAME \
    --use_langid \
    --langid_map_path "${BASE_DIR}/debug/extra_tokens_langid_map.json" \
    --reinitialize_langid_embeddings "bos" \
    # --do_train \
    $(echo -n ${OPTIONAL_ARGS[@]}) |& tee "${OUTPUT_DIR}/run.log"

if [[ "$COPY_LAST_CHECKPOINT" != "" ]]; then
    mkdir -p "$COPY_LAST_CHECKPOINT"
    cp -r "${OUTPUT_DIR}/checkpoint-${max_steps}" "${COPY_LAST_CHECKPOINT}/${BASENAME}"
fi