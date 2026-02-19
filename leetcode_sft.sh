set -ex

source /volume/pt-train/users/rbliu/miniconda3/bin/activate openrlhf
python --version

export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1

# CONFIG FOR MULTI-NODE
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${NNODES:-"1"}
NODE_RANK=${NODE_RANK:-"0"}

# 计算总卡数
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NNODES: $NNODES"
echo "NODE_RANK: $NODE_RANK"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "WORLD_SIZE: $WORLD_SIZE"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

for LR in 1e-6 2e-6 5e-6 1e-5; do

    NUM_EPOCHS=3
    BATCH_SIZE=128

    MICRO_BATCH_SIZE=16
    GRAD_ACCU=$(($BATCH_SIZE / $WORLD_SIZE / $MICRO_BATCH_SIZE))

    echo "BATCH_SIZE: $BATCH_SIZE"
    echo "MICRO_BATCH_SIZE: $MICRO_BATCH_SIZE"
    echo "GRAD_ACCU: $GRAD_ACCU"

    DATA_NAME="all_dataset"
    DATA_PATH="/volume/pt-train/users/rbliu/dataset/leetcode_dataset/$DATA_NAME.jsonl"

    MODEL_NAME="deepseek-coder-6.7b-base"
    MODEL_PATH="/volume/pt-train/users/rbliu/model/$MODEL_NAME"

    OUTPUT_PATH="/volume/pt-train/users/rbliu/checkpoint/leetcode/${MODEL_NAME}/${DATA_NAME}/epoch-${NUM_EPOCHS}_batch-${BATCH_SIZE}_lr-${LR}"
    mkdir -p $OUTPUT_PATH

    LOG_FILE="/volume/pt-train/users/rbliu/logs/leetcode/${MODEL_NAME}/${DATA_NAME}/epoch-${NUM_EPOCHS}_batch-${BATCH_SIZE}_lr-${LR}.log"
    mkdir -p "$(dirname "$LOG_FILE")"

    cd /volume/pt-train/users/rbliu/github/OpenSyntheticCC

    torchrun ${DISTRIBUTED_ARGS} finetune.py \
        --model_name_or_path $MODEL_PATH \
        --data_path $DATA_PATH \
        --output_dir $OUTPUT_PATH \
        --num_train_epochs $NUM_EPOCHS \
        --model_max_length 4096 \
        --per_device_train_batch_size $MICRO_BATCH_SIZE \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps $GRAD_ACCU \
        --eval_strategy "no" \
        --save_strategy "steps" \
        --save_steps 100 \
        --save_total_limit 100 \
        --learning_rate $LR \
        --warmup_ratio 0.1 \
        --logging_steps 1 \
        --lr_scheduler_type "cosine" \
        --gradient_checkpointing True \
        --report_to "none" \
        --deepspeed deepspeed/zero2.json \
        --bf16 True \
        --seed 347 2>&1 | tee "$LOG_FILE"
done