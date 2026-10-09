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


LR=1e-6
NUM_EPOCHS=4
BATCH_SIZE=256

MICRO_BATCH_SIZE=8
GRAD_ACCU=$(($BATCH_SIZE / $WORLD_SIZE / $MICRO_BATCH_SIZE))

echo "BATCH_SIZE: $BATCH_SIZE"
echo "MICRO_BATCH_SIZE: $MICRO_BATCH_SIZE"
echo "GRAD_ACCU: $GRAD_ACCU"

DATA_NAME="jd_dataset"
DATA_PATH="/volume/pt-train/users/rbliu/rbliu/jd/$DATA_NAME.jsonl"

MODEL_NAME="Qwen2.5-Coder-7B-Instruct"
MODEL_PATH="/volume/pt-train/users/rbliu/model/$MODEL_NAME"

OUTPUT_PATH="/volume/pt-train/users/rbliu/rbliu/jd/checkpoint/${MODEL_NAME}/${DATA_NAME}/epoch-${NUM_EPOCHS}_batch-${BATCH_SIZE}_lr-${LR}"
mkdir -p $OUTPUT_PATH

LOG_FILE="/volume/pt-train/users/rbliu/rbliu/jd/logs/${MODEL_NAME}/${DATA_NAME}/epoch-${NUM_EPOCHS}_batch-${BATCH_SIZE}_lr-${LR}.log"
EVAL_LOG_FILE="/volume/pt-train/users/rbliu/rbliu/jd/logs/${MODEL_NAME}/${DATA_NAME}/epoch-${NUM_EPOCHS}_batch-${BATCH_SIZE}_lr-${LR}-eval.log"
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
    --save_total_limit 9999 \
    --learning_rate $LR \
    --warmup_ratio 0.1 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "none" \
    --deepspeed deepspeed/zero2.json \
    --bf16 True \
    --seed 347 \
    2>&1 | tee "$LOG_FILE"

sleep 60

cd /volume/pt-train/users/rbliu/github/OpenCodeEval/agent

# 配置参数
CONFIG_PATH="/volume/pt-train/users/rbliu/github/OpenCodeEval/agent/config/humaneval.json"           # benchmark配置文件
NUM_GPUS=1                                   # 每个checkpoint使用的GPU数
NUM_WORKERS=8                                # 每个checkpoint使用的CPU worker数
MAX_PARALLEL=8                               # 最大并行任务数（可选）

# 示例1: 基本用法（自动计算并行数）
python parallel_eval.py \
    --checkpoint_dir "${OUTPUT_PATH}" \
    --config_path "${CONFIG_PATH}" \
    --num_gpus ${NUM_GPUS} \
    --num_workers ${NUM_WORKERS} \
    2>&1 | tee "${EVAL_LOG_FILE}"