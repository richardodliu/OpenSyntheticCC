set -ex

source miniconda3/bin/activate sft
python --version

export NCCL_ALGO=Tree
export PYTHONNOUSERSITE=1

# CONFIG FOR MA
export LOCAL_WORLD_SIZE=$(echo $MA_NUM_GPUS)
export WORLD_SIZE=$(echo $MA_NUM_HOSTS)
export RANK=$(echo $VC_TASK_INDEX)
export MASTER_ADDR=$(echo $VC_WORKER_HOSTS)
export MASTER_PORT=9899

if [[ $MASTER_ADDR != "" ]]; then
  MASTER_ADDR=$(echo $MASTER_ADDR | cut -d',' -f1)
fi

if [[ $MASTER_ADDR == "" ]]; then
  echo "No master address found."
  MASTER_ADDR='127.0.0.1'
fi

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
echo "LOCAL_WORLD_SIZE: $LOCAL_WORLD_SIZE"

DISTRIBUTED_ARGS="
    --nproc_per_node $LOCAL_WORLD_SIZE \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

BATCH_SIZE=128
MICRO_BATCH_SIZE=4
GRAD_ACCU=$(($BATCH_SIZE / $WORLD_SIZE / $MICRO_BATCH_SIZE / $LOCAL_WORLD_SIZE))

LR=1e-5
DATA_PATH="code_precade.jsonl"
OUTPUT_PATH="ckpt/ds-coder-6.7b/code_precade/$LR"
MODEL_PATH="deepseek-coder-6.7b-base"

mkdir -p $OUTPUT_PATH

torchrun ${DISTRIBUTED_ARGS} finetune.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 3 \
    --model_max_length 4096 \
    --per_device_train_batch_size $MICRO_BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRAD_ACCU \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate $LR \
    --warmup_ratio 0.1 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "none" \
    --deepspeed deepspeed.json \
    --bf16 True \
    --seed 3407