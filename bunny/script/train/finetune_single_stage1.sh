#!/bin/bash
MODEL_TYPE=phi-2

PRETRAIN_DIR='Bunny-v1_0-3B'
OUTPUT_DIR='lora_q_instruct_pathway'

mkdir -p ./output/checkpoints-$MODEL_TYPE/$OUTPUT_DIR

nohup deepspeed --master_port 28801 --include localhost:0,1 bunny/train/train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed bunny/script/deepspeed/zero2.json \
    --model_name_or_path pretrained_weight/Bunny-v1_0-3B/ \
    --model_type $MODEL_TYPE \
    --version bunny \
    --data_path data/q-instruct/pathway_all.json \
    --image_folder data/q-instruct/images/ \
    --vision_tower /path/to/siglip-so400m-patch14-384 \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir VLM_weight_output/q-adapt/checkpoints-$MODEL_TYPE/$OUTPUT_DIR/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
> VLM_weight_output/q-adapt/checkpoints-$MODEL_TYPE/$OUTPUT_DIR/log.log 2>&1 &



