cd  /mnt/workspace/workgroup/lyt/Q-Adapt/

PRETRAIN_DIR='Bunny-v1_0-3B'
OUTPUT_DIR='lora_co_instruct_pathway_230k'
mkdir -p ./output/checkpoints-$MODEL_TYPE/$OUTPUT_DIR

nohup deepspeed --master_port 28802 --include localhost:0,1,2,3 bunny/train/train_multiimages.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed bunny/script/deepspeed/zero2.json \
    --model_name_or_path pretrained_weight/Bunny-v1_0-3B/ \
    --model_type $MODEL_TYPE \
    --version bunny \
    --data_path data/coinstruct/coinstruct_230k.json \
    --image_folder data/coinstruct/images/ \
    --vision_tower /path/to/siglip-so400m-patch14-384 \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 False \
    --fp16 True \
    --output_dir ./output/checkpoints-$MODEL_TYPE/$OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --multimodal_keywords mm_projector,vision_tower,vlm_att_encoder \
    --tune_mm_mlp_adapter True \
    --freeze_backbone True \
> VLM_weight_output/q-adapt/checkpoints-$MODEL_TYPE/$OUTPUT_DIR/log.log 2>&1 &
