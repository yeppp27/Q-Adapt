
gpu_list="${CUDA_VISIBLE_DEVICES:-1}"  
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_TYPE=phi-2
MODEL_BASE=pretrained_weight/Bunny-v1_0-3B/
TARGET_DIR=qformer_stage2_multiiamges
MODEL_STAGE1=VLM_weight_output/lora_co_instruct_230k/


CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} nohup python  bunny/eval/model_qbench2_multistage.py \
    --model-path VLM_weight_output/checkpoints-$MODEL_TYPE/$TARGET_DIR \
    --model-type $MODEL_TYPE \
    --model-base $MODEL_BASE \
    --model-stage1 $MODEL_STAGE1 \
    --image-folder q-bench2/all_single_images/ \
    --questions-file q-bench2/q-bench2-a1-test_withanswer.jsonl \
    --answers-file VLM_weight_output/checkpoints-$MODEL_TYPE/$TARGET_DIR/qbench2-a1-test.jsonl \
    --device cuda:0 \
    --temperature 0 \
    --conv-mode bunny  \
    --bert-type qformer_pretrain_freeze \
> VLM_weight_output/checkpoints-$MODEL_TYPE/$TARGET_DIR/log_eval_qbench2-test-a1.log 2>&1 &
