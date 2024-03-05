#!/bin/bash

############ META-LLaMA-2-chat ############
PROMPT_VERSION="llava_llama_2"
MODEL_VERSION="Llama-2-13b-chat-hf"
################### END ###################

################## CUDA ####################
export CUDA_VISIBLE_DEVICES=0
echo "CUDA IS" ${CUDA_VISIBLE_DEVICES}
################## CUDA ####################

    # --pretrain_mm_mlp_adapter /AI/models/llava-336px-pretrain-vicuna-13b-v1.3/mm_projector.bin  \

################# TRAINING #################
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --lora_enable True \
    --model_name_or_path /path/to/your/models/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path ./data/datasets/iglu_generation_train.json \
    --image_folder ./data/images \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-$MODEL_VERSION-finetune-lora_generation \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "linear" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none 