# !/bin/bash

##################################### MODEL #####################################
MODEL_NAME="llava-Llama-2-13b-chat-hf-finetune-lora_generation"

MODEL_BASE="Llama-2-13b-chat-hf"
##################################### TIME ######################################
TIME="$(date +'%Y_%m_%d_%H_%M_%S')"
echo "Time is" ${TIME}
################################## CHOOSE CUDA ##################################
export CUDA_VISIBLE_DEVICES=0
echo "CUDA is" ${CUDA_VISIBLE_DEVICES}
###################################### END ######################################


#################################### TESTING ####################################
deepspeed ./llava/eval/test_llava.py \
    --model-path ./path/to/your/models/$MODEL_NAME \
    --model-base /path/to/your/models/$MODEL_BASE \
    --model-name $MODEL_NAME \
    --input-file-path ./data/datasets/llava_test.xlsx \
    --image-path ./data/images \
    --task generation 