#!/bin/bash
set -x

wandb login

export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8
export NNODES=1
export BATCH_SIZE=8
export GRADIENT_ACCU_STEPS=1
export MASTER_PORT=29502
export CPUS_PER_TASK=24
export QUOTA=reserved

export DATA_PATH=data/llava/llava_pretrain/blip_laion_cc_sbu_558k.json
export SAVE_PATH=llava-v1.6-7b_vicuna-7b_pretrain_lcs-558k_ft-mlp-lr-1e-3
export BASE_LR=1e-3

# Get the master node IP address, assuming this script is run on the master node
MASTER_ADDR=$(hostname -I | awk '{print $1}')

# Assuming the current node is node_rank 0
NODE_RANK=0

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=${MASTER_PORT} \
llava/train/train_mem.py \
--deepspeed ./scripts/zero2.json \
--model_name_or_path /mnt/shared-storage/tenant/hypertext/kanelin/models/vicuna-7b-v1.5 \
--version plain \
--data_path ${DATA_PATH} \
--image_folder /mnt/shared-storage/tenant/hypertext/kanelin/data/llava-next/llava/llava_pretrain \
--vision_tower /mnt/shared-storage/tenant/hypertext/kanelin/models/clip-vit-large-patch14-336 \
--mm_projector_type mlp2x_gelu \
--tune_mm_mlp_adapter True \
--unfreeze_mm_vision_tower False \
--image_aspect_ratio anyres \
--mm_vision_select_layer -2 \
--mm_vision_select_feature patch \
--mm_patch_merge_type spatial_unpad \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--bf16 True \
--output_dir checkpoints/${SAVE_PATH} \
--num_train_epochs 1 \
--per_device_train_batch_size ${BATCH_SIZE} \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps ${GRADIENT_ACCU_STEPS} \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 50000 \
--save_total_limit 1 \
--learning_rate ${BASE_LR} \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 4096 \
--gradient_checkpointing True \
--dataloader_num_workers 4 \
--lazy_preprocess True \
--report_to wandb \
--run_name ${SAVE_PATH}

# sudo service slurmd restart 
# sudo service slurmd enable 
# sudo service slurmctld restart 
# sudo service slurmctld enable 