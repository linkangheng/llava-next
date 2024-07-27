#!/bin/bash
set -x

wandb login

export CUDA_DEVICE_MAX_CONNECTIONS=1
export BATCH_SIZE=4
export GRADIENT_ACCU_STEPS=1
export SMARTRUN=/mnt/shared-storage/tenant/jobutils/scripts/smartrun

export DATA_PATH=data/llava-next/open-llava-next_instruct_mix1M.json
export SAVE_PATH=llava-v1.6-7b_vicuna-7b_clip-large-336_pretrain_lcs-558k_sft-mix1M_lr-mlp-2e-5-vit-2e-6-llm-2e-5
export BASE_LR=2e-5
export VIT_LR=2e-6

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    rlaunch --gpu 8 --cpu 64 --memory=$((1024*512)) --charged-group stepone_mm --private-machine=yes --positive-tags  feature/gpfs=yes -P 1 --i-know-i-am-wasting-resource -- $SMARTRUN \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /mnt/shared-storage/tenant/hypertext/kanelin/models/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ${DATA_PATH} \
    --image_folder data \
    --vision_tower /mnt/shared-storage/tenant/hypertext/kanelin/models/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter /data/Open-LLaVA-NeXT/checkpoints/llava-v1.6-7b_vicuna-7b_pretrain_lcs-558k_ft-mlp-lr-1e-3/mm_projector.bin \
    --unfreeze_mm_vision_tower True \
    --mm_vision_tower_lr ${VIT_LR} \
    --image_aspect_ratio anyres \
    --group_by_modality_length True \
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