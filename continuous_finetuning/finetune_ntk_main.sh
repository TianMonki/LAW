#!/bin/bash
# ----------------- Scripts for origin Llama, PI, NTK and YaRN Methos-------------------
RECIPE_NAME=main # option:[baseline1, baseline2, baseline3, baseline4]     
METHOD_NAME=ntk # option:[origin, pi, ntk, yarn]
TRAINING_LENGTH=65536 
# export CUDA_VISIBLE_DEVICES=0,1,2,3
WANDB_NAME=${RECIPE_NAME}_${METHOD_NAME}_${TRAINING_LENGTH}

torchrun  --nproc_per_node=8 \
        continuous_finetuning/fine-tune.py  \
        --model_name_or_path "llm/Llama-2-7b-hf" \
        --bf16 True \
        --output_dir ckpts/${RECIPE_NAME}/${WANDB_NAME} \
        --model_max_length ${TRAINING_LENGTH} \
        --use_flash_attn True \
        --low_rank_training False \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 16 \
        --evaluation_strategy "no" \
        --save_strategy "epoch" \
        --save_total_limit 2 \
        --learning_rate 2e-5 \
        --weight_decay 0.0 \
        --warmup_steps 20 \
        --deepspeed continuous_finetuning/ds_configs/stage3_offload.json \
        --lr_scheduler_type "constant_with_warmup" \
        --logging_steps 5     \
        --tf32 True \
        --report_to "none" \
        --use_wandb False \
        --dataset_dir data/train/mixed_long_context.jsonl \
        --method_name ${METHOD_NAME} \
        --wandb_name ${WANDB_NAME} 
