#!/usr/bin/env bash

export DATA_DIR=/home/mxdong/Data/MuSiQue/class_data/sequence_data
# export DATA_DIR=/home/mxdong/Data/MuSiQue/class_data/parallel_data

export OUTPUT_DIR=/home/mxdong/Model/Decomposition/MuSiQue/Sequence
# export OUTPUT_DIR=/home/mxdong/Model/Decomposition/MuSiQue/Parallel


CUDA_VISIBLE_DEVICES=0 python ../run_fine_tuning.py \
    --model_type=bart \
    --model_name_or_path=facebook/bart-large \
    --data_dir ${DATA_DIR} \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --per_gpu_train_batch_size=32 \
    --per_gpu_eval_batch_size=32 \
    --gradient_accumulation_steps=1 \
    --logging_steps 750 \
    --max_src_len 160\
    --max_tgt_len 64\
    --output_dir ${OUTPUT_DIR}\
    --num_train_epochs=2 \
    --warmup_steps 100 \
    --overwrite_output_dir \
    --save_steps -1 \
    --learning_rate 1e-5 \
    --adam_epsilon 1e-8 \
    --seed 42 \
