#!/usr/bin/env bash

# bsub -n 1 -q HPC.S1.GPU.X795.suda -o train.generation.gpt2.log -gpu num=1:mode=exclusive_process sh finetune.sh

nvidia-smi
CUDA_VISIBLE_DEVICES=0 python run_fine_tuning.py \
    --model_type=gpt2 \
    --model_name_or_path=/SISDC_GPFS/Home_SE/hy-suda/zfli/Models/init_model/gpt2-large \
    --data_dir=../../Data/protoQA/ \
    --do_train --do_eval --evaluate_during_training \
    --per_gpu_train_batch_size=8 \
    --per_gpu_eval_batch_size=64 \
    --gradient_accumulation_steps=1 \
    --logging_steps 500 \
    --output_dir=outputs/gpt2_finetune_8848 \
    --num_train_epochs=1 \
    --warmup_steps 150 \
    --no_cuda --overwrite_output_dir \
    --save_steps -1 --learning_rate 1e-5 \
    --adam_epsilon 1e-6 --seed 8848
    # --train_name _similarity --max_steps 5621
    # For full data  remove train_name and max_steps 
    # For non-overlap  --train_name _non-overlap --max_steps 5621
CUDA_VISIBLE_DEVICES=0 python run_generation.py \
    --model_type gpt2 \
    --model_name_or_path outputs/gpt2_finetune_8848 \
    --length=10 \
    --num_samples=300 \
    --temperature=0.69 \
    --input_file='../../Data/protoQA/dev.crowdsourced.jsonl' 


# For CommonGen 
# CUDA_VISIBLE_DEVICES=0 python run_commongen.py \
#     --model_type=bart \
#     --model_name_or_path=facebook/bart-large \
#     --data_dir ../../Data/CommonGen/ \
#     --do_train --do_eval --evaluate_during_training \
#     --per_gpu_train_batch_size=16 \
#     --per_gpu_eval_batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --logging_steps 500 \
#     --output_dir bart_finetune_commongen \
#     --num_train_epochs 2 \
#     --warmup_steps 500 \
#     --save_steps -1 --learning_rate 1e-5 \
#     --adam_epsilon 1e-6 --seed 15213 
#     # For min-overlap --train_name _min-overlap --max_steps 8424
#     # For random --train_name _random17K --max_steps 8424
