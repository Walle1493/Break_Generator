#!/usr/bin/env bash

# bsub -n 1 -q HPC.S1.GPU.X785.suda -o train.T5-single.log -gpu num=4:mode=exclusive_process sh Shell/finetune.sh

    # --model_name_or_path=/SISDC_GPFS/Home_SE/hy-suda/pre-train_model/T5-v1.1-xxl \
# nvidia-smi
# CUDA_VISIBLE_DEVICES=0,1,2,3 python run_fine_tuning.py \
#     --model_type=T5 \
#     --model_name_or_path=/SISDC_GPFS/Home_SE/hy-suda/pre-train_model/T5-3B \
#     --data_dir=/SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/Project-Final/Data/Raw_data/ProtoQA/with_keyword_v1_dpr/ \
#     --do_train --do_eval --evaluate_during_training \
#     --per_gpu_train_batch_size 4 \
#     --per_gpu_eval_batch_size 32 \
#     --gradient_accumulation_steps=2 \
#     --logging_steps 500 \
#     --max_src_len 110\
#     --max_tgt_len 120\
#     --output_dir=/SISDC_GPFS/Home_SE/hy-suda/zfli/Models/Finetuned_model/ProtoQA/T5-3B_rmbad \
#     --num_train_epochs=1 \
#     --warmup_steps 150 \
#     --overwrite_output_dir \
#     --save_steps -1 --learning_rate 1e-5 \
#     --adam_epsilon 1e-6 --seed 6660 \
#     --T5_split 4 \
#     --switch_dataset \
#     --experiment rm_bad
    # --experiment keyword_wkdt_dpr
    # --experiment keyword_wkdt_dpr_multi
    
    # --fp16 \
    # --train_name _similarity --max_steps 5621
    # For full data  remove train_name and max_steps 
    # For non-overlap  --train_name _non-overlap --max_steps 5621

# CUDA_VISIBLE_DEVICES=0 python run_generation.py \
#     --model_type gpt2 \
#     --model_name_or_path outputs/gpt2_finetune_8848 \
#     --length=10 \
#     --num_samples=300 \
#     --temperature=0.69 \
#     --input_file='/SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/Project-Final/Data/Raw_data/ProtoQA/Origin_format/dev.crowdsourced.jsonl' 


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

# CUDA_VISIBLE_DEVICES=0 python ../run_fine_tuning.py \
#     --model_type=T5 \
#     --model_name_or_path=t5-base \
#     --data_dir=/home/mxdong/Data/MuSiQue/seq2seq_data \
#     --do_train --do_eval --evaluate_during_training \
#     --per_gpu_train_batch_size 64 \
#     --per_gpu_eval_batch_size 64 \
#     --gradient_accumulation_steps=1 \
#     --logging_steps 500 \
#     --max_src_len 70 \
#     --max_tgt_len 100 \
#     --output_dir=/home/mxdong/Model/Decomposition/MuSiQue/T5-Base \
#     --num_train_epochs=10 \
#     --warmup_steps 100 \
#     --overwrite_output_dir \
#     --save_steps -1 \
#     --learning_rate 1e-5 \
#     --adam_epsilon 1e-8 \
#     --seed 42 \
#     --T5_split -1 \

CUDA_VISIBLE_DEVICES=1 python ../run_fine_tuning.py \
    --model_type=T5 \
    --model_name_or_path=t5-large \
    --data_dir=/home/mxdong/Data/MuSiQue/seq2seq_data \
    --do_train --do_eval --evaluate_during_training \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 16 \
    --gradient_accumulation_steps=1 \
    --logging_steps 500 \
    --max_src_len 70\
    --max_tgt_len 100\
    --output_dir=/home/mxdong/Model/Decomposition/MuSiQue/T5-Large \
    --num_train_epochs=3 \
    --warmup_steps 100 \
    --overwrite_output_dir \
    --save_steps -1 \
    --learning_rate 1e-5 \
    --adam_epsilon 1e-8 \
    --seed 42 \
    --T5_split -1 \

# CUDA_VISIBLE_DEVICES=2 python ../run_fine_tuning.py \
#     --model_type=T5 \
#     --model_name_or_path=t5-3b \
#     --data_dir=/home/mxdong/Data/MuSiQue/seq2seq_data_ \
#     --do_train --do_eval --evaluate_during_training \
#     --per_gpu_train_batch_size 1 \
#     --per_gpu_eval_batch_size 1 \
#     --gradient_accumulation_steps=1 \
#     --logging_steps 500 \
#     --max_src_len 70\
#     --max_tgt_len 100\
#     --output_dir=/home/mxdong/Model/Decomposition/MuSiQue/T5-3B \
#     --num_train_epochs=4 \
#     --warmup_steps 150 \
#     --overwrite_output_dir \
#     --save_steps -1 \
#     --learning_rate 1e-5 \
#     --adam_epsilon 1e-6 \
#     --seed 6660 \
#     --T5_split 2 \
