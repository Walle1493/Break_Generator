#!/usr/bin/env bash

# bsub -n 1 -q HPC.S1.GPU.X785.suda -o train.bart.log -gpu num=1:mode=exclusive_process sh Shell/finetune_bart.sh
    # --data_dir=/SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/Project-Final/Data/Raw_data/ProtoQA/Origin_format \
    # --model_name_or_path=/SISDC_GPFS/Home_SE/hy-suda/pre-train_model/T5-v1.1-xxl \
    # --data_dir=/SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/Project-Final/Data/Constructed_data/ProtoQA/Prefix_knowledge/v2/ \

seed=42
output_dir=/home/mxdong/Model/MultiStep/MuSiQue/Bart-Large
# data_dir=/SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/Project-Final/Data/Constructed_data/ProtoQA/Prefix_knowledge/v2/
data_dir=/home/mxdong/Data/MuSiQue/multi_step_data
# data_dir=/SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/Project-Final/Data/Constructed_data/ProtoQA/Few-shot_description_GPT-3/protoqa_v3_curie/fewshot_description_v1
# data_dir=/SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/Project-Final/Data/Constructed_data/ProtoQA/Few-shot_knowledge/protoqa_v3_curie/fewshotv1/

# echo $output_dir

# # bart base
# nvidia-smi
# CUDA_VISIBLE_DEVICES=2 python ../run_fine_tuning.py \
#     --model_type=bart \
#     --model_name_or_path=facebook/bart-base \
#     --data_dir $data_dir \
#     --do_train --do_eval --evaluate_during_training \
#     --per_gpu_train_batch_size=64 \
#     --per_gpu_eval_batch_size=64 \
#     --gradient_accumulation_steps=1 \
#     --logging_steps 500 \
#     --max_src_len 70\
#     --max_tgt_len 100\
#     --output_dir $output_dir\
#     --num_train_epochs=4 \
#     --warmup_steps 100 \
#     --overwrite_output_dir \
#     --save_steps -1 --learning_rate 1e-5 \
#     --adam_epsilon 1e-8 --seed $seed \
#     # --switch_dataset \
#     # --experiment knowledge

# bart large
nvidia-smi
CUDA_VISIBLE_DEVICES=3 python ../run_fine_tuning.py \
    --model_type=bart \
    --model_name_or_path=facebook/bart-large \
    --data_dir $data_dir \
    --do_train --do_eval --evaluate_during_training \
    --per_gpu_train_batch_size=32 \
    --per_gpu_eval_batch_size=32 \
    --gradient_accumulation_steps=1 \
    --logging_steps 500 \
    --max_src_len 160\
    --max_tgt_len 48\
    --output_dir $output_dir\
    --num_train_epochs=10 \
    --warmup_steps 100 \
    --overwrite_output_dir \
    --save_steps -1 --learning_rate 1e-5 \
    --adam_epsilon 1e-8 --seed $seed \


# CUDA_VISIBLE_DEVICES=0 python run_generation.py \
#     --model_type bart \
#     --model_name_or_path $output_dir \
#     --tokenizer_name_or_path /SISDC_GPFS/Home_SE/hy-suda/zfli/Models/init_model/bart-large\
#     --input_file=$data_dir/dev.crowdsourced.jsonl \
#     --length=15 \
#     --num_samples=15 \
#     --temperature=0.69 \
#     --output_name dev-result.jsonl \
#     --switch_dataset \
#     --disable_mostcommon \
#     --experiment knowledge

# nohup sh evaluate_result.sh $output_dir