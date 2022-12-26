# export DATA_DIR=/home/mxdong/Data/MuSiQue/multi_choice_short
export DATA_DIR=/home/mxdong/Data/MuSiQue/multi_choice_data
# export DATA_DIR=/home/mxdong/Data/MuSiQue/multi_choice_title

export TASK_NAME=MuSiQue
# export TASK_NAME=MuSiQue_Title
export MODEL_NAME=bert-base-uncased
export OUTPUT_DIR=/home/mxdong/Model/Selector/${TASK_NAME}/${MODEL_NAME}


# Bert-Base
CUDA_VISIBLE_DEVICES=0 python ../run_musique.py \
    --model_type bert \
    --model_name_or_path ${MODEL_NAME} \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --train_file train.json \
    --predict_file dev.json \
    --max_seq_length 512 \
    --max_query_length 32 \
    --per_gpu_train_batch_size 8   \
    --per_gpu_eval_batch_size 8   \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --num_train_epochs 3.0 \
    --logging_steps 2000 \
    --save_steps 4000 \
    --adam_epsilon 1e-8 \
    --warmup_steps 300 \
    --overwrite_output_dir \
    --evaluate_during_training \
