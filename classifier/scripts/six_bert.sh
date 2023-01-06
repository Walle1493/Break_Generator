# export DATA_DIR=/home/mxdong/Data/MuSiQue/short_data
export DATA_DIR=/home/mxdong/Data/MuSiQue/format_data

# export TASK_NAME=Binary
export TASK_NAME=Six
export MODEL_NAME=bert-large-uncased
export OUTPUT_DIR=/home/mxdong/Model/Classification/${TASK_NAME}/${MODEL_NAME}


# # Bert-Large & Binary Classification
# CUDA_VISIBLE_DEVICES=0 python ../run_classifier.py \
#     --model_type bert \
#     --model_name_or_path ${MODEL_NAME} \
#     --do_train \
#     --do_eval \
#     --do_lower_case \
#     --data_dir ${DATA_DIR} \
#     --output_dir ${OUTPUT_DIR} \
#     --train_file train.json \
#     --predict_file dev.json \
#     --max_seq_length 32 \
#     --per_gpu_train_batch_size 32   \
#     --per_gpu_eval_batch_size 32   \
#     --gradient_accumulation_steps 1 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 6.0 \
#     --logging_steps 300 \
#     --save_steps 600 \
#     --adam_epsilon 1e-8 \
#     --warmup_steps 300 \
#     --overwrite_output_dir \
#     --evaluate_during_training \


# # Bert-Large & Six Classification
CUDA_VISIBLE_DEVICES=0 python ../run_classifier.py \
    --model_type bert \
    --model_name_or_path ${MODEL_NAME} \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --train_file train.json \
    --predict_file dev.json \
    --max_seq_length 33 \
    --per_gpu_train_batch_size 32   \
    --per_gpu_eval_batch_size 32   \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --logging_steps 300 \
    --save_steps 600 \
    --adam_epsilon 1e-8 \
    --warmup_steps 300 \
    --overwrite_output_dir \
    --evaluate_during_training \
    --six_classification

