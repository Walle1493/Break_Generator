# CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
#     --model_type=bart \
#     --tokenizer_name=facebook/bart-base \
#     --model_name_or_path=/home/mxdong/Model/Decomposition/MuSiQue/Bart-Base \
#     --data_dir=/home/mxdong/Data/MuSiQue/format_data/dev.json \
#     --max_src_len 70\
#     --max_tgt_len 100\
#     --seed 42 \

CUDA_VISIBLE_DEVICES=1 python ../evaluate.py \
    --model_type=bart \
    --tokenizer_name=facebook/bart-large \
    --model_name_or_path=/home/mxdong/Model/Decomposition/MuSiQue/Bart-Large \
    --data_dir=/home/mxdong/Data/MuSiQue/format_data/dev.json \
    --max_src_len 70\
    --max_tgt_len 100\
    --seed 42 \
