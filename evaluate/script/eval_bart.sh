# CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
#     --model_type=bart \
#     --tokenizer_name=facebook/bart-base \
#     --model_name_or_path=/home/mxdong/Model/Decomposition/MuSiQue/Bart-Base \
#     --data_dir=/home/mxdong/Data/MuSiQue/format_data/dev.json \
#     --max_src_len 70\
#     --max_tgt_len 100\
#     --seed 42 \

# # Seq2seq generator
# CUDA_VISIBLE_DEVICES=3 python ../evaluate.py \
#     --model_type=bart \
#     --tokenizer_name=facebook/bart-large \
#     --model_name_or_path=/home/mxdong/Model/Seq2seq/MuSiQue/Bart-Large \
#     --data_dir=/home/mxdong/Data/MuSiQue/format_data/dev.json \
#     --switch=seq2 \
#     --max_src_len 70\
#     --max_tgt_len 100\
#     --seed 42 \

# Multi-step generator
CUDA_VISIBLE_DEVICES=3 python ../evaluate.py \
    --model_type=bart \
    --tokenizer_name=facebook/bart-large \
    --model_name_or_path=/home/mxdong/Model/MultiStep/MuSiQue/Bart-Large \
    --data_dir=/home/mxdong/Data/MuSiQue/format_data/dev.json \
    --switch=multi \
    --max_src_len 160\
    --max_tgt_len 48\
    --seed 42 \
