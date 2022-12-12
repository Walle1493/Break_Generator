# CUDA_VISIBLE_DEVICES=2 python ../evaluate.py \
#     --model_type=t5 \
#     --tokenizer_name=t5-base \
#     --model_name_or_path=/home/mxdong/Model/Decomposition/MuSiQue/T5-Base \
#     --data_dir=/home/mxdong/Data/MuSiQue/format_data/dev.json \
#     --max_src_len 70\
#     --max_tgt_len 100\
#     --seed 42 \

CUDA_VISIBLE_DEVICES=3 python ../evaluate.py \
    --model_type=t5 \
    --tokenizer_name=t5-large \
    --model_name_or_path=/home/mxdong/Model/Decomposition/MuSiQue/T5-Large \
    --data_dir=/home/mxdong/Data/MuSiQue/format_data/dev.json \
    --max_src_len 70\
    --max_tgt_len 100\
    --seed 42 \
