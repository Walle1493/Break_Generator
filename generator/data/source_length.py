import sys
import os
from transformers import BertTokenizer
import numpy as np


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def cal_length(path):

    length_list = []

    with open(path) as f:
        for data in f.readlines():
            data = data.strip()
            tokens = tokenizer.encode(data)
            length = len(tokens) - 2
            length_list.append(length)
    
    return length_list


def main():

    mode = sys.argv[1]  # train or dev
    assert mode in ["train", "dev"]

    source_path = "/home/mxdong/Data/MuSiQue/multi_step_data"
    target_path = "/home/mxdong/Data/MuSiQue/multi_step_data"

    source_path = os.path.join(source_path, mode + ".source")
    target_path = os.path.join(target_path, mode + ".target")

    source_list = cal_length(source_path)
    target_list = cal_length(target_path)

    np.save(mode + ".source", source_list)
    np.save(mode + ".target", target_list)

    print(max(source_list))
    print(max(target_list))


if __name__ == "__main__":
    
    main()
