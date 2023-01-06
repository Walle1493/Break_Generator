import sys
import os
import json
import argparse


def merge(seq_dir, para_dir, mix_dir, mode):
    """merge sequence data and parallel data"""

    seq_src_path = os.path.join(seq_dir, mode + ".source")
    seq_tgt_path = os.path.join(seq_dir, mode + ".target")
    para_src_path = os.path.join(para_dir, mode + ".source")
    para_tgt_path = os.path.join(para_dir, mode + ".target")

    with open(seq_src_path) as f:
        seq_src = f.readlines()
    with open(seq_tgt_path) as f:
        seq_tgt = f.readlines()
    with open(para_src_path) as f:
        para_src = f.readlines()
    with open(para_tgt_path) as f:
        para_tgt = f.readlines()
    
    seq_src.extend(para_src)
    seq_tgt.extend(para_tgt)

    mix_src_path = os.path.join(mix_dir, mode + ".source")
    mix_tgt_path = os.path.join(mix_dir, mode + ".target")

    with open(mix_src_path, "w") as f:
        for line in seq_src:
            f.write(line)
    with open(mix_tgt_path, "w") as f:
        for line in seq_tgt:
            f.write(line)
    
    return 


if __name__ == "__main__":

    mode = sys.argv[1]  # train or dev
    assert mode in ["train", "dev"]

    seq_dir = "/home/mxdong/Data/MuSiQue/class_data/sequence_data"
    para_dir = "/home/mxdong/Data/MuSiQue/class_data/parallel_data"
    mix_dir = "/home/mxdong/Data/MuSiQue/class_data/mix_data"
    if not os.path.exists(mix_dir):
        os.mkdir(mix_dir)

    merge(seq_dir, para_dir, mix_dir, mode)
