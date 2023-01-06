import sys
import os
import json
import argparse


def q_s(src_path, seq_dir, para_dir, mode):
    """Binary Classification"""

    with open(src_path) as f:
        dataset = json.load(f)
    
    seq_sources = []
    seq_targets = []
    para_sources = []
    para_targets = []
    
    for data in dataset:
        _id = data["id"]
        _type = _id.split("_")[0]
        question = data["question"]
        decompositions = data["question_decomposition"]
        
        if _type in ["2hop", "3hop1", "4hop1"]:
            for i, decomp in enumerate(decompositions):
                if i == 0:
                    source = question
                    target = decomp["question"]
                elif i < len(decompositions) - 1:
                    source += " ; " + target + " " + decompositions[i-1]["answer"]
                    target = decomp["question"]
                else:
                    source += " ; " + target + " " + decompositions[i-1]["answer"]
                    target = decomp["question"] + " </s> </s>"
                if "#" in target:
                    pos = -1
                    pos = target.find("#", pos + 1)
                    while pos != -1:
                        num = int(target[pos + 1])
                        if num < 5:
                            target = target.replace(target[pos:pos+2], decompositions[num - 1]["answer"])
                        pos += 1
                        pos = target.find("#", pos + 1)
                seq_sources.append(source)
                seq_targets.append(target)
        elif _type in ["3hop2", "4hop2"]:
            for i in range(len(decompositions)):
                if i == 0:
                    source = question
                    target = decompositions[i]["question"] + " ; " + decompositions[i+1]["question"]
                elif i == 1:
                    continue
                elif i == 2:
                    source += " ; " + decompositions[i-2]["question"] + " " + decompositions[i-2]["answer"] + \
                        " ; " + decompositions[i-1]["question"] + " " + decompositions[i-1]["answer"]
                    target = decompositions[i]["question"]
                else:
                    source += " ; " + target + " " + decompositions[i-1]["answer"]
                    target = decompositions[i]["question"] + " </s> </s>"
                if "#" in target:
                    pos = -1
                    pos = target.find("#", pos + 1)
                    while pos != -1:
                        num = int(target[pos + 1])
                        if num < 5:
                            target = target.replace(target[pos:pos+2], decompositions[num - 1]["answer"])
                        pos += 1
                        pos = target.find("#", pos + 1)
                if i == 0:
                    para_sources.append(source)
                    para_targets.append(target)
                else:
                    seq_sources.append(source)
                    seq_targets.append(target)
        else:
            for i in range(len(decompositions)):
                if i == 0:
                    source = question
                    target = decompositions[i]["question"] + " ; " + decompositions[i+2]["question"]
                elif i == 1:
                    source += " ; " + decompositions[i-1]["question"] + " " + decompositions[i-1]["answer"] + \
                        " ; " + decompositions[i+1]["question"] + " " + decompositions[i+1]["answer"]
                    target = decompositions[i]["question"]
                elif i == 2:
                    continue
                else:
                    source += " ; " + target + " " + decompositions[i-2]["answer"]
                    target = decompositions[i]["question"] + " </s> </s>"
                if "#" in target:
                    pos = -1
                    pos = target.find("#", pos + 1)
                    while pos != -1:
                        num = int(target[pos + 1])
                        if num < 5:
                            target = target.replace(target[pos:pos+2], decompositions[num - 1]["answer"])
                        pos += 1
                        pos = target.find("#", pos + 1)
                if i == 0:
                    para_sources.append(source)
                    para_targets.append(target)
                else:
                    seq_sources.append(source)
                    seq_targets.append(target)
    
    assert len(seq_sources) == len(seq_targets)
    assert len(para_sources) == len(para_targets)
    print("#sequence_data: {}".format(len(seq_sources)))
    print("#parallel_data: {}".format(len(para_sources)))

    with open(os.path.join(seq_dir, mode + ".source"), "w") as f:
        for source in seq_sources:
            f.write(source + '\n')
    
    with open(os.path.join(seq_dir, mode + ".target"), "w") as f:
        for target in seq_targets:
            f.write(target + '\n')
    
    with open(os.path.join(para_dir, mode + ".source"), "w") as f:
        for source in para_sources:
            f.write(source + '\n')
    
    with open(os.path.join(para_dir, mode + ".target"), "w") as f:
        for target in para_targets:
            f.write(target + '\n')

    return 


if __name__ == "__main__":

    mode = sys.argv[1]  # train or dev
    assert mode in ["train", "dev"]

    src_dir = "/home/mxdong/Data/MuSiQue/format_data"
    seq_dir = "/home/mxdong/Data/MuSiQue/class_data/sequence_data"
    para_dir = "/home/mxdong/Data/MuSiQue/class_data/parallel_data"
    if not os.path.exists(seq_dir):
        os.mkdir(seq_dir)
    if not os.path.exists(para_dir):
        os.mkdir(para_dir)

    src_path = os.path.join(src_dir, mode + ".json")
    q_s(src_path, seq_dir, para_dir, mode)
