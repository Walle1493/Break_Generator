import sys
import os
import json


def multi_step(src_path):

    with open(src_path) as f:
        data = json.load(f)
    
    sources = []
    targets = []
    
    for item in data:
        question = item["question"]
        decompositions = item["question_decomposition"]
        
        # sources.append(question)
        # target = ""
        for i, decomp in enumerate(decompositions):
            if i == 0:
                source = question
                target = decomp["question"]
            elif i < len(decompositions) - 1:
                # BART: </s> for sep and eos
                source += " </s> " + target + " " + decompositions[i-1]["answer"]
                target = decomp["question"]
            else:
                source += " </s> " + target + " " + decompositions[i-1]["answer"]
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
            sources.append(source)
            targets.append(target)
    
    assert len(sources) == len(targets)
    return sources, targets


def save_file(source_path, dest_path, sources, targets):

    with open(source_path, "w") as f:
        for source in sources:
            f.write(source + '\n')
    
    with open(dest_path, "w") as f:
        for target in targets:
            f.write(target + '\n')


if __name__ == "__main__":

    mode = sys.argv[1]  # train or dev
    assert mode in ["train", "dev"]

    src_path = "/home/mxdong/Data/MuSiQue/format_data"
    dest_path = "/home/mxdong/Data/MuSiQue/multi_step_data"
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)

    src_path = os.path.join(src_path, mode + ".json")
    source_dest_path = os.path.join(dest_path, mode + ".source")
    target_dest_path = os.path.join(dest_path, mode + ".target")

    sources, targets = multi_step(src_path)
    save_file(source_dest_path, target_dest_path, sources, targets)
