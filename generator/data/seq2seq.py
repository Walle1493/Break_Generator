import sys
import os
import json


def seq2seq(src_path):

    with open(src_path) as f:
        data = json.load(f)
    
    sources = []
    targets = []
    
    for item in data:
        question = item["question"]
        decompositions = item["question_decomposition"]
        
        sources.append(question)
        target = ""
        for i, decomp in enumerate(decompositions):
            sub_question = decomp["question"]
            # sub_question = replace_sharp(sub_question)
            if i == 0:
                target += sub_question
            else:
                # target += " @@SEP@@ " + sub_question
                target += " </s> " + sub_question
        targets.append(target)
    
    assert len(sources) == len(targets)
    return sources, targets

def replace_sharp(text):
    """#x -> @@x@@"""
    if '#' in text:
        text = text.replace('#1', '@@1@@')
        text = text.replace('#2', '@@2@@')
        text = text.replace('#3', '@@3@@')
        text = text.replace('#4', '@@4@@')
        text = text.replace('#5', '@@5@@')
    return text


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
    dest_path = "/home/mxdong/Data/MuSiQue/seq2seq_data"
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)

    src_path = os.path.join(src_path, mode + ".json")
    source_dest_path = os.path.join(dest_path, mode + ".source")
    target_dest_path = os.path.join(dest_path, mode + ".target")

    sources, targets = seq2seq(src_path)
    save_file(source_dest_path, target_dest_path, sources, targets)
