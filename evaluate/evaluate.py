import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import BertTokenizer
import argparse
import json
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score, single_meteor_score
from rouge import Rouge
import logging
import pdb


logger = logging.getLogger(__name__)

# BLEU
def _bleu_score(reference, hypothesis, smooth=None):
    # Pre-process
    references = [word_tokenize(reference)]
    hypothesis = word_tokenize(hypothesis)

    # (BLEU-1 + BLUE-2 + BLUE-3 + BLUE-4) / 4
    score = sentence_bleu(references, hypothesis, smoothing_function=smooth.method1)

    return score


# METEOR
def _meteor_score(reference, hypothesis):
    # Pre-process
    references = [word_tokenize(reference)]
    hypothesis = word_tokenize(hypothesis)

    score = meteor_score(references, hypothesis)

    return score


# ROUGE
def _rouge_score(reference, hypothesis, rouger=None):
    # ROUGE-2 F
    # rouger = Rouge()
    scores = rouger.get_scores(hypothesis, reference)
    score = scores[0]["rouge-2"]["f"]
    return score


def eval(args):

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")

    # tokenizer and model
    if args.model_type == "t5":
        tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name)
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    elif args.model_type == "bart":
        tokenizer = BartTokenizer.from_pretrained(args.tokenizer_name)
        model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model.to(device)

    # generate simple questions
    def get_decomposition(question):
        if args.model_type == "t5":
            input_text = "Paraphrase: " + question
        else:
            input_text = question
        features = tokenizer([input_text], return_tensors='pt').to(device)
        output = model.generate(input_ids=features['input_ids'], 
                attention_mask=features['attention_mask'],
                max_length=args.max_tgt_len)
        return tokenizer.decode(output[0])

    with open(args.data_dir) as f:
        dataset = json.load(f)
    
    # Pre-process metrics
    smooth = SmoothingFunction()
    rouger = Rouge()

    # calculate metrics
    bleu, meteor, rouge = 0.0, 0.0, 0.0
    # 2,3,4-hop metrics
    metrics = [[0.0 for _ in range(3)] for _ in range(3)]
    counts = [0 for _ in range(3)]
    for i, data in enumerate(dataset):
        # 2,3,4-hop counts
        hop = len(data["question_decomposition"])
        counts[hop - 2] += 1
        # 2,3,4-hop counts END
        question = data["question"]
        label = ""
        for j, decomp in enumerate(data["question_decomposition"]):
            if j == 0:
                label += decomp["question"]
            else:
                label += " ; " + decomp["question"]
        prediction = get_decomposition(question)
        # Plain Text Process
        prediction = prediction.replace("<s>", "").replace("</s>", "").replace("<pad>", "")
        _bleu = _bleu_score(label, prediction, smooth=smooth)
        _meteor = _meteor_score(label, prediction)
        _rouge = _rouge_score(label, prediction, rouger=rouger)
        bleu += _bleu
        meteor += _meteor
        rouge += _rouge
        # 2,3,4-hop metrics
        metrics[hop - 2][0] += _bleu
        metrics[hop - 2][1] += _meteor
        metrics[hop - 2][2] += _rouge
        # 2,3,4-hop metrics END
        # # Log
        # logger.info("***** Case No.%d *****", (i + 1))
        # logger.info("BLEU Score: %f", bleu)
        # logger.info("METEOR Score: %f", meteor)
        # logger.info("ROUGE Score: %f", rouge)
        # Print
        print("***** Case No.%d *****" % (i + 1))
        print("BLEU Score: %f" % _bleu)
        print("METEOR Score: %f" % _meteor)
        print("ROUGE Score: %f" % _rouge)

    bleu = bleu / len(dataset)
    meteor = meteor / len(dataset)
    rouge = rouge / len(dataset)

    for i in range(3):
        for j in range(3):
            metrics[i][j] /= counts[i]

    return bleu, meteor, rouge, metrics


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir.")

    ## Other parameters
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--max_src_len", default=70, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--max_tgt_len", default=100, type=int,
                        help="Optional target sequence length after tokenization.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    
    args = parser.parse_args()
    result = eval(args)
    
    bleu, meteor, rouge, metrics = result

    # logger.info("***** Final Result *****")
    # logger.info("Final BLEU: %f", bleu)
    # logger.info("Final METEOR: %f", meteor)
    # logger.info("Final ROUGE: %f", rouge)

    print("***** Final Result *****")
    print("Final BLEU: %f" % bleu)
    print("Final METEOR: %f" % meteor)
    print("Final ROUGE: %f" % rouge)

    print("***** 2-Hop Result *****")
    print("2-Hop BLEU: %f" % metrics[0][0])
    print("2-Hop METEOR: %f" % metrics[0][1])
    print("2-Hop ROUGE: %f" % metrics[0][2])

    print("***** 3-Hop Result *****")
    print("3-Hop BLEU: %f" % metrics[1][0])
    print("3-Hop METEOR: %f" % metrics[1][1])
    print("3-Hop ROUGE: %f" % metrics[1][2])

    print("***** 4-Hop Result *****")
    print("4-Hop BLEU: %f" % metrics[2][0])
    print("4-Hop METEOR: %f" % metrics[2][1])
    print("4-Hop ROUGE: %f" % metrics[2][2])


if __name__ == "__main__":
    main()
