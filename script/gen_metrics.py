# Three METRICS: BLEU, METEOR, ROUGE for Generating

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score, single_meteor_score
from rouge import Rouge 
# TODO: Use nltk/gensim to calculate Perplexity


# BLEU
def _bleu_score(references, hypothesis):
    # (BLEU-1 + BLUE-2 + BLUE-3 + BLUE-4) / 4
    score = sentence_bleu(references, hypothesis)
    # (BLUE-1 + BLUE-2 + BLUE-3) / 3
    # score = sentence_bleu(references, hypothesis, weights=(1./3, 1./3, 1./3))
    # (BLEU-1 + BLUE-2) / 2
    # score = sentence_bleu(references, hypothesis, weights=(1./2, 1./2))
    return score


# METEOR
def _meteor_score(references, hypothesis):
    score = meteor_score(references, hypothesis)
    return score


# ROUGE
def _rouge_score(reference, hypothesis):
    rouger = Rouge()
    scores = rouger.get_scores(hypothesis, reference)
    # ROUGE-2 F
    score = scores[0]["rouge-2"]["f"]
    return score


def main():
    
    # Input
    reference = "who does the voice of stan on #1"
    bart_hypothesis = "who does the voice of stan on the episode the hobbit"
    t5_hypothesis = "who does the voice of stan on the show the hobbit"
    gold_hypothesis = "who does the voice of stan on #1"

    # # Pre-process
    # references = [word_tokenize(reference)]
    # bart_hypothesis = word_tokenize(bart_hypothesis)
    # t5_hypothesis = word_tokenize(t5_hypothesis)
    # gold_hypothesis = word_tokenize(gold_hypothesis)

    # Choose metrics
    option = "BLEU"
    if option == "BLEU":
        # Pre-process
        references = [word_tokenize(reference)]
        bart_hypothesis = word_tokenize(bart_hypothesis)
        t5_hypothesis = word_tokenize(t5_hypothesis)
        gold_hypothesis = word_tokenize(gold_hypothesis)
        # Function
        bart_score = _bleu_score(references, bart_hypothesis)
        t5_score = _bleu_score(references, t5_hypothesis)
        gold_score = _bleu_score(references, gold_hypothesis)
        # Result
        # 0.5706745777055999
        # 0.5706745777055999
        # 1.0
    elif option == "METEOR":
        # Pre-process
        references = [word_tokenize(reference)]
        bart_hypothesis = word_tokenize(bart_hypothesis)
        t5_hypothesis = word_tokenize(t5_hypothesis)
        gold_hypothesis = word_tokenize(gold_hypothesis)
        # Function
        bart_score = _meteor_score(references, bart_hypothesis)
        t5_score = _meteor_score(references, t5_hypothesis)
        gold_score = _meteor_score(references, gold_hypothesis)
        # Result
        # 0.7309228039041704
        # 0.7309228039041704
        # 0.9993141289437586
    elif option == "ROUGE":
        # Function
        bart_score = _rouge_score(reference, bart_hypothesis)
        t5_score = _rouge_score(reference, t5_hypothesis)
        gold_score = _rouge_score(reference, gold_hypothesis)
        # Result
        # 0.7058823480968859
        # 0.7058823480968859
        # 0.999999995

    print(bart_score)
    print(t5_score)
    print(gold_score)


if __name__ == "__main__":
    main()
