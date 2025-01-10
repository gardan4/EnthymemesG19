import sys
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score


def load_file(file_path):
 """
 Load lines from a file and return as a list of stripped strings.
 Handles encoding issues by falling back to a different encoding.
 """
 try:
  with open(file_path, 'r', encoding='utf-8') as f:
   return [line.strip() for line in f.readlines()]
 except UnicodeDecodeError:
  print(f"Failed to read {file_path} with UTF-8. Trying ISO-8859-1 encoding.")
  with open(file_path, 'r', encoding='ISO-8859-1') as f:
   return [line.strip() for line in f.readlines()]

def compute_bleu(reference_texts, hypothesis_texts):
    """
    Compute BLEU-1 and BLEU-2 scores.
    """
    smoothing = SmoothingFunction().method1
    bleu1_scores = []
    bleu2_scores = []

    for ref, hypo in zip(reference_texts, hypothesis_texts):
        reference = [ref.split()]
        candidate = hypo.split()

        bleu1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        bleu2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)

        bleu1_scores.append(bleu1)
        bleu2_scores.append(bleu2)

    avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores)
    avg_bleu2 = sum(bleu2_scores) / len(bleu2_scores)

    return avg_bleu1, avg_bleu2

def compute_bertscore(reference_texts, hypothesis_texts):
    """
    Compute BERTScore.
    """
    P, R, F1 = score(hypothesis_texts, reference_texts, lang="en", rescale_with_baseline=True)
    avg_bs_f1 = F1.mean().item()
    return avg_bs_f1

def main():
    reference_file = "../Datasets/D1test/semevaldata.target"
    hypothesis_file = "./semevaldata.hypo"

    # Load files
    reference_texts = load_file(reference_file)
    hypothesis_texts = load_file(hypothesis_file)

    if len(reference_texts) != len(hypothesis_texts):
        raise ValueError("The number of lines in the reference and hypothesis files must be the same.")

    # Compute BLEU scores
    bleu1, bleu2 = compute_bleu(reference_texts, hypothesis_texts)

    # Compute BERTScore
    bertscore_f1 = compute_bertscore(reference_texts, hypothesis_texts)

    # Print results
    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-2: {bleu2:.4f}")
    print(f"BERTScore (F1): {bertscore_f1:.4f}")

if __name__ == "__main__":
    main()
