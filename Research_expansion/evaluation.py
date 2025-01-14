import torch
import evaluate
import os
import csv
from tqdm import tqdm


def load_texts(file_path, source_file_path):
    """
    Load texts from a file and return as a list of stripped strings.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        # Also read source file to get the source text
        with open(source_file_path, 'r', encoding='utf-8') as f_source:
            line = [line.strip() for line in f.readlines()]
            line_source = [line.strip() for line in f_source.readlines()]
            # The source line contains a #; save the content before and after the # in two variables
            line_source = [line.split('#') for line in line_source]
            # Remove both elements of the list line_source from the line of the hypothesis
            for i in range(len(line_source)):
                line[i] = line[i].replace(line_source[i][0], '')
                line[i] = line[i].replace(line_source[i][1], '')
        return line


def main():
    # Define explicit pairs of hypothesis and target files
    # Format: (hypothesis_file_path, target_file_path, source_file_path)
    file_pairs = [
        ("./semevaldata.hypo", "../Datasets/D1test/semevaldata.target", "../Datasets/D1test/semevaldata.source")
    ]

    # Output CSV file to store evaluation results
    output_csv = 'evaluation_results.csv'

    # Initialize metrics
    bleu = evaluate.load('bleu')
    bertscore = evaluate.load('bertscore')
    rouge = evaluate.load('rouge')  # Optional, if ROUGE is desired

    # Prepare CSV header
    csv_header = ['Filename', 'BLEU-1', 'BLEU-2',
                  'ROUGE-1', 'ROUGE-2', 'ROUGE-L',
                  'BERTScore Precision', 'BERTScore Recall', 'BERTScore F1']

    # Open CSV file for writing
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_header)

        print(f"Starting evaluation of {len(file_pairs)} file pairs.\n")

        # Iterate over each file pair with a progress bar
        for hypo_path, target_path, source_path in tqdm(file_pairs, desc="Evaluating file pairs"):
            # Extract filename from the hypothesis file path
            hypo_filename = os.path.basename(hypo_path)

            # Check if both files exist
            if not os.path.exists(hypo_path):
                print(f"Hypothesis file not found: {hypo_path}. Skipping.")
                continue
            if not os.path.exists(target_path):
                print(f"Target file not found: {target_path}. Skipping.")
                continue

            # Load texts
            generated_hypo_texts = load_texts(hypo_path, source_path)
            target_texts = load_texts(target_path, source_path)

            # Validate that all datasets have the same number of samples
            if len(target_texts) != len(generated_hypo_texts):
                print(f"Line count mismatch between hypothesis and target for {hypo_filename}. Skipping.")
                continue  # Skip if line counts do not match

            # Compute BLEU-1
            bleu1 = bleu.compute(
                predictions=generated_hypo_texts,
                references=[[ref] for ref in target_texts],
                max_order=1
            )

            # Compute BLEU-2
            bleu2 = bleu.compute(
                predictions=generated_hypo_texts,
                references=[[ref] for ref in target_texts],
                max_order=2
            )

            # Compute ROUGE
            ro = rouge.compute(
                predictions=generated_hypo_texts,
                references=target_texts,
                use_stemmer=True
            )

            # Compute BERTScore
            bs = bertscore.compute(
                predictions=generated_hypo_texts,
                references=target_texts,
                lang='en',
                model_type='bert-base-uncased',
                verbose=False
            )

            # Calculate average BERTScore metrics
            bert_precision = sum(bs['precision']) / len(bs['precision']) * 100
            bert_recall = sum(bs['recall']) / len(bs['recall']) * 100
            bert_f1 = sum(bs['f1']) / len(bs['f1']) * 100

            # Write results to CSV
            writer.writerow([
                hypo_filename,
                f"{bleu1['bleu'] * 100:.2f}",
                f"{bleu2['bleu'] * 100:.2f}",
                f"{ro['rouge1'] * 100:.2f}",
                f"{ro['rouge2'] * 100:.2f}",
                f"{ro['rougeL'] * 100:.2f}",
                f"{bert_precision:.2f}",
                f"{bert_recall:.2f}",
                f"{bert_f1:.2f}"
            ])

            # Print summary for this pair
            print(f"Evaluated {hypo_filename}: BLEU-1={bleu1['bleu'] * 100:.2f}, BLEU-2={bleu2['bleu'] * 100:.2f}, "
                  f"ROUGE-1={ro['rouge1'] * 100:.2f}, ROUGE-2={ro['rouge2'] * 100:.2f}, ROUGE-L={ro['rougeL'] * 100:.2f}, "
                  f"BERTScore F1={bert_f1:.2f}")

        print(f"\nEvaluation completed. Results saved to {output_csv}.")


# Execute the evaluation
main()
