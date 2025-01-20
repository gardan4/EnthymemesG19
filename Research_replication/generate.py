import torch
from fairseq.models.bart import BARTModel
import os
import time
import numpy as np


def setup_device():
    """
    Set up the device for computation. Use GPU if available.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("GPU not available, using CPU.")
    return device


def load_bart_model(cpdir, checkpoint_file, datadir, device):
    """
    Load the BART model from the specified checkpoint.

    Args:
        cpdir (str): Directory containing the model checkpoint.
        checkpoint_file (str): Filename of the checkpoint.
        datadir (str): Path to the data directory.
        device (torch.device): Device to load the model on.

    Returns:
        BARTModel: Loaded BART model.
    """
    try:
        bart = BARTModel.from_pretrained(
            cpdir,
            checkpoint_file=checkpoint_file,
            data_name_or_path=datadir
        )
        bart.to(device)
        bart.eval()
        print(f"Loaded model from {cpdir}/{checkpoint_file} on {device}.")
        return bart
    except Exception as e:
        print(f"Error loading model from {cpdir}/{checkpoint_file}: {e}")
        return None


def generate_hypotheses(bart, source_texts, output_path, bsz=32, maxb=200, minb=7):
    """
    Generate hypotheses for the given source texts using the BART model.

    Args:
        bart (BARTModel): Loaded BART model.
        source_texts (list): List of source sentences.
        output_path (str): Path to save the generated hypotheses.
        bsz (int): Batch size.
        maxb (int): Maximum length of generated sequence.
        minb (int): Minimum length of generated sequence.
    """
    with open(output_path, 'w', encoding='utf-8') as fout:
        slines = []
        total = len(source_texts)
        for idx, sline in enumerate(source_texts, 1):
            slines.append(sline.strip())
            if len(slines) == bsz:
                print(
                    f"Processing batch {idx // bsz} ({len(slines)} samples) on device: {next(bart.parameters()).device}")
                with torch.no_grad():
                    hypotheses_batch = bart.sample(
                        slines,
                        beam=5,
                        lenpen=2.0,
                        max_len_b=maxb,
                        min_len_b=minb,
                        no_repeat_ngram_size=3,
                        verbose=False
                    )
                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis.replace('\n', '') + '\n')
                slines = []
        # Process any remaining lines
        if slines:
            print(f"Processing final batch ({len(slines)} samples) on device: {next(bart.parameters()).device}")
            with torch.no_grad():
                hypotheses_batch = bart.sample(
                    slines,
                    beam=5,
                    lenpen=2.0,
                    max_len_b=maxb,
                    min_len_b=minb,
                    no_repeat_ngram_size=3,
                    verbose=False
                )
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis.replace('\n', '') + '\n')
    print(f"Hypotheses saved to {output_path}.")


def main():
    # Set random seeds for reproducibility
    np.random.seed(4)
    torch.manual_seed(4)

    # Setup device
    device = setup_device()

    # Define models and source files
    # Each model is a tuple of (cpdir, checkpoint_file)
    models = [
        ("bart_checkpoint_enthymemes_paracomet", "checkpoint50.pt")
    ]

    # List of source files
    sourcefiles = [
        "./D1test/semevaldata.source",
        "./D1test/semevaldataparacomet.source",
        "./D2test/jandata.source",
        "./D2test/jandataparacomet.source",
        "./D3test/ikatdata.source",
        "./D3test/ikatdataparacomet.source"
    ]

    # Data directory required by Fairseq
    datadir = 'enthymemes_data'

    # Output directory for hypotheses
    output_dir = 'generated_hypotheses'
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each model
    for cpdir, checkpoint_file in models:
        # Load the model
        bart = load_bart_model(cpdir, checkpoint_file, datadir, device)
        if bart is None:
            print(f"Skipping model {cpdir}/{checkpoint_file} due to loading error.")
            continue  # Skip to the next model if loading failed

        # Iterate over each source file
        for sourcefile in sourcefiles:
            if not os.path.exists(sourcefile):
                print(f"Source file {sourcefile} does not exist. Skipping.")
                continue  # Skip if source file does not exist

            # Determine the base name for output file
            basename = os.path.basename(sourcefile)
            name, ext = os.path.splitext(basename)

            # Construct a unique identifier for the model
            model_id = os.path.splitext(checkpoint_file)[0]  # e.g., 'checkpoint50'

            # Define the output hypothesis file path
            output_filename = f"{name}_{model_id}.hypo"
            output_path = os.path.join(output_dir, output_filename)

            # Load source texts
            with open(sourcefile, 'r', encoding='utf-8') as f:
                source_texts = [line.strip() for line in f.readlines()]

            print(f"\nGenerating hypotheses for source file '{sourcefile}' using model '{cpdir}/{checkpoint_file}'.")
            start_time = time.time()

            # Generate and save hypotheses
            generate_hypotheses(bart, source_texts, output_path, bsz=32, maxb=200, minb=7)

            end_time = time.time()
            elapsed = end_time - start_time
            print(f"Completed in {elapsed:.2f} seconds.")

        # Optionally, free up GPU memory by deleting the model
        del bart
        torch.cuda.empty_cache()
        print(f"Released memory for model {cpdir}/{checkpoint_file}.")


if __name__ == "__main__":
    main()
