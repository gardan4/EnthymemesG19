#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_enthymemes.py

Generates enthymeme expansions from a source file using multiple models
and multiple prompts, writing each combination to its own .hypo file.

Usage:
    python generate_enthymemes.py

Dependencies:
    pip install transformers accelerate bitsandbytes
"""

import os
import sys
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

###############################################################################
#                             CONFIGURATION                                   #
###############################################################################

# Source file containing lines like:
#   The hayride was in October. # to have fun # It was the perfect start to the fall season.
SOURCE_FILE = "Datasets/D1test/semevaldataparacomet.source"

# List of model names or paths
MODEL_LIST = [
    # Example: a non-quantized model that we can quantize with bitsandbytes if desired.
    "O1-OPEN/OpenO1-LLama-8B-v0.1",
]

# Quantization choice: "4bit", "8bit", or "none"
QUANTIZATION_TYPE = "4bit"

# Define multiple prompts:
PROMPTS = {
    "zero_shot": """Rewrite the following sentence to make the hidden reason (enthymeme) explicit,
    inserting a short connecting statement that clarifies the reasoning.
    Output only the final, single-sentence rewrite, with no additional commentary.

    Original: "{sentence}"
    Rewritten:
""",

    "few_shot": """Below is an example of rewriting a sentence to make the hidden reason explicit.

    Example:
    Input: "I forgot my umbrella. # it was raining # I got soaked."
    Rewrite: "I forgot my umbrella, and because it was raining, I ended up getting soaked."

    Now do the same for the following input.
    Output only the final, single-sentence rewrite, with no additional commentary.

    Input: "{sentence}"
    Rewrite:
""",

    "reasoning": """You have a sentence that has an implied reason indicated by '#'.
    Your task is to rewrite the sentence to explicitly state that reason in one concise sentence.
    Output only the final, single-sentence rewrite, with no additional commentary.

    Sentence: "{sentence}"
    Rewrite:
"""
}

# Generation hyperparams (tweak as desired)
GEN_KWARGS = {
    "num_return_sequences": 1,
    "do_sample": True,
    "top_p": 0.9,
    "temperature": 0.8,
}

###############################################################################
#                          HELPER FUNCTIONS                                   #
###############################################################################

def create_bnb_config(quantization_type: str):
    """
    Create a BitsAndBytesConfig object, or return None if quantization_type is "none".
    """
    if quantization_type == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quantization_type == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    else:
        return None

def load_model(model_name: str, quantization_type: str):
    """
    Load model and tokenizer using either bitsandbytes quantization or full float16.
    """
    bnb_config = create_bnb_config(quantization_type)
    if bnb_config is not None:
        print(f"  --> Using bitsandbytes {quantization_type} quantization for {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
    else:
        print(f"  --> Loading model in float16 (no bitsandbytes) for {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer

def generate_raw_text(model, tokenizer, prompt_text, max_new_tokens=1000):
    """
    Basic text generation (returns raw string with all possible extra text).
    """
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, **GEN_KWARGS)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def extract_final_sentence(raw_generation: str):
    """
    Attempt to strip away system prompts or chain-of-thought,
    returning only the final user-facing sentence (the 'answer').

    This can be done in multiple ways:
    1) You might look for a snippet after a known 'Output:' prefix.
    2) Or remove everything after a <Thought> or similar tag.
    3) Or use a regex to isolate the lines between certain markers.

    Below is an example that tries to:
       - Cut off anything after '</output>'
       - Remove any leading instructions before 'Output:' or quotes.

    Adjust logic as needed for your actual text patterns.
    """
    # 1) Optional: cut off chain-of-thought if it appears after '</output>'
    if '</output>' in raw_generation:
        raw_generation = raw_generation.split('</output>')[0]

    # 2) If there's a line with "Output:\n", keep only what's after "Output:" 
    #    (this depends on how your model or prompt is structured).
    #    We'll do a lazy approach: search for the first "Output:" line and take everything after it
    #    except trailing whitespace or quotes.

    output_idx = raw_generation.lower().find("output:")
    if output_idx != -1:
        final_candidate = raw_generation[output_idx + len("output:"):]
        final_candidate = final_candidate.strip()
        # remove any leading quotes or colons
        final_candidate = final_candidate.lstrip(' ":\n')

        # remove any trailing quotes
        final_candidate = final_candidate.rstrip('"')

        return final_candidate.strip()

    # Fallback: if we don't see "Output:" we just return the raw generation
    return raw_generation.strip()

###############################################################################
#                            MAIN SCRIPT                                      #
###############################################################################

def main():
    # Check source file
    if not os.path.exists(SOURCE_FILE):
        print(f"Source file '{SOURCE_FILE}' not found.")
        sys.exit(1)

    # Read all lines from the source
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    # For each model
    for model_name in MODEL_LIST:
        print(f"Loading model: {model_name}")
        try:
            model, tokenizer = load_model(model_name, QUANTIZATION_TYPE)
        except Exception as e:
            print(f"Could not load model {model_name} due to: {e}")
            continue

        # For each prompt
        for prompt_name, prompt_template in PROMPTS.items():
            print(f"  Using prompt: {prompt_name}")

            # Create a separate output file for this model + prompt
            safe_model_name = model_name.replace("/", "_").replace(":", "_")
            out_filename = f"{safe_model_name}_{prompt_name}.hypo"

            with open(out_filename, 'w', encoding='utf-8') as out_f:
                for line_idx, source_line in enumerate(lines):
                    # Format the prompt
                    prompt_text = prompt_template.format(sentence=source_line)

                    # Generate raw text
                    raw_output = generate_raw_text(model, tokenizer, prompt_text)

                    # Post-process to keep only the final user-facing sentence
                    final_sentence = extract_final_sentence(raw_output)

                    # Write only the final sentence
                    out_f.write(final_sentence.strip() + "\n")
                    out_f.write("="*50 + "\n")

            print(f"    -> Results written to {out_filename}")

    print("\nAll model+prompt combinations processed.")


if __name__ == "__main__":
    main()