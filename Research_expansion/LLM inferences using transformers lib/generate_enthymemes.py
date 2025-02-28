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
from tqdm import tqdm
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
    "Qwen/Qwen2.5-3B",
]

# Quantization choice: "4bit", "8bit", or "none"
QUANTIZATION_TYPE = "4bit"

# Define multiple prompts:
PROMPTS = {
    "CoT": """Rewrite the following sentence to make the hidden reason (enthymeme) explicit, inserting a short connecting statement that clarifies the reason. 
    Use the information between the hashtags as a direction for the reason. 
    Follow these steps to think carefully before answering:

    Step 1: Read and understand the sentence, identifying the main idea and the hashtag information.  
    Step 2: Break the sentence into parts and consider how the ideas are connected.  
    Step 3: Use the hashtag information to infer the hidden reason.  
    Step 4: Formulate one short sentence that explicitly states the reason.  

    Output only the final short sentence reason (nothing else).

    Original: "{sentence}"
    """,
    "few_shot": """Rewrite the following sentence to make the hidden reason (enthymeme) explicit,
    inserting a short connecting statement that clarifies the reason.
    Use the information between the hashtags as a dircetion for the reason
    Output only the final, very short single-sentence reason, with no additional commentary.

    Here are 2 examples:
    1.  Input: "I forgot my umbrella. # it was raining # I got soaked."
        Implicit premise: "and because it was raining."

    2.  Input: "Interns are replacing employees. # to have a better job # Unpaid internship exploit college students
        Implicit premise: And since that helps the company's bottom line.

    Original: "{sentence}"
    Rewritten: 
    """,
    "zero_shot": """Rewrite the following sentence to make the hidden reason (enthymeme) explicit,
    inserting a short connecting statement that clarifies the reason.
    Use the information between the hashtags as a dircetion for the reason
    Output only the final, very short single-sentence reason, with no additional commentary.

    Original: "{sentence}"
    Rewritten: 
    """,
}

# Generation hyperparams (tweak as desired)
GEN_KWARGS = {
    "max_new_tokens": 1000,
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

def generate_raw_text(model, tokenizer, prompt_text):
    """
    Basic text generation (returns raw string with all possible extra text).
    """
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, **GEN_KWARGS)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def extract_one_final_line(raw_generation: str) -> str:
    """
    Returns only the last non-empty line of `raw_generation`.
    Useful if your prompt instructs the model to:
       "Output the one final rewritten sentence directly after outputting a newline."
    """
    # Split on newlines, stripping whitespace
    lines = [line.strip() for line in raw_generation.split('\n') if line.strip()]
    if not lines:
        # If nothing remains, return an empty string
        return ''
    # The last element in `lines` should be the final rewritten sentence
    print(lines)
    return lines[-1]

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
                for line_idx, source_line in enumerate(tqdm(lines, desc="Processing lines")):
                    # Format the prompt
                    prompt_text = prompt_template.format(sentence=source_line)

                    # Generate raw text
                    raw_output = generate_raw_text(model, tokenizer, prompt_text)

                    # Post-process to keep only the final user-facing sentence
                    final_sentence = extract_one_final_line(raw_output)

                    # Write only the final sentence
                    out_f.write(final_sentence.strip() + "\n")

    print("\nAll model+prompt combinations processed.")


if __name__ == "__main__":
    main()