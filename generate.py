import torch
from fairseq.models.bart import BARTModel
import os
import time
import numpy as np

#print device
print(torch.cuda.current_device())

datadir = 'enthymemes_data'
cpdir = 'bart-enthymemes-paracomet1/'
bart = BARTModel.from_pretrained(
    cpdir,
    checkpoint_file='checkpoint50.pt',
    data_name_or_path=datadir
)

# Move the model to GPU and print device information

bart.cuda()
print(f"Model is on device: {next(bart.parameters()).device}")
bart.eval()
np.random.seed(4)
torch.manual_seed(4)

bsz = 32  # Increased batch size
maxb = 200
minb = 7

with open('./D3test/ikatdataparacomet.source') as source, open('./D3test/ikatdataparacomet.hypo', 'w') as fout:
    slines = []
    for sline in source:
        slines.append(sline.strip())
        if len(slines) == bsz:
            # Optionally, print batch processing information
            print(f"Processing batch of size {len(slines)} on device: {next(bart.parameters()).device}")
            with torch.no_grad():
                hypotheses_batch = bart.sample(
                    slines,
                    beam=5,
                    lenpen=2.0,
                    max_len_b=maxb,
                    min_len_b=minb,
                    no_repeat_ngram_size=3
                )
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis.replace('\n', '') + '\n')
            slines = []
    # Process any remaining lines
    if slines:
        print(f"Processing final batch of size {len(slines)} on device: {next(bart.parameters()).device}")
        with torch.no_grad():
            hypotheses_batch = bart.sample(
                slines,
                beam=5,
                lenpen=2.0,
                max_len_b=maxb,
                min_len_b=minb,
                no_repeat_ngram_size=3
            )
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis.replace('\n', '') + '\n')
