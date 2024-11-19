# EnthymemesG19

This project replicates the experiment, aiming to improve the results https://github.com/tuhinjubcse/EnthymemesEMNLP2021/tree/main

It now includes fairseq as a dependency, so there is no need to download it separately.

## Installation

- pip install requirements.txt
- install a `torch` version compatible with `CUDA`
```commandline
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

```
- ./run.sh
  - applies BPE encoding to train and val data
  - pre-processes the raw data to prepare it for training. (creates `dict.target.txt`. If already present it will give an error)
  - trains the BART model on the data

- run `./train_bart.sh`
  - This will start training the BART model on the data.
  - This will also run the validation 50 times. You will have `checkpoint1.pt` file in `bart-enthymemes-paracomet1` after the first validation.

After running this script, you will see "train.bpe.source" and "train.bpe.target" files in the `enthymemes_data_bpe` folder.

- run `generate.py`. This will perform text generation on input lines from `ikatdataparacomet.source`, and output the generation in `ikatdataparacomet.hypo`

`generate.py` performs it on checkpoint 50, but you can set it to a different one for testing purposes.

## Validation

- run `evaluate.py`. It computes the accuracy score.

## How is Para-Comet used in this experiment?

The data was already augmented with Para-Comet, and added to the three data-sets. For example, in D3test, we have `ikatdata.source`, and `ikatdataparacomet.source`. The dataset was augmented with paracomet to generate additional commonsense inferences.

