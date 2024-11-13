fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --srcdict "gpt2_bpe/dict.txt" \
  --trainpref "enthymemes_data_bpe/train.bpe" \
  --validpref "enthymemes_data_bpe/val.bpe" \
  --destdir "enthymemes_data_bin" \
  --workers 10 \
  --joined-dictionary

read -p "Press any key to continue..."