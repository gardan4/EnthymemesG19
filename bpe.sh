mkdir -p gpt2_bpe
for SPLIT in train val
do
  for LANG in source target
  do
    python -m fairseq.examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json "gpt2_bpe/encoder.json" \
    --vocab-bpe "gpt2_bpe/vocab.bpe" \
    --inputs "enthymemes_data/$SPLIT.$LANG" \
    --outputs "enthymemes_data_bpe/$SPLIT.bpe.$LANG" \
    --workers 10 \
    --keep-empty;
  done
done

read -p "Press any key to continue..."