poetry run python3 src/inference.py  \
  --model_name_or_path=rinna/japanese-gpt2-small \
  --output_dir=models/it-life-hack \
  --use_fast_tokenizer=False \
  --prompt="今日、我々は家電量販店に"