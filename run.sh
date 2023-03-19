poetry run python3 run_clm.py \
  --model_name_or_path=rinna/japanese-gpt2-medium \
  --train_file=./dataset/dokujo_train.txt \
  --validation_file=./dataset/dokujo_val.txt \
  --do_train \
  --do_eval \
  --num_train_epochs=10 \
  --save_steps=500 \
  --block_size 512 \
  --save_total_limit=3 \
  --per_device_train_batch_size=1 \
  --per_device_eval_batch_size=1 \
  --output_dir=text_generation/ \
  --overwrite_output_dir \
  --use_fast_tokenizer=False \
  --logging_steps=5
