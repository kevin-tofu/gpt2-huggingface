# huggingface-trainer

## Install Dependencies

```bash

poetry install --no-root

```

## Execute Training

```bash

CUDA_VISIBLE_DEVICES="0" poetry run python3 ./src/run_clm.py \
  --model_name_or_path=rinna/japanese-gpt2-small \
  --train_file=./dataset/it-life-hack_train.txt \
  --validation_file=./dataset/it-life-hack_val.txt \
  --do_train \
  --do_eval \
  --num_train_epochs=10 \
  --save_steps=500 \
  --block_size 512 \
  --save_total_limit=3 \
  --per_device_train_batch_size=1 \
  --per_device_eval_batch_size=1 \
  --output_dir=models/it-life-hack \
  --overwrite_output_dir \
  --use_fast_tokenizer=False \
  --logging_steps=5

```

## Inference

### Prompt

```

poetry run python3 src/inference.py  \
  --model_name_or_path=rinna/japanese-gpt2-small \
  --output_dir=models/it-life-hack \
  --use_fast_tokenizer=False \
  --prompt="今日、我々は家電量販店に"

```

### The Output

```

completion 0 :  今日、我々は家電量販店に集った。私はすぐに店に入ったが、店員は驚いた。私もレジでミスをし、私の代わりに行ったが、全然覚えてくれなかった。店員の一番下のおばさんが、少し長い腕振りをしてレジに立っているが、誰も、私に何か質問をされたようには見えなかった。

```