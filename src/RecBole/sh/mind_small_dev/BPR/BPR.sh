#!/bin/bash

python train.py \
  --model BPR \
  --dataset mind_small_dev \
  --config_file_list '["sh/mind_small_dev/BPR/test.yaml"]'
