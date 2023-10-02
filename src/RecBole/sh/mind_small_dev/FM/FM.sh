#!/bin/bash

python train.py \
  --model FM \
  --dataset mind_small_dev \
  --config_file_list '["sh/mind_small_dev/FM/test.yaml"]'
