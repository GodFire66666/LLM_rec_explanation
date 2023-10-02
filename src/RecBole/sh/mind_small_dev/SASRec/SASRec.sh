#!/bin/bash

python train.py \
  --model SASRec \
  --dataset mind_small_dev \
  --config_file_list '["sh/mind_small_dev/SASRec/test.yaml"]'
