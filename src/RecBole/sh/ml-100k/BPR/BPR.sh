#!/bin/bash

python train.py \
  --model BPR \
  --dataset ml-100k \
  --config_file_list '["sh/ml-100k/BPR/test.yaml"]'
