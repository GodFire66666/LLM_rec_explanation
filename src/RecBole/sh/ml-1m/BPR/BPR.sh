#!/bin/bash

python train.py \
  --model BPR \
  --dataset ml-1m \
  --config_file_list '["sh/ml-1m/BPR/test.yaml"]'
