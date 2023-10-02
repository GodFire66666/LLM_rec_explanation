#!/bin/bash

python train.py \
  --model LightGCN \
  --dataset mind_small_dev \
  --config_file_list '["sh/mind_small_dev/LightGCN/test.yaml"]'
