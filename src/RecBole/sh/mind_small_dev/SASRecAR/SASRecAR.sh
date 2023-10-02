#!/bin/bash

python train.py \
  --model SASRecAR \
  --dataset mind_small_dev \
  --config_file_list '["sh/mind_small_dev/SASRecAR/test.yaml"]'
