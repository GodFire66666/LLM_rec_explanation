#!/bin/bash

python train.py \
  --model SASRecAR \
  --dataset ml-100k \
  --config_file_list '["sh/ml-100k/SASRecAR/test.yaml"]'
