#!/bin/bash

python train.py \
  --model SASRecAR \
  --dataset ml-1m \
  --config_file_list '["sh/ml-1m/SASRecAR/test.yaml"]'
