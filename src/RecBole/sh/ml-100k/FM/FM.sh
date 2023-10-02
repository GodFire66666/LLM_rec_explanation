#!/bin/bash

python train.py \
  --model FM \
  --dataset ml-100k \
  --config_file_list '["sh/ml-100k/FM/test.yaml"]'
