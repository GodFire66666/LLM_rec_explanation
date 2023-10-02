#!/bin/bash

python train.py \
  --model FM \
  --dataset ml-1m \
  --config_file_list '["sh/ml-1m/FM/test.yaml"]'
