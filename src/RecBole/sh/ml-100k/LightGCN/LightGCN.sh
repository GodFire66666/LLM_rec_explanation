#!/bin/bash

python train.py \
  --model LightGCN \
  --dataset ml-100k \
  --config_file_list '["sh/ml-100k/LightGCN/test.yaml"]'
