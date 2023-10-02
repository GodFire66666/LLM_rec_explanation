#!/bin/bash

python train.py \
  --model LightGCN \
  --dataset ml-1m \
  --config_file_list '["sh/ml-1m/LightGCN/test.yaml"]'
