#!/bin/bash

python train.py \
  --model LightGCN \
  --dataset Amazon_Luxury_Beauty \
  --config_file_list '["sh/Amazon_Luxury_Beauty/LightGCN/test.yaml"]'
