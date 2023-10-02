#!/bin/bash

python train.py \
  --model BPR \
  --dataset Amazon_Luxury_Beauty \
  --config_file_list '["sh/Amazon_Luxury_Beauty/BPR/test.yaml"]'
