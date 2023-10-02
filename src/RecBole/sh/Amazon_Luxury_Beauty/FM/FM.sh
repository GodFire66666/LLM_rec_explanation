#!/bin/bash

python train.py \
  --model FM \
  --dataset Amazon_Luxury_Beauty \
  --config_file_list '["sh/Amazon_Luxury_Beauty/FM/test.yaml"]'
