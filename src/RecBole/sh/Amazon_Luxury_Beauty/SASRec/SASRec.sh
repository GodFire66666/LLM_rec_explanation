#!/bin/bash

python train.py \
  --model SASRec \
  --dataset Amazon_Luxury_Beauty \
  --config_file_list '["sh/Amazon_Luxury_Beauty/SASRec/test.yaml"]'
