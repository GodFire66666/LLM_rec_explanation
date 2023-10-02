#!/bin/bash

python train.py \
  --model SASRec \
  --dataset ml-100k \
  --config_file_list '["sh/ml-100k/SASRec/test.yaml"]'
