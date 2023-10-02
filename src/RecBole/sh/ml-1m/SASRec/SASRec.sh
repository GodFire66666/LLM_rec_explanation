#!/bin/bash

python train.py \
  --model SASRec \
  --dataset ml-1m \
  --config_file_list '["sh/ml-1m/SASRec/test.yaml"]'
