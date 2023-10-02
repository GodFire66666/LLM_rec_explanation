#!/bin/bash

python train.py \
  --model SASRecAR \
  --dataset Amazon_All_beauty \
  --config_file_list '["sh/Amazon_All_beauty/SASRecAR/test.yaml"]'
