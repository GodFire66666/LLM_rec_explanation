#!/bin/bash

python train.py \
  --model SASRecAR \
  --dataset yelp2018 \
  --config_file_list '["sh/yelp2018/SASRecAR/test.yaml"]'
