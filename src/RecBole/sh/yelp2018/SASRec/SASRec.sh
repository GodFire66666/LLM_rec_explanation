#!/bin/bash

python train.py \
  --model SASRec \
  --dataset yelp2018 \
  --config_file_list '["sh/yelp2018/SASRec/test.yaml"]'
