#!/bin/bash

python train.py \
  --model BPR \
  --dataset yelp2018 \
  --config_file_list '["sh/yelp2018/BPR/test.yaml"]'
