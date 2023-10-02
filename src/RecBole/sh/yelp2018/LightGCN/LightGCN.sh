#!/bin/bash

python train.py \
  --model LightGCN \
  --dataset yelp2018 \
  --config_file_list '["sh/yelp2018/LightGCN/test.yaml"]'
