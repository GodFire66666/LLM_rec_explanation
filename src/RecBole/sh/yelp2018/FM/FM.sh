#!/bin/bash

python train.py \
  --model FM \
  --dataset yelp2018 \
  --config_file_list '["sh/yelp2018/FM/test.yaml"]'
