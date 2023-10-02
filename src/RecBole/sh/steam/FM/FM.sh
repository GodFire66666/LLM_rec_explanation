#!/bin/bash

python train.py \
  --model FM \
  --dataset steam \
  --config_file_list '["sh/steam/FM/test.yaml"]'
