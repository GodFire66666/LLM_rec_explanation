#!/bin/bash

python train.py \
  --model BPR \
  --dataset steam \
  --config_file_list '["sh/steam/BPR/test.yaml"]'
