#!/bin/bash

python train.py \
  --model LightGCN \
  --dataset steam \
  --config_file_list '["sh/steam/LightGCN/test.yaml"]'
