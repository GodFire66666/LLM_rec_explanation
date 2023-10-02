#!/bin/bash

python train.py \
  --model SASRec \
  --dataset steam \
  --config_file_list '["sh/steam/SASRec/test.yaml"]'
