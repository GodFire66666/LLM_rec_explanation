#!/bin/bash

python train.py \
  --model SASRecAR \
  --dataset steam \
  --config_file_list '["sh/steam/SASRecAR/test.yaml"]'
