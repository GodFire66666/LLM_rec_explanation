#!/bin/bash
#DSUB --job_type cosched
#DSUB -n rank
#DSUB -A root.bingxing2.gpuuser600
#DSUB -q root.default
#DSUB -R 'cpu=12;gpu=2;mem=100000'
#DSUB -l wuhanG5500
#DSUB -N 1
#DSUB -e %J.out
#DSUB -o %J.out

module load anaconda/2021.11 
module load cuda/11.8
source activate pytorch_39

exp_tag="e13-LLama2-Alpaca-Chinese-7B-all"

echo "*********** merge ***********\n"
python merge/src/export_model.py \
    --model_name 'LLaMA2' \
    --model_name_or_path './experiments/文修/LLaMA2-Alpaca/LLaMA2-Alpaca-base/hf_ckpt' \
    --checkpoint_dir './experiments/文修/LLaMA2-Alpaca/'$exp_tag \
    --output_dir './experiments/文修/LLaMA2-Alpaca/'$exp_tag'/hf_ckpt/'
    