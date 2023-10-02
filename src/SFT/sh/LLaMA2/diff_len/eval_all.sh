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

exp_tag="e19-LLama2-chat-7B-all"

echo "*********** infer ***********\n"
python infer_discriminator.py \
    --model_name 'LLaMA2' \
    --base_model './base_model/LLaMA2-7b-chat-base-model/hf_ckpt' \
    --lora_weights './experiments/判别器/LLaMA2-判别器/'$exp_tag \
    --use_lora True \
    --instruct_dir './data/判别器data/test_sft.json' \
    --prompt_template_name_cn '判别器LLaMA2' \
    --prompt_template_name_en '判别器LLaMA2'

# python infer.py \
# --model_name 'internlm-chat' \
# --base_model 'experiments/文修/internlm-chat/e6-internlm-7B-all/hf_ckpt' \
# --lora_weights './experiments/文修/internlm-chat/'$exp_tag \
# --use_lora False \
# --instruct_dir './data/文修test_data/all_data.json' \
# --prompt_template_name_cn '文修template中' \
# --prompt_template_name_en '文修template英'