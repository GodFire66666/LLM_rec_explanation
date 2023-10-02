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

exp_tag="e18-LLama2-chat-7B-all"

echo "*********** infer ***********\n"
python infer.py \
    --model_name 'LLaMA2' \
    --base_model './experiments/可解释推荐/LLaMA2/'$exp_tag'/hf_ckpt' \
    --lora_weights './experiments/可解释推荐/LLaMA2/'$exp_tag \
    --use_lora False \
    --instruct_dir './data/可解释推荐data/test.json' \
    --prompt_template_name_cn '可解释推荐LLaMA2' \
    --prompt_template_name_en '可解释推荐LLaMA2'

# python infer.py \
# --model_name 'internlm-chat' \
# --base_model 'experiments/文修/internlm-chat/e6-internlm-7B-all/hf_ckpt' \
# --lora_weights './experiments/文修/internlm-chat/'$exp_tag \
# --use_lora False \
# --instruct_dir './data/文修test_data/all_data.json' \
# --prompt_template_name_cn '文修template中' \
# --prompt_template_name_en '文修template英'