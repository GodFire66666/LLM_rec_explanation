#!/bin/bash
#DSUB --job_type cosched
#DSUB -n rank
#DSUB -A root.bingxing2.gpuuser600
#DSUB -q root.default
#DSUB -R 'cpu=12;gpu=2;mem=90000'
#DSUB -l wuhanG5500
#DSUB -N 1
#DSUB -e %J.out
#DSUB -o %J.out

module load anaconda/2021.11 
module load cuda/11.8
source activate pytorch_39
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2

# # 创建状态文件，用于控制采集的进程
# STATE_FILE="state_${BATCH_JOB_ID}"
# /usr/bin/touch ${STATE_FILE}

# # 后台循环采集，每间隔 1s 采集一次 GPU 数据。
# # 采集的数据将输出到本地 gpu_作业 ID.log 文件中

# function gpus_collection(){
#         while [[ `cat "${STATE_FILE}" | grep "over" | wc -l` == "0" ]]; do
#                 /usr/bin/sleep 1
#                 /usr/bin/nvidia-smi >> "gpu_${BATCH_JOB_ID}.log"
#         done
# }
# gpus_collection  &
len=500
a=3
b=6
exp_tag="e24-LLama2-chat-7B-diff_len_"$len
val_set_size=`expr $len \* $a / $b`

echo "*********** Finetune ***********\n"

python finetune.py \
    --model_name 'LLaMA2' \
    --base_model './base_model/LLaMA2-7b-chat-base-model/hf_ckpt' \
    --data_path './data/diff_len/'$len'/train.json' \
    --output_dir './experiments/diff_len/'$len'/'$exp_tag \
    --prompt_template_name_cn '可解释推荐LLaMA2' \
    --prompt_template_name_en '可解释推荐LLaMA2' \
    --micro_batch_size 8 \
    --batch_size 8 \
    --cutoff_len 4096 \
    --wandb_project diff_len \
    --wandb_run_name $exp_tag \
    --num_epochs 2 \
    --val_set_size $val_set_size \
    --lora_r 8 \
    --eval_steps 8 \
    --save_steps 8


# echo "*********** merge ***********\n"
# python merge/src/export_model.py \
#     --model_name 'LLaMA2' \
#     --model_name_or_path './base_model/LLaMA2-7b-chat-base-model/hf_ckpt' \
#     --checkpoint_dir './experiments/判别器/LLaMA2/'$exp_tag \
#     --output_dir './experiments/判别器/LLaMA2/'$exp_tag'/hf_ckpt/'


# echo "*********** infer ***********\n"
# python infer_discriminator.py \
#     --model_name 'LLaMA2' \
#     --base_model './base_model/LLaMA2-7b-chat-base-model/hf_ckpt' \
#     --lora_weights './experiments/判别器/LLaMA2-判别器/'$exp_tag \
#     --use_lora True \
#     --instruct_dir './data/判别器data/train_new.json' \
#     --prompt_template_name_cn '判别器LLaMA2' \
#     --prompt_template_name_en '判别器LLaMA2'