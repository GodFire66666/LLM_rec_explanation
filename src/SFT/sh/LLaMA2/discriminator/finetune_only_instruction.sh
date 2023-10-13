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

exp_tag="e23-LLama2-chat-7B-all"
echo "*********** Finetune ***********\n"
python finetune.py \
    --model_name 'LLaMA2' \
    --base_model './base_model/LLaMA2-7b-chat-base-model/hf_ckpt' \
    --data_path './data/disc_data/test_new.json' \
    --output_dir './experiments/disc/LLaMA2-disc/'$exp_tag \
    --prompt_template_name_cn 'disc_LLaMA2' \
    --prompt_template_name_en 'disc_LLaMA2' \
    --micro_batch_size 8 \
    --batch_size 8 \
    --cutoff_len 2048 \
    --wandb_project disc \
    --wandb_run_name $exp_tag \
    --num_epochs 1 \
    --val_set_size 100 \
    --lora_r 8 \
    --eval_steps 8 \
    --save_steps 8


# echo "*********** merge ***********\n"
# python merge/src/export_model.py \
#     --model_name 'LLaMA2' \
#     --model_name_or_path './base_model/LLaMA2-7b-chat-base-model/hf_ckpt' \
#     --checkpoint_dir './experiments/disc_/LLaMA2/'$exp_tag \
#     --output_dir './experiments/disc_/LLaMA2/'$exp_tag'/hf_ckpt/'


echo "*********** infer ***********\n"
python infer_discriminator.py \
    --model_name 'LLaMA2' \
    --base_model './base_model/LLaMA2-7b-chat-base-model/hf_ckpt' \
    --lora_weights './experiments/disc_/LLaMA2-disc/'$exp_tag \
    --use_lora True \
    --instruct_dir './data/disc_data/train_new.json' \
    --prompt_template_name_cn 'disc_LLaMA2' \
    --prompt_template_name_en 'disc_LLaMA2'