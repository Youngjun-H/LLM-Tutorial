#!/bin/bash
#SBATCH --job-name=finetuning
#SBATCH --nodelist=hpe159,hpe162
##SBATCH --nodelist=nv174
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=8G
#SBATCH --comment="LLM Fintuning TEST"

##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Job name:= " "$SLURM_JOB_NAME"
echo "Nodelist:= " "$SLURM_JOB_NODELIST"
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

echo "Run started at:- "
date
hostname -I;

# Conda 환경 활성화
# source /purestorage/AILAB/AI_2/yjhwang/opt/miniconda3/envs/llm

# # 다중 노드 통신 디버깅을 위한 환경 변수 (NCCL 디버그)
# # 문제가 발생할 경우 유용합니다. 평소에는 주석 처리해도 됩니다.
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# Huggingface cache directory
export HF_HOME=./cache/hf
# Kagglehub cache directory
export KAGGLEHUB_CACHE=./cache/kagglehub

accelerate launch train_sft.py \
    --model_name_or_path "meta-llama/Llama-2-13b-hf" \
    --dataset_name "mlabonne/guanaco-llama2-1k" \
    --use_peft True \
    --lora_r 64 \
    --lora_alpha 16 \
    --output_dir "./llama2-13b-sft-lora-ds" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --learning_rate 2e-4 \
    --logging_steps 10 \
    --bf16 True \
    --deepspeed "ds_config_zero2.json" \
    --max_len 2048 \
    --packing True

echo "################################################################################"
echo "Job Finished: $(date)"
echo "################################################################################"