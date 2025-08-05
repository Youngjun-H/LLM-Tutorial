#!/bin/bash
#SBATCH --job-name=SFT
##SBATCH --nodelist=hpe159,hpe162
#SBATCH --nodelist=cubox12
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=8G
#SBATCH --comment="LLM Fintuning TEST"
#SBATCH --output=notebook_%A.log

##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Job name:= " "$SLURM_JOB_NAME"
echo "Nodelist:= " "$SLURM_JOB_NODELIST"
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

echo "Run started at:- "
date
hostname -I;

# Huggingface cache directory
# export HF_HOME=./cache/hf
export HF_HOME=./cache/hf

# Kagglehub cache directory
# export KAGGLEHUB_CACHE=./cache/kagglehub
export KAGGLEHUB_CACHE=./cache/kagglehub

srun jupyter notebook --ip 0.0.0.0