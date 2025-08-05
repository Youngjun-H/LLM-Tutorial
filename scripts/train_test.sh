#!/bin/bash

#SBATCH --job-name=multinode
#SBATCH --output=logs/train_test_%A_%j.log
#SBATCH --error=logs/train_test_%A_%j.err
#SBATCH --nodes=1                   # number of nodes
#SBATCH --nodelist=cubox03
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:8                # number of GPUs per node (8에서 4로 줄임)
#SBATCH --cpus-per-task=14         # number of cores per tasks (160에서 32로 줄임)

######################
### Set environment ###
######################
export GPUS_PER_NODE=8
######################

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################

export LAUNCHER="accelerate launch \
    --config_file SFT/deepspeed_zero3.yaml \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    "
export ACCELERATE_DIR="${ACCELERATE_DIR:-/accelerate}"
export SCRIPT="${ACCELERATE_DIR}/SFT/train.py"
# export SCRIPT_ARGS=" \
#     --mixed_precision fp16 \
#     --output_dir ${ACCELERATE_DIR}/examples/output \
#     "
export PYTHON_FILE="SFT/train.py"

# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $PYTHON_FILE" 
srun $CMD