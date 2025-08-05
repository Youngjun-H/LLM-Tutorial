#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --comment="LLM supervised fine-tuning"
#SBATCH --nodelist=cubox01,cubox02,cubox03,cubox04,cubox05,cubox06,cubox08,cubox09
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=8G

##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Job name:= " "$SLURM_JOB_NAME"
echo "Nodelist:= " "$SLURM_JOB_NODELIST"
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

echo "Run started at:- "
date

# Huggingface cache directory
export HF_HOME=./cache/hf

huggingface-cli login --token YOUR_HF_TOKEN

NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=8
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))
echo "World size: $NUM_NODES x $GPUS_PER_NODE = $WORLD_SIZE"

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
echo "$nodes"

# Get the IP address of the head node
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
head_port=29500

# NCCL network configuration (for Infiniband)
export NCCL_SOCKET_IFNAME=eno1 # TODO NCCL 통신 에러 발생하면 이 부분 원복.

echo "****Starting HEAD at $head_node, $head_node_ip:$head_port"

# Single node, 8 processes
srun --nodes=1 --ntasks=1 -w "$head_node" \
    accelerate launch \
        --config_file lmtrain/sft/deepspeed_zero3.yaml \
        --num_machines "$NUM_NODES" \
        --num_processes "$WORLD_SIZE" \
        --main_process_ip "$head_node_ip" \
        --main_process_port "$head_port" \
        --machine_rank 0 \
    lmtrain/sft/train.py &
sleep 15

# Start worker from 1 (0 is head node)
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "****Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
      accelerate launch \
        --config_file lmtrain/sft/deepspeed_zero3.yaml \
        --num_machines "$NUM_NODES" \
        --num_processes "$WORLD_SIZE" \
        --main_process_ip "$head_node_ip" \
        --main_process_port "$head_port" \
        --machine_rank "$i" \
      lmtrain/sft/train.py &
    sleep 5
done

# 주의) 모든 데몬 프로세스가 끝날 때 까지 대기하는 명령어.
wait