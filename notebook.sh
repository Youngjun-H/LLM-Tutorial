#! /bin/bash

#SBATCH --job-name=nemodask
#SBATCH --comment="Dask cluster for LLM dataset curation"
#SBATCH --nodelist=cubox12,cubox13
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=16G

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =================================================================
# Begin easy customization
# =================================================================

hostname -I;

export HF_HOME=/purestorage/AILAB/AI_2/huggingface_cache
export TORCH_HOME=./cache/torch

# Base directory for all Slurm job logs and files
# Does not affect directories referenced in your script
export BASE_JOB_DIR=`pwd`/nemo-curator-jobs
export JOB_DIR=$BASE_JOB_DIR/$SLURM_JOB_ID

# Directory for Dask cluster communication and logging
# Must be paths inside the container that are accessible across nodes
export LOGDIR=$JOB_DIR/logs
export PROFILESDIR=$JOB_DIR/profiles
export SCHEDULER_FILE=$LOGDIR/scheduler.json
export SCHEDULER_LOG=$LOGDIR/scheduler.log
export DONE_MARKER=$LOGDIR/done.txt

# Create directories
mkdir -p $JOB_DIR
mkdir -p $LOGDIR
mkdir -p $PROFILESDIR

# Device type
# This will change depending on the module to run
export DEVICE="cpu"

# Container parameters
export CONTAINER_IMAGE=/path/to/container
# Make sure to mount the directories your script references
export BASE_DIR=`pwd`
export MOUNTS="${BASE_DIR}:${BASE_DIR}"
# Below must be path to entrypoint script on your system
export CONTAINER_ENTRYPOINT=`pwd`/entrypoint.sh

# Network interface specific to the cluster being used
export INTERFACE=eno1 # eth0
export PROTOCOL=tcp

# CPU related variables
# export CPU_WORKER_MEMORY_LIMIT=0  # 0 means no memory limit
# export CPU_WORKER_PER_NODE=128  # number of cpu workers per node
# --- 새로운 설정 (After) ---
# 워커 수를 128개에서 8개로 대폭 줄여, 워커 하나가 더 많은 메모리와 CPU를 사용하도록 합니다.
# 계산 근거:
#   - 노드당 총 자원: 96 코어, 1536 GB RAM
#   - 워커당 할당량: 96코어/8워커 = 12코어, 1536GB/8워커 = 192GB

export CPU_WORKER_PER_NODE=8          # 노드당 워커 수를 8개로 변경
export CPU_WORKER_MEMORY_LIMIT="192G" # 각 워커의 메모리 제한을 192GB로 명시

# GPU related variables
export RAPIDS_NO_INITIALIZE="1"
export CUDF_SPILL="1"
export RMM_SCHEDULER_POOL_SIZE="1GB"
export RMM_WORKER_POOL_SIZE="72GiB"
export LIBCUDF_CUFILE_POLICY=OFF


# =================================================================
# End easy customization
# =================================================================

# Start the container
srun ${CONTAINER_ENTRYPOINT}