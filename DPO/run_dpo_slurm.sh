#!/bin/bash
#SBATCH --job-name=dpo_training
#SBATCH --nodelist=cubox13,cubox14
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --output=logs/dpo_%j.out
#SBATCH --error=logs/dpo_%j.err

# 환경 설정
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# SLURM 노드 정보 설정
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID

echo "Master node: $MASTER_ADDR"
echo "World size: $WORLD_SIZE"
echo "Local rank: $LOCAL_RANK"
echo "Global rank: $RANK"

# 로그 디렉토리 생성
mkdir -p logs
mkdir -p cache
mkdir -p Qwen2-0.5B-DPO

# 가상환경 활성화 (필요한 경우)
# source /path/to/your/venv/bin/activate

# 패키지 설치
echo "Installing required packages..."
pip install --upgrade pip
pip install torch>=2.0.0 transformers>=4.35.0 datasets>=2.14.0 trl>=0.7.0 accelerate>=0.24.0 peft>=0.6.0 bitsandbytes>=0.41.0

# 설치 확인
echo "Checking installed packages..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import datasets; print(f'Datasets version: {datasets.__version__}')"
python -c "import trl; print(f'TRL version: {trl.__version__}')"
python -c "import accelerate; print(f'Accelerate version: {accelerate.__version__}')"

# DPO 학습 실행
echo "Starting DPO training..."
accelerate launch --multi_gpu --num_processes=16 --num_machines=2 --machine_rank=0 --main_process_port=29500 --mixed_precision=bf16 train_dpo.py

echo "Training completed!" 