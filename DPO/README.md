# DPO (Direct Preference Optimization) Training

이 프로젝트는 Qwen2-0.5B-Instruct 모델을 사용하여 DPO 학습을 수행합니다.

## 파일 구조

- `train_dpo.py`: 메인 학습 스크립트
- `run_dpo_slurm.sh`: SLURM 환경에서 실행하는 스크립트
- `accelerate_config.yaml`: Accelerate 설정 파일
- `requirements.txt`: 필요한 패키지 목록

## 사용 방법

### 1. SLURM 환경에서 실행

```bash
# SLURM 작업 제출
sbatch run_dpo_slurm.sh
```

### 2. 로컬 환경에서 실행

```bash
# 패키지 설치
pip install -r requirements.txt

# Accelerate 설정 (처음 실행시에만)
accelerate config

# 학습 실행
accelerate launch train_dpo.py
```

## SLURM 설정

현재 SLURM 스크립트는 다음 설정으로 구성되어 있습니다:

- **노드**: cubox13, cubox14 (2개)
- **GPU**: 16개 (H100, 노드당 8개)
- **CPU**: 16코어 (노드당)
- **통신 포트**: 29500
- **마스터 노드**: 자동 감지

### 멀티 노드 통신 설정

스크립트는 자동으로 다음 환경 변수를 설정합니다:
- `MASTER_ADDR`: 첫 번째 노드의 호스트명
- `MASTER_PORT`: 노드 간 통신 포트 (29500)
- `WORLD_SIZE`: 총 프로세스 수 (16)
- `LOCAL_RANK`: 각 노드 내에서의 로컬 랭크
- `RANK`: 전체 노드에서의 글로벌 랭크

### Accelerate 실행 방식

설정 파일 대신 명령어 라인 파라미터를 사용합니다:
- `--multi_gpu`: 멀티 GPU 활성화
- `--num_processes=16`: 총 16개 프로세스
- `--num_machines=2`: 2개 노드
- `--machine_rank=0`: 마스터 노드 랭크
- `--main_process_port=29500`: 통신 포트
- `--mixed_precision=bf16`: 혼합 정밀도

필요에 따라 `run_dpo_slurm.sh` 파일의 SLURM 파라미터를 수정하세요.

## 모델 설정

- **베이스 모델**: Qwen/Qwen2-0.5B-Instruct
- **데이터셋**: trl-lib/ultrafeedback_binarized
- **학습률**: 5e-5
- **배치 크기**: 8 (per device, H100 최적화)
- **Gradient Accumulation**: 2
- **에포크**: 3

## 출력

학습된 모델은 `Qwen2-0.5B-DPO` 디렉토리에 저장됩니다.

## 로그

학습 로그는 `logs/` 디렉토리에 저장됩니다:
- `logs/dpo_<job_id>.out`: 표준 출력
- `logs/dpo_<job_id>.err`: 에러 로그 