# train_dpo.py
import os

from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOConfig, DPOTrainer

# Accelerator 초기화
accelerator = Accelerator()

# 모델과 토크나이저 로드
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct", cache_dir="./cache"
)
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct", cache_dir="./cache"
)

# 패딩 토큰 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 데이터셋 로드
train_dataset = load_dataset(
    "trl-lib/ultrafeedback_binarized", split="train", cache_dir="./cache"
)

# DPO 설정
training_args = DPOConfig(
    output_dir="Qwen2-0.5B-DPO",
    num_train_epochs=3,
    per_device_train_batch_size=8,  # H100 성능에 맞게 증가
    gradient_accumulation_steps=2,  # 총 배치 크기 유지
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    warmup_steps=100,
    report_to=None,  # wandb 비활성화
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
)

# DPO 트레이너 초기화
trainer = DPOTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
)

# Accelerate를 사용하여 모델, 옵티마이저, 스케줄러 준비
model, optimizer, lr_scheduler = accelerator.prepare(
    trainer.model, trainer.optimizer, trainer.lr_scheduler
)

# 학습 실행
trainer.train()

# 모델 저장
if accelerator.is_main_process:
    trainer.save_model()
    tokenizer.save_pretrained("Qwen2-0.5B-DPO")
