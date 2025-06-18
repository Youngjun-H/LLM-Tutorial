
import os
import torch
from dotenv import load_dotenv
from peft import LoraConfig
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import SFTConfig, SFTTrainer
from arguments import ModelArguments, DataArguments, PeftArguments

def main():
    # ======================================================================================
    # 인자 파싱
    # HfArgumentParser는 위에서 정의한 dataclass들을 파싱하여 TrainingArguments와 함께 사용
    # ======================================================================================
    parser = HfArgumentParser((ModelArguments, DataArguments, PeftArguments, SFTConfig))
    model_args, data_args, peft_args, sft_config = parser.parse_args_into_dataclasses()

    # ======================================================================================
    # 데이터셋 로딩
    # ======================================================================================
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = f"### 지시: {example['instruction'][i]}\n\n### 입력: {example['input'][i]}\n\n### 응답: {example['output'][i]}"
            output_texts.append(text)
        return output_texts

    # ======================================================================================
    # 모델 로딩 (bfloat16 및 PEFT/LoRA 적용)
    # ======================================================================================
    print(f"'{model_args.model_name_or_path}' 모델을 로딩합니다...")
    
    # torch_dtype을 문자열에서 실제 torch 데이터 타입으로 변환
    # getattr(torch, 'bfloat16') -> torch.bfloat16
    torch_dtype = getattr(torch, model_args.torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        # device_map을 사용하지 않음. accelerate가 자동으로 GPU에 분산 배치함.
        # device_map="auto" # <-- DeepSpeed 사용 시 이 옵션은 비활성화해야 합니다.
    )

    # ======================================================================================
    # 토크나이저 로딩
    # Causal LM은 보통 오른쪽에 패딩을 추가합니다.
    # ======================================================================================
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    # ======================================================================================
    # PEFT (LoRA) 설정
    # ======================================================================================
    peft_config = None
    if peft_args.use_peft:
        print("PEFT (LoRA) 설정을 적용합니다.")
        peft_config = LoraConfig(
            r=peft_args.lora_r,
            lora_alpha=peft_args.lora_alpha,
            lora_dropout=peft_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

    # ======================================================================================
    # 데이터셋 로딩 및 준비
    # ======================================================================================
    print(f"'{data_args.dataset_name}' 데이터셋을 로딩합니다.")
    dataset = load_dataset(data_args.dataset_name, split="train")
    
    # 데이터셋 형식 예시:
    # 각 샘플은 'text' 필드에 다음과 같은 형식의 문자열을 포함해야 합니다.
    # "<s>[INST] What is the capital of France? [/INST] The capital of France is Paris.</s>"
    # SFTTrainer는 이 텍스트를 자동으로 토크나이징하고 라벨을 생성합니다.
    # print(f"사용할 텍스트 필드: '{data_args.dataset_text_field}'")
    # print(f"데이터셋 샘플: \n{dataset[0][data_args.dataset_text_field]}")


    # ======================================================================================
    # TRL SFTTrainer 초기화
    # SFTTrainer는 TrainingArguments를 상속받아 분산 학습 및 DeepSpeed를 완벽하게 지원합니다.
    # ======================================================================================
    trainer = SFTTrainer(
        model=model,
        args=sft_config, # TrainingArguments 대신 SFTConfig 객체를 전달합니다.
        peft_config=peft_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func
    )

    # ======================================================================================
    # 훈련 시작
    # ======================================================================================
    print("훈련을 시작합니다.")
    trainer.train()
    print("훈련이 완료되었습니다.")

    # ======================================================================================
    # 모델 저장
    # PEFT를 사용하면 어댑터(LoRA 가중치)만 저장되어 매우 가볍습니다.
    # full-finetuning을 했다면 전체 모델이 저장됩니다.
    # ======================================================================================
    print(f"모델을 '{training_args.output_dir}'에 저장합니다.")
    trainer.save_model()
    print("모델 저장이 완료되었습니다.")

if __name__ == "__main__":
    load_dotenv()
    main()