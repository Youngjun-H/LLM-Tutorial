import argparse
import os

import wandb
from dotenv import load_dotenv
from recipe import KoreanWebTextCPTRecipe, QAXOTRecipe
from transformers import AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer


def print_trainable_parameters(model):
    """
    모델의 학습 가능한 파라미터 수와 그 목록을 출력합니다.
    """
    trainable_params = 0
    all_param = 0
    trainable_param_names = []

    try:
        # DDP 모드에서는 모델의 module을 사용
        model_to_check = model.module if hasattr(model, "module") else model

        for name, param in model_to_check.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                trainable_param_names.append(name)

        print(f"✅ 학습 가능한 파라미터 수: {trainable_params}")
        print(f"✅ 전체 파라미터 수: {all_param}")

        # ZeroDivisionError 방지
        if all_param > 0:
            print(
                f"✅ 학습 가능한 파라미터 비율: {100 * trainable_params / all_param:.2f}%"
            )
        else:
            print("✅ 학습 가능한 파라미터 비율: 계산 불가 (분산 학습 모드)")

        print("\n--- 학습 가능한 파라미터 목록 ---")
        for name in trainable_param_names:
            print(name)

    except Exception as e:
        print(f"⚠️ 파라미터 계산 중 오류 발생: {e}")
        print(
            "분산 학습 모드에서는 파라미터가 분산되어 있어 정확한 계산이 어려울 수 있습니다."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepspeed", type=str, help="DeepSpeed config file")
    args = parser.parse_args()

    # Load recipe
    recipe = KoreanWebTextCPTRecipe()
    # Training Arguments
    training_args = recipe.load_config()

    # wandb 초기화 (메인 프로세스에서만) -> 잘 적용되는지 확인 해볼 것
    if training_args.local_rank <= 0:
        wandb.login()
        wandb.init(
            project="sft-training",
            name="korean-webtext-cpt-training",
        )

    # Model & tokenizer
    model_name = recipe.load_model()

    # 실제 모델 객체 로드 (DeepSpeed 설정 제거)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",
        use_cache=False,
    )

    if training_args.local_rank <= 0:
        print(f"model architecture:\n {model}")

    # 부분적 freezing 설정
    for param in model.parameters():
        param.requires_grad = False
    for param in model.model.embed_tokens.parameters():
        param.requires_grad = True

    tokenizer = recipe.load_tokenizer()

    ds = recipe.load_dataset()
    train_ds = ds["train"]
    eval_ds = ds["test"]

    # Trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
    )

    # 학습 시작
    trainer.train()

    # 메인 프로세스에서만 wandb 종료
    if training_args.local_rank <= 0:
        wandb.finish()


if __name__ == "__main__":
    load_dotenv()
    main()
