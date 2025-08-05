import os

import wandb
from accelerate import Accelerator
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def main():
    wandb.init()

    dataset = load_dataset("stanfordnlp/imdb", split="train", cache_dir="./cache")

    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-350m", cache_dir="./cache"
    )
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m", cache_dir="./cache")

    # 방법 1: 파일 경로 사용 (권장)
    training_args = SFTConfig(max_length=512, output_dir="/tmp/sft-opt")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    main()
