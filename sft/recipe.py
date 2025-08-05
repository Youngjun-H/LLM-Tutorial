import os

from datasets import DatasetDict, interleave_datasets, load_dataset
from transformers import AutoTokenizer
from trl import SFTConfig


class Recipe:

    def load_dataset(self) -> DatasetDict:
        raise NotImplementedError()

    def load_model(self) -> str:
        raise NotImplementedError()

    def load_tokenizer(self):
        raise NotImplementedError()

    def load_config(self) -> SFTConfig:
        raise NotImplementedError()


class KoreanWebTextCPTRecipe(Recipe):

    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-1.5B"

    def load_dataset(self) -> DatasetDict:
        ds = interleave_datasets(
            [
                load_dataset("HAERAE-HUB/KOREAN-WEBTEXT", split="train", cache_dir="/purestorage/AILAB/AI_2/yjhwang/work/cache/hf"),  # type: ignore
                load_dataset("lehduong/math-dolmino", split="train", cache_dir="/purestorage/AILAB/AI_2/yjhwang/work/cache/hf").shuffle(seed=42).select_columns(["text"]).shuffle(seed=42).select(range(2000000)),  # type: ignore
            ],
            [0.9, 0.1],
            seed=42,
        )
        return ds.train_test_split(test_size=4000, seed=42)

    def load_model(self) -> str:
        return self.model_name

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.model_name, cache_dir="/purestorage/AILAB/AI_2/yjhwang/work/cache/hf"
        )

    def load_config(self) -> SFTConfig:
        # 노드 수 제한 제거 - 유연한 노드 수 지원
        slurm_job_id = os.getenv("SLURM_JOB_ID")
        assert slurm_job_id is not None

        # 현재 노드 수 확인 (디버깅용)
        num_nodes = int(os.getenv("SLURM_JOB_NUM_NODES", "1"))
        print(f"현재 할당된 노드 수: {num_nodes}")

        return SFTConfig(
            output_dir=f"./output_{slurm_job_id}",
            bf16=True,
            model_init_kwargs=dict(
                trust_remote_code=True,
                torch_dtype="bfloat16",
                attn_implementation="flash_attention_2",
                use_cache=False,  # 메모리 사용량 감소
            ),
            use_liger_kernel=True,
            assistant_only_loss=False,
            dataset_num_proc=16,
            dataloader_num_workers=0,
            dataloader_drop_last=True,
            max_length=1024 * 8,
            packing=True,
            padding_free=False,
            dataset_text_field="text",
            max_steps=1000,  # 500,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=5e-6,  # 1e-5,
            warmup_ratio=0.0,
            weight_decay=0.1,  # 0.01,
            lr_scheduler_type="linear",
            gradient_checkpointing=True,
            logging_strategy="steps",
            logging_steps=10,
            report_to="wandb",
            save_strategy="steps",
            save_steps=250,
            save_only_model=True,
            save_safetensors=True,
            save_total_limit=5,
            eval_strategy="steps",
            eval_steps=250,
        )


class QAXOTRecipe(Recipe):

    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-1.5B"

    def load_dataset(self) -> DatasetDict:
        ds = interleave_datasets(
            [
                load_dataset("HAERAE-HUB/KOREAN-WEBTEXT", split="train", cache_dir="/purestorage/AILAB/AI_2/yjhwang/work/cache/hf"),  # type: ignore
                load_dataset("lehduong/math-dolmino", split="train", cache_dir="/purestorage/AILAB/AI_2/yjhwang/work/cache/hf").shuffle(seed=42).select_columns(["text"]).shuffle(seed=42).select(range(2000000)),  # type: ignore
            ],
            [0.9, 0.1],
            seed=42,
        )
        return ds.train_test_split(test_size=4000, seed=42)

    def load_model(self) -> str:
        return self.model_name

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            "skt/A.X-4.0-Light",
            cache_dir="/purestorage/AILAB/AI_2/yjhwang/work/cache/hf",
            trust_remote_code=True,
        )

    def load_config(self) -> SFTConfig:
        # 노드 수 제한 제거 - 유연한 노드 수 지원
        slurm_job_id = os.getenv("SLURM_JOB_ID")
        assert slurm_job_id is not None
        return SFTConfig(
            output_dir=f"./output_{slurm_job_id}",
            bf16=True,
            model_init_kwargs=dict(
                trust_remote_code=True,
                torch_dtype="bfloat16",
                attn_implementation="flash_attention_2",
                use_cache=False,  # 메모리 사용량 감소
            ),
            use_liger_kernel=True,
            assistant_only_loss=False,
            dataset_num_proc=16,
            dataloader_num_workers=0,
            dataloader_drop_last=True,
            max_length=1024 * 8,
            packing=True,
            padding_free=False,
            dataset_text_field="text",
            max_steps=1000,  # 500,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=5e-6,  # 1e-5,
            warmup_ratio=0.0,
            weight_decay=0.1,  # 0.01,
            lr_scheduler_type="linear",
            gradient_checkpointing=True,
            logging_strategy="steps",
            logging_steps=10,
            report_to="wandb",
            save_strategy="steps",
            save_steps=250,
            save_only_model=True,
            save_safetensors=True,
            save_total_limit=5,
            eval_strategy="steps",
            eval_steps=250,
        )
