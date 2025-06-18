from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    """
    모델과 토크나이저 관련 인자
    """
    model_name_or_path: str = field(
        metadata={"help": "파인튜닝할 모델의 Hugging Face Hub 경로 또는 로컬 경로"}
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={
            "help": ("모델을 로드할 데이터 타입. A100/H100 GPU에서는 'bfloat16' 권장, "
                     "V100/T4 등에서는 'float16' 사용. 'auto'도 가능.")
        },
    )

@dataclass
class DataArguments:
    """
    데이터셋 관련 인자
    """
    dataset_name: str = field(
        metadata={"help": "학습에 사용할 데이터셋의 Hugging Face Hub 경로"}
    )

@dataclass
class PeftArguments:
    """
    PEFT(LoRA) 관련 파라미터
    """
    use_peft: bool = field(
        default=True,
        metadata={"help": "PEFT(LoRA)를 사용하여 파라미터 효율적 파인튜닝을 할지 여부"}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA attention dimension (rank)"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha 파라미터 (일종의 scaling factor)"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA 레이어의 드롭아웃 비율"}
    )
