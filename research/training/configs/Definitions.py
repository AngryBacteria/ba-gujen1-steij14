from pydantic import BaseModel
from typing import Optional


class ModelConfig(BaseModel):
    lower_precision: bool
    attention_implementation: str
    lora: bool
    qlora: bool
    galore: bool


class DataProcessingConfig(BaseModel):
    processing_threads: int


class TrainerConfig(BaseModel):
    sequence_length: int
    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    optimizer: str


class GeneralConfig(BaseModel):
    model_id: str
    debug: bool
    wandb_logging: bool
    disable_annoying_warnings: bool
    run_name: Optional[str] = ""


class TrainConfig(BaseModel):
    general: GeneralConfig
    model: ModelConfig
    data_processing: DataProcessingConfig
    trainer: TrainerConfig
