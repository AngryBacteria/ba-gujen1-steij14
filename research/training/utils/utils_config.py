import argparse
import json
import warnings

from pydantic import BaseModel
from typing import Optional


class ModelConfigCLM(BaseModel):
    id_model: str
    lower_precision: bool
    attention_implementation: str
    lora: bool
    qlora: bool
    galore: bool


class DataProcessingConfigCLM(BaseModel):
    sequence_length: int
    processing_threads: int


class TrainerConfigCLM(BaseModel):
    epochs: int
    batch_size: int
    optimizer: str
    learning_rate: float
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    use_reentrant: bool
    mixed_precision: bool


class GeneralConfigCLM(BaseModel):
    debug: bool
    wandb_logging: bool
    disable_annoying_warnings: bool
    run_name: Optional[str] = ""
    gpu: int
    save_model: bool
    logging_steps: int


class TrainConfigCLM(BaseModel):
    general: GeneralConfigCLM
    model: ModelConfigCLM
    data_processing: DataProcessingConfigCLM
    trainer: TrainerConfigCLM


def parse_clm_config():
    # Parse arguments (config file location)
    parser = argparse.ArgumentParser(
        description="Run the training script with a specified config file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="research/training/configs/clm_base.json",
        required=False,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    # Load config file
    with open(args.config, "r") as file:
        config_json = json.loads(file.read())
        config = TrainConfigCLM(**config_json)

    config = normalize_clm_config(config)
    return config


def normalize_clm_config(config: TrainConfigCLM):
    if config.model.qlora and not config.model.lora:
        raise ValueError("QLORA can only be used in combination with LORA.")
    if config.model.galore and (config.model.lora or config.model.qlora):
        raise ValueError("GALORE can not be used in combination with LORA or QLORA.")
    if config.model.galore and config.trainer.gradient_accumulation_steps != 1:
        print(
            "Gradient accumulation steps are not supported with GALORE. Disabling it."
        )
        config.trainer.gradient_accumulation_steps = 1

    if config.general.disable_annoying_warnings:
        warnings.filterwarnings(
            "ignore", category=UserWarning, module="torch.utils.checkpoint"
        )
        warnings.filterwarnings("ignore", category=FutureWarning, module="accelerate")

    return config
