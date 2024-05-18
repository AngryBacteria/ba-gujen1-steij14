import argparse
import json
import warnings
from typing import Optional

from pydantic import BaseModel


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
    test_size: float


class TrainerConfigCLM(BaseModel):
    epochs: int
    batch_size: int
    evals_per_epoch: int
    optimizer: str
    learning_rate: float
    warmup_steps: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    mixed_precision: bool


class GeneralConfigCLM(BaseModel):
    debug: bool
    wandb_logging: bool
    disable_annoying_warnings: bool
    run_name: Optional[str] = ""
    gpu: str
    save_model: bool
    upload_model: bool
    output_dir: str
    save_steps: int
    logs_per_epoch: int


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

    if config.trainer.gradient_accumulation_steps < 1:
        raise ValueError(
            "Gradient accumulation steps must be at least 1. A value of 1 means it is disabled."
        )

    if config.trainer.batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    if config.general.logs_per_epoch < 1:
        raise ValueError("Logs per epoch must be at least 1.")

    if config.trainer.evals_per_epoch < 1:
        raise ValueError("Evals per epoch must be at least 1.")

    return config


def get_steps_per_epoch(
    training_examples: int,
    batch_size: int,
    gradient_accumulation: int,
    evals_per_epoch: int,
    logs_per_epoch: int,
    amount_of_gpus: int,
) -> tuple[int, int]:
    """
    Calculate the number of steps per epoch for evaluation and logging.
    :return:
    """
    _steps_per_epoch = max(
        1,
        round(
            training_examples
            / (batch_size * gradient_accumulation)
        ),
    )
    eval_steps = max(1, round(_steps_per_epoch / evals_per_epoch))
    logging_steps = max(1, round(_steps_per_epoch / logs_per_epoch))

    return eval_steps, logging_steps
