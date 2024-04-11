import argparse
import json
import os
import setproctitle

from research.training.Definitions import TrainConfig

# load config
parser = argparse.ArgumentParser(
    description="Run the training script with a specified config file."
)
parser.add_argument(
    "--config",
    type=str,
    default="configs/base_config.json",
    required=False,
    help="Path to the configuration file.",
)
args = parser.parse_args()
config_path = args.config
with open(config_path, "r") as file:
    config_json = json.loads(file.read())
    config = TrainConfig(**config_json)

# set gpu environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
setproctitle.setproctitle("gujen1 - bachelorthesis")

import torch
import gc
import warnings
import wandb
from datasets import load_dataset
from peft import prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    MistralForCausalLM,
)
from transformers.training_args import OptimizerNames

# Pre-checks
if config.model.qlora and not config.model.lora:
    raise ValueError("QLORA can only be used in combination with LORA.")
if config.model.galore and (config.model.lora or config.model.qlora):
    raise ValueError("GALORE can not be used in combination with LORA or QLORA.")
if config.model.lora and config.trainer.gradient_checkpointing:
    print("Gradient checkpointing is not supported with LORA. Disabling it.")
    GRADIENT_CHECKPOINTING = False
if config.model.galore and config.trainer.gradient_accumulation_steps != 1:
    print("Gradient accumulation steps are not supported with GALORE. Disabling it.")
    GRADIENT_ACCUMULATION_STEPS = 1

# Setup
if config.general.disable_annoying_warnings:
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="torch.utils.checkpoint"
    )
    warnings.filterwarnings("ignore", category=FutureWarning, module="accelerate")

# Load model
if config.model.lower_precision:
    MODEL_PRECISION = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
else:
    MODEL_PRECISION = torch.float

if config.model.qlora and config.model.lora:
    # TODO: maybe implement LoftQ
    print(f"{15 * '='} Load QLora model {15 * '='}")
    _bnb_quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # theoretically better according to docs
        bnb_4bit_compute_dtype=torch.float,
    )
    model = MistralForCausalLM.from_pretrained(
        config.general.model_id,
        quantization_config=_bnb_quantization_config,
        torch_dtype=MODEL_PRECISION,
        attn_implementation=config.model.attention_implementation,
    )
    model = prepare_model_for_kbit_training(model)
else:
    print(f"{15 * '='} Load model {15 * '='}")
    model = MistralForCausalLM.from_pretrained(
        config.general.model_id,
        torch_dtype=MODEL_PRECISION,
        attn_implementation=config.model.attention_implementation,
    )

# Load tokenizer
print(f"{15 * '='} Load tokenizer {15 * '='}")
tokenizer = AutoTokenizer.from_pretrained(config.general.model_id, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Load and prepare dataset
def preprocess_function(examples):
    inputs = examples["text"]
    return tokenizer(
        inputs, padding=True, truncation=True, max_length=config.trainer.sequence_length
    )


print(f"{15 * '='} Load and prepare dataset {15 * '='}")
dataset = load_dataset(
    "tatsu-lab/alpaca",
    split="train[:100]",
    num_proc=config.data_processing.processing_threads,
)
_train_val_dataset = dataset.train_test_split(test_size=0.2, seed=42)
_train_dataset = _train_val_dataset["train"]
_val_dataset = _train_val_dataset["test"]

data_collator_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
tokenized_train_dataset = _train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=config.data_processing.processing_threads,
    remove_columns=["instruction", "input", "output", "text"],
)
tokenized_val_dataset = _val_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=config.data_processing.processing_threads,
    remove_columns=["instruction", "input", "output", "text"],
)

#  Init LORA
if config.model.lora:
    print(f"{15 * '='} Loading LORA {15 * '='}")
    from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

    _peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        # bias="none", # TODO: find out what this does
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    model = get_peft_model(model, _peft_config)
    model.print_trainable_parameters()

# Setup training arguments
print(f"{15 * '='} Train model {15 * '='}")
training_args = TrainingArguments(
    output_dir="my_awesome_new_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=config.trainer.epochs,
    report_to=["none"],
    logging_strategy="steps",
    logging_steps=2,
    # optimizations
    per_device_train_batch_size=config.trainer.batch_size,
    per_device_eval_batch_size=config.trainer.batch_size,
    gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
    gradient_checkpointing=config.trainer.gradient_checkpointing,
    optim=config.trainer.optimizer,
)
# Init GALORE
if config.model.galore:
    training_args.optim = OptimizerNames.GALORE_ADAMW
    training_args.optim_target_modules = (["attn", "mlp"],)
    training_args.optim_args = ("rank=1024, update_proj_gap=200, scale=2",)
if config.general.debug:
    training_args.include_tokens_per_second = True
    training_args.include_num_input_tokens_seen = True
if config.general.wandb_logging:
    training_args.report_to = ["wandb"]
    os.environ["WANDB_PROJECT"] = "bachelor-thesis-testing"
if config.general.run_name != "":
    training_args.run_name = config.general.run_name

# Train model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator_fn,
)
trainer.train()
trainer.save_model()
wandb.finish()

# cleanup
del model
del tokenizer
gc.collect()
torch.cuda.empty_cache()
