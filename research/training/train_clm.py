import os
import setproctitle

from research.training.utils.utils_config_parser import parse_training_config

config = parse_training_config()
# Setup gpu environment (needs to happen before importing huggingface library)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{config.general.gpu}"
os.environ["TOKENIZERS_PARALLELISM"] = (
    "false"  # https://github.com/huggingface/transformers/issues/5486
)
setproctitle.setproctitle("gujen1 - bachelorthesis")

import torch

# The server is new enough to support the TF32 data type
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from research.training.custom_callbacks import GPUMemoryUsageCallback
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

# MODEL
if config.model.lower_precision:
    MODEL_PRECISION = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
else:
    MODEL_PRECISION = torch.float

if config.model.qlora and config.model.lora:
    print(f"{30 * '='} Load 4bit QLora model {30 * '='}")
    _bnb_quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # theoretically better according to docs
        bnb_4bit_compute_dtype=MODEL_PRECISION,
        bnb_4bit_quant_storage=torch.bfloat16,  # axolotl uses this
    )
    model = MistralForCausalLM.from_pretrained(
        config.model.id_model,
        quantization_config=_bnb_quantization_config,
        attn_implementation=config.model.attention_implementation,
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.trainer.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": config.trainer.use_reentrant},
    )
else:
    print(f"{30 * '='} Load model [{MODEL_PRECISION}] {30 * '='}")
    model = MistralForCausalLM.from_pretrained(
        config.model.id_model,
        torch_dtype=MODEL_PRECISION,
        attn_implementation=config.model.attention_implementation,
    )
if config.model.lora:
    print(f"{30 * '='} Loading LORA [alpha 32, rank 8] {30 * '='}")
    from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

    _peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "gate_proj",
            "down_proj",
            "up_proj",
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
        ],
    )
    model = get_peft_model(model, _peft_config)
    model.print_trainable_parameters()

# Tokenizer
print(f"{30 * '='} Load fast tokenizer {30 * '='}")
tokenizer = AutoTokenizer.from_pretrained(config.model.id_model, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Dataset
# TODO: maybe remove the data collator does already padding
def preprocess_function(examples):
    inputs = [
        text * 100 for text in examples["text"]
    ]  # makes sure that whole context is used
    return tokenizer(
        inputs,
        padding=True,
        truncation=True,
        max_length=config.data_processing.sequence_length,
    )


print(f"{30 * '='} Load and prepare dataset {30 * '='}")
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

# Training arguments
print(
    f"{30 * '='} "
    f"Train model [optim {config.trainer.optimizer}, "
    f"Precision {MODEL_PRECISION}, "
    f"Mixed Training {config.trainer.mixed_precision}, "
    f"epochs {config.trainer.epochs}, batch {config.trainer.batch_size}, "
    f"accumulation {config.trainer.gradient_accumulation_steps}, "
    f"checkpointing {config.trainer.gradient_checkpointing}, "
    f"sequence {config.data_processing.sequence_length}] "
    f"{30 * '='}"
)
training_args = TrainingArguments(
    output_dir="my_awesome_new_model",
    learning_rate=config.trainer.learning_rate,
    num_train_epochs=config.trainer.epochs,
    report_to=["none"],
    logging_strategy="steps",
    logging_steps=1,
    lr_scheduler_type="cosine",  # axolotl does this
    # optimizations
    per_device_train_batch_size=config.trainer.batch_size,
    per_device_eval_batch_size=config.trainer.batch_size,
    gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
    gradient_checkpointing=config.trainer.gradient_checkpointing,
    gradient_checkpointing_kwargs={
        "use_reentrant": config.trainer.use_reentrant
    },  # https://github.com/huggingface/transformers/issues/26969
    optim=config.trainer.optimizer,
)

if config.trainer.eval_steps > 0:
    training_args.evaluation_strategy = "steps"
    training_args.eval_steps = config.trainer.eval_steps
else:
    training_args.evaluation_strategy = "epoch"

if config.model.galore:  # setup GaLore
    print(f"{30 * '='} Setup GaLore [rank 1024, proj_gap 200, scale 2] {30 * '='}")
    training_args.optim = OptimizerNames.GALORE_ADAMW
    training_args.optim_target_modules = (["attn", "mlp"],)
    training_args.optim_args = ("rank=1024, update_proj_gap=200, scale=2",)

if (
    config.trainer.gradient_checkpointing
):  # re-enable gradient checkpointing just to be sure :)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": config.trainer.use_reentrant}
    )

if config.general.debug:  # setup logging and debugging
    training_args.include_tokens_per_second = True
    training_args.include_num_input_tokens_seen = True
    custom_callbacks = [GPUMemoryUsageCallback(config.general.gpu, True)]
else:
    custom_callbacks = []
if config.general.wandb_logging:
    training_args.report_to = ["wandb"]
    os.environ["WANDB_PROJECT"] = "bachelor-thesis-testing"
    if config.general.run_name != "":
        training_args.run_name = config.general.run_name

if config.trainer.mixed_precision:  # setup mixed precision training
    if MODEL_PRECISION == torch.float16:
        training_args.fp16 = True
        training_args.fp16_full_eval = True
    if MODEL_PRECISION == torch.bfloat16:
        training_args.bf16 = True
        training_args.bf16_full_eval = True
    if MODEL_PRECISION == torch.float:
        training_args.tf32 = True

# Training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator_fn,
    callbacks=custom_callbacks,
)
trainer.train()
if config.general.save_model:
    trainer.save_model()

# cleanup
wandb.finish()
del model
del trainer
del tokenizer
del tokenized_val_dataset
del tokenized_train_dataset
del _train_dataset
del _val_dataset
torch.cuda.empty_cache()
