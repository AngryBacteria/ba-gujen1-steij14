import os

import setproctitle
from research.training.utils.utils_config import parse_clm_config

config = parse_clm_config()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{config.general.gpu}"
setproctitle.setproctitle("gujen1 - bachelorthesis")

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from research.training.utils.utils_config import get_steps_per_epoch
from research.training.utils.printing_utils import (
    print_welcome_message,
    print_with_heading,
)
from research.training.utils.utils_gpu import print_gpu_support
from research.training.utils.custom_callbacks import GPUMemoryUsageCallback
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    MistralForCausalLM,
)

# Welcome messages
print_welcome_message()
print_gpu_support(f"{config.general.gpu}")


# Model
if config.model.lower_precision:
    MODEL_PRECISION = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
else:
    MODEL_PRECISION = torch.float

if config.model.qlora and config.model.lora:
    from transformers import BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training

    print_with_heading("Load 4bit QLora model")
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
    model = prepare_model_for_kbit_training(model)
else:
    print_with_heading(f"Load model [{MODEL_PRECISION}]")
    model = MistralForCausalLM.from_pretrained(
        config.model.id_model,
        torch_dtype=MODEL_PRECISION,
        attn_implementation=config.model.attention_implementation,
    )
if config.model.lora:
    print_with_heading("Loading LORA [alpha 32, rank 8]")
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
print_with_heading("Load fast tokenizer")
tokenizer = AutoTokenizer.from_pretrained(config.model.id_model, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Dataset
def preprocess_function(examples):
    return tokenizer(
        examples["text"],
    )


print_with_heading("Load and prepare dataset")
_dataset = load_dataset("json", data_files={"data": "prompts.json"})[
    "data"
].train_test_split(test_size=config.data_processing.test_size, shuffle=True, seed=42)
tokenized_dataset = _dataset.map(
    preprocess_function,
    batched=True,
    num_proc=config.data_processing.processing_threads,
    remove_columns=["text", "type", "task", "source"],
)
print(
    f"Dataset length before: {len(tokenized_dataset['train']) + len(tokenized_dataset['test'])}"
)
tokenized_dataset = tokenized_dataset.filter(
    lambda example: len(example["input_ids"]) <= config.data_processing.sequence_length,
    num_proc=config.data_processing.processing_threads,
)
print(
    f"Dataset length after: {len(tokenized_dataset['train']) + len(tokenized_dataset['test'])}"
)

data_collator_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training setup
print_with_heading(
    f"Train model [optim {config.trainer.optimizer}, "
    f"Precision {MODEL_PRECISION}, "
    f"Mixed Training {config.trainer.mixed_precision}, "
    f"epochs {config.trainer.epochs}, batch {config.trainer.batch_size}, "
    f"accumulation {config.trainer.gradient_accumulation_steps}, "
    f"checkpointing {config.trainer.gradient_checkpointing}, "
    f"sequence {config.data_processing.sequence_length}] "
)

# TODO: weight decay and grad norm
training_args = TrainingArguments(
    # training setup
    num_train_epochs=config.trainer.epochs,
    per_device_train_batch_size=config.trainer.batch_size,
    per_device_eval_batch_size=config.trainer.batch_size,
    gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
    gradient_checkpointing=config.trainer.gradient_checkpointing,
    gradient_checkpointing_kwargs={
        "use_reentrant": False,
    },  # https://github.com/huggingface/transformers/issues/26969
    group_by_length=True,  # should speed up training
    # optimization setup
    optim=config.trainer.optimizer,
    learning_rate=config.trainer.learning_rate,
    lr_scheduler_type="cosine",  # axolotl does this
    warmup_steps=config.trainer.warmup_steps,
    # logging
    report_to=["none"],
    logging_strategy="steps",
    # saving
    output_dir=config.general.output_dir,
    # evaluation
    evaluation_strategy="steps",
)

# setup steps for logging and evaluation
EVAL_STEPS, LOGGING_STEPS = get_steps_per_epoch(
    len(tokenized_dataset["train"]),
    config.trainer.batch_size,
    config.trainer.gradient_accumulation_steps,
    config.trainer.evals_per_epoch,
    config.general.logs_per_epoch,
)
training_args.eval_steps = EVAL_STEPS
training_args.logging_steps = LOGGING_STEPS

if config.general.save_model:  # Setup saving
    # Bigger than one means save every x steps
    if config.general.save_steps > 0:
        training_args.save_steps = config.general.save_steps
        training_args.save_strategy = "steps"
        training_args.save_total_limit = 2
    # Smaller than one means save every epoch
    else:
        training_args.save_strategy = "epoch"
        training_args.save_total_limit = 2

if config.model.galore:  # Setup GaLore
    from transformers.training_args import OptimizerNames

    print_with_heading("Setup GaLore [rank 1024, proj_gap 200, scale 2]")
    training_args.optim = OptimizerNames.GALORE_ADAMW
    training_args.optim_target_modules = (["attn", "mlp"],)
    training_args.optim_args = ("rank=1024, update_proj_gap=200, scale=2",)

if config.general.debug:  # Setup logging and debugging
    training_args.include_tokens_per_second = True
    training_args.include_num_input_tokens_seen = True
    custom_callbacks = [GPUMemoryUsageCallback(config.general.gpu, LOGGING_STEPS)]
else:
    custom_callbacks = []

if config.general.wandb_logging:  # Setup wandb logging
    training_args.report_to = ["wandb"]
    os.environ["WANDB_PROJECT"] = "bachelor-thesis-testing"
    if config.general.run_name != "":
        training_args.run_name = config.general.run_name

if config.trainer.mixed_precision:  # Setup mixed precision training
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
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator_fn,
    callbacks=custom_callbacks,
)
trainer.train()
eval_results = trainer.evaluate()
print(f"Evaluation: {eval_results}")

if config.general.save_model:
    trainer.save_model(config.general.output_dir)

# cleanup
if config.general.wandb_logging:
    import wandb

    wandb.finish()
del model
del trainer
del tokenizer
del tokenized_dataset
torch.cuda.empty_cache()
