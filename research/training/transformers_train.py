import gc
import os
import warnings
import torch
import wandb
from datasets import load_dataset
import setproctitle
from peft import prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)

# Variables General
GPU_ID = 0
MODEl_ID = (
    "mistralai/Mistral-7B-Instruct-v0.2"  # microsoft/phi-1_5, mistralai/Mistral-7B-v0.1
)
DEBUG = True
WANDB_LOGGING = True  # First you have to login with "wandb login"
SETUP_ENVIRONMENT = True
DISABLE_ANNOYING_WARNINGS = True
RUN_NAME = "test_qlora1"

# Variables Model
MODEL_PRECISION = (
    torch.float
)  # Lower makes training faster, but can also lead to convergence problems. Possible values: torch.float16, torch.bfloat16, torch.float
ATTENTION_IMPLEMENTATION = "sdpa"  # sdpa, eager, flash_attention_2

# LORA / QLORA
LORA = True
QLORA = True

# Variables Data processing
PROCESSING_THREADS = 4

# Variables Trainer
SEQUENCE_LENGTH = 512
EPOCHS = 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = (
    4  # 1 to disable. Should be proportional to the batch size. Reduces VRAM usage.
)
GRADIENT_CHECKPOINTING = True
OPTIMIZER = (
    "adamw_torch_fused"  # adamw_bnb_8bit, adamw_torch, adafactor, adamw_torch_fused
)

# Pre-checks
if QLORA and not LORA:
    raise ValueError("QLORA can only be used in combination with LORA.")
if LORA and GRADIENT_CHECKPOINTING:
    print("Gradient checkpointing is not supported with LORA. Disabling it.")
    GRADIENT_CHECKPOINTING = False

# Setup
if SETUP_ENVIRONMENT:
    torch.cuda.set_device(GPU_ID)
    setproctitle.setproctitle("gujen1 - bachelorthesis")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
if DISABLE_ANNOYING_WARNINGS:
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="torch.utils.checkpoint"
    )
    warnings.filterwarnings("ignore", category=FutureWarning, module="accelerate")

# Load model
if QLORA and LORA:
    # TODO: maybe implement LoftQ
    print(f"{15 * '='} Load QLora model {15 * '='}")
    bnb_quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # theoretically better according to docs
        bnb_4bit_compute_dtype=torch.float,  # less resources
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEl_ID,
        quantization_config=bnb_quantization_config,
        torch_dtype=MODEL_PRECISION,
        attn_implementation=ATTENTION_IMPLEMENTATION,
        device_map=GPU_ID,
    )
    model = prepare_model_for_kbit_training(model)
else:
    print(f"{15 * '='} Load model {15 * '='}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEl_ID,
        torch_dtype=MODEL_PRECISION,
        attn_implementation=ATTENTION_IMPLEMENTATION,
        device_map=GPU_ID,
    )

# Load tokenizer
print(f"{15 * '='} Load tokenizer {15 * '='}")
tokenizer = AutoTokenizer.from_pretrained(MODEl_ID, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token


# Load and prepare dataset
def preprocess_function(examples):
    inputs = examples["text"]
    return tokenizer(inputs, padding=True, truncation=True, max_length=SEQUENCE_LENGTH)


print(f"{15 * '='} Load and prepare dataset {15 * '='}")
dataset = load_dataset(
    "tatsu-lab/alpaca", split="train[:100]", num_proc=PROCESSING_THREADS
)
train_val_dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_val_dataset["train"]
val_dataset = train_val_dataset["test"]

data_collator_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
tokenized_train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=PROCESSING_THREADS,
    remove_columns=["instruction", "input", "output", "text"],
)
tokenized_val_dataset = val_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=PROCESSING_THREADS,
    remove_columns=["instruction", "input", "output", "text"],
)

#  Load LORA
if LORA:
    print(f"{15 * '='} Loading LORA {15 * '='}")
    from peft import (
        get_peft_model,
        LoraConfig,
        prepare_model_for_kbit_training,
    )

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        # bias="none", # TODO: find out what this does
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

# Setup training arguments
print(f"{15 * '='} Train model {15 * '='}")
training_args = TrainingArguments(
    output_dir="my_awesome_new_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=EPOCHS,
    report_to=["none"],
    logging_strategy="steps",
    logging_steps=2,
    # optimizations
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    gradient_checkpointing=GRADIENT_CHECKPOINTING,
    optim=OPTIMIZER,
)
if DEBUG:
    training_args.include_tokens_per_second = True
    training_args.include_num_input_tokens_seen = True
if WANDB_LOGGING:
    training_args.report_to = ["wandb"]
    os.environ["WANDB_PROJECT"] = "bachelor-thesis-testing"
if RUN_NAME != "":
    training_args.run_name = RUN_NAME

# Train model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator_fn,
)
trainer.train()
wandb.finish()

# cleanup
del model
del tokenizer
gc.collect()
torch.cuda.empty_cache()
