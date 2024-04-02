import os

import torch
from datasets import load_dataset
import setproctitle
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

# Variables
GPU_ID = 0
MODEl_ID = "mistralai/Mistral-7B-v0.1"
DEBUG = True
SEQUENCE_LENGTH = 512
BATCH_SIZE = 1
OPTIMIZER = "adamw_bnb_8bit"  # adamw_bnb_8bit, adamw_torch, adafactor
LOAD_LOWER_PRECISION = True

# Setup
setproctitle.setproctitle("gujen1 - ba-mistralai - testing.py")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"


def preprocess_function(examples):
    inputs = examples["text"]
    return tokenizer(inputs, padding=True, truncation=True, max_length=SEQUENCE_LENGTH)


# Load model and tokenizer
print(f"{15 * '='} Load model and tokenizer {15 * '='}")
tokenizer = AutoTokenizer.from_pretrained(MODEl_ID, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

if LOAD_LOWER_PRECISION:
    model = AutoModelForCausalLM.from_pretrained(
        MODEl_ID,
        attn_implementation="sdpa",
        torch_dtype=torch.float16,
    ).to("cuda")
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEl_ID,
        attn_implementation="sdpa",
    ).to("cuda")

# Load and prepare dataset
print(f"{15 * '='} Load and prepare dataset {15 * '='}")
dataset = load_dataset("tatsu-lab/alpaca", split="train[:100]")
train_val_dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_val_dataset["train"]
val_dataset = train_val_dataset["test"]

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
tokenized_train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=["instruction", "input", "output", "text"],
)
tokenized_val_dataset = val_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=["instruction", "input", "output", "text"],
)

# Train model
print(f"{15 * '='} Train model {15 * '='}")
training_args = TrainingArguments(
    output_dir="my_awesome_new_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    # optimizations
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=BATCH_SIZE * 4,
    gradient_checkpointing=True,
    optim=OPTIMIZER,
)
if DEBUG:
    training_args.include_tokens_per_second = True
    training_args.include_num_input_tokens_seen = True

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator,
)
trainer.train()
