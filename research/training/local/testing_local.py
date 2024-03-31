from datasets import load_dataset
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

MODEl_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEBUG = True


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(
        f"GPU memory occupied: {info.used // 1024 ** 2} / {info.total // 1024 ** 2} MB"
    )


def preprocess_function(examples):
    inputs = examples["text"]
    return tokenizer(inputs, padding=True, truncation=True, max_length=512)


# Load model and tokenizer
print(f"{15 * '='} Load model and tokenizer {15 * '='}")
print_gpu_utilization()
tokenizer = AutoTokenizer.from_pretrained(MODEl_ID, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEl_ID)
print_gpu_utilization()


# Load and prepare dataset
print(f"{15 * '='} Load and prepare dataset {15 * '='}")
print_gpu_utilization()
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
print_gpu_utilization()


# Train model
print(f"{15 * '='} Train model {15 * '='}")
print_gpu_utilization()
training_args = TrainingArguments(
    output_dir="my_awesome_new_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    # optimizations
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    optim="adafactor",
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
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
