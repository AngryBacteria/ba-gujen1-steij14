import os

import setproctitle

GPU = 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU}"
setproctitle.setproctitle("gujen1 - bachelorthesis")

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
)
import math
from research.training.utils.custom_callbacks import GPUMemoryUsageCallback
from research.training.utils.printing_utils import (
    print_welcome_message,
    print_with_heading,
)
from research.training.utils.utils_gpu import print_gpu_support

EPOCHS = 3
BATCH_SIZE = 8
LOGGING_STEPS = 200
DEBUG = True
WANDB = False
RUN_NAME = ""
SAVE_MODEL = False
BLOCK_SIZE = 384

# Welcome messages
print_welcome_message()
print_gpu_support(f"{GPU}")

# Model
print_with_heading("Load model")
model = AutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base")

# Tokenizer
print_with_heading("Load tokenizer")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")
tokenizer.pad_token = tokenizer.eos_token


# Dataset
def tokenize_function(examples):
    result = tokenizer(examples["text"])
    return result


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= BLOCK_SIZE:
        total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated_examples.items()
    }
    return result


print_with_heading("Load dataset")
_dataset = load_dataset("json", data_files={"data": "pretrain.json"})[
    "data"
].train_test_split(test_size=0.1, shuffle=True, seed=42)
_dataset_tokenizer = _dataset.map(
    tokenize_function, batched=True, remove_columns=["text", "task", "source"]
)
dataset_mlm_train = _dataset_tokenizer["train"].map(group_texts, batched=True)
dataset_mlm_test = _dataset_tokenizer["test"].map(group_texts, batched=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm_probability=0.15, mlm=True
)


# Setup training
print_with_heading("Train model")
training_args = TrainingArguments(
    output_dir="bert_mlm_model",
    optim="adamw_torch_fused",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=2e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    report_to=["none"],
    logging_strategy="steps",
    logging_steps=LOGGING_STEPS,
)

if DEBUG:  # setup logging and debugging
    training_args.include_tokens_per_second = True
    training_args.include_num_input_tokens_seen = True
    custom_callbacks = [GPUMemoryUsageCallback(GPU, 16)]
else:
    custom_callbacks = []
if WANDB:
    training_args.report_to = ["wandb"]
    os.environ["WANDB_PROJECT"] = "bachelor-thesis-testing"
    if RUN_NAME != "":
        training_args.run_name = RUN_NAME

# Train model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_mlm_train,
    eval_dataset=dataset_mlm_test,
    data_collator=data_collator,
    callbacks=custom_callbacks,
)

trainer.train()
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
