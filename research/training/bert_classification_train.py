import os

import setproctitle

GPU = 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU}"
setproctitle.setproctitle("gujen1 - bachelorthesis")

from research.training.utils.utils_config import get_steps_per_epoch
from research.training.utils.custom_callbacks import GPUMemoryUsageCallback
from research.training.utils.printing_utils import (
    print_welcome_message,
    print_with_heading,
)
from research.training.utils.utils_gpu import print_gpu_support
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    AutoModelForSequenceClassification,
)
import evaluate

# Config
EPOCHS = 2
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
TEST_SIZE = 0.2
DEBUG = True
WANDB = False
RUN_NAME = ""
SAVE_MODEL = False
EVALS_PER_EPOCH = 8
LOGS_PER_EPOCH = 2

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
NUM_LABELS = 2

print_welcome_message()
print_gpu_support(f"{GPU}")


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)


# Load tokenizer
print_with_heading("Load fast tokenizer")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Load dataset
print_with_heading("Load dataset")
imdb_dataset = load_dataset("imdb", split="train").train_test_split(test_size=TEST_SIZE)
imdb_dataset_tokenized = imdb_dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load model
print_with_heading("Load model")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased",
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
)

# Load metric
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred: EvalPrediction):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)


# Train
print_with_heading("Training arguments")
training_args = TrainingArguments(
    # training setup
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    # optimization setup
    optim="adamw_torch_fused",
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    # logging
    report_to=["none"],
    logging_strategy="steps",
    # saving
    output_dir="bert_classification_model",
    save_strategy="epoch",
    save_total_limit=2,
    # evaluation
    evaluation_strategy="steps",
)

# setup steps for logging and evaluation
EVAL_STEPS, LOGGING_STEPS = get_steps_per_epoch(
    len(imdb_dataset_tokenized["train"]),
    BATCH_SIZE,
    1,
    EVALS_PER_EPOCH,
    LOGS_PER_EPOCH,
)
training_args.eval_steps = EVAL_STEPS
training_args.logging_steps = LOGGING_STEPS

if DEBUG:  # setup logging and debugging
    training_args.include_tokens_per_second = True
    training_args.include_num_input_tokens_seen = True
    custom_callbacks = [GPUMemoryUsageCallback(GPU, LOGGING_STEPS)]
else:
    custom_callbacks = []
if WANDB:
    training_args.report_to = ["wandb"]
    os.environ["WANDB_PROJECT"] = "bachelor-thesis-testing"
    if RUN_NAME != "":
        training_args.run_name = RUN_NAME

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=imdb_dataset_tokenized["train"],
    eval_dataset=imdb_dataset_tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=custom_callbacks,
)

trainer.train()

eval_results = trainer.evaluate()
print(f"Evaluation: {eval_results}")


if SAVE_MODEL:
    trainer.save_model("bert_classification_model")
