import os

import setproctitle

from research.training.utils.custom_callbacks import GPUMemoryUsageCallback

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = f"0"
setproctitle.setproctitle("gujen1 - bachelorthesis")

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
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
NUM_LABELS = 2
EPOCHS = 2
BATCH_SIZE = 64


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Load dataset
test_dataset = load_dataset("imdb", split="test[:5%]")
train_dataset = load_dataset("imdb", split="train[:40%]")
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased",
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
)

# Load metric
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred: EvalPrediction):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# Train
training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    report_to=["none"],
    logging_strategy="steps",
    logging_steps=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=test_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    custom_callbacks=[GPUMemoryUsageCallback(0, True)],
)

trainer.train()
