import os

import setproctitle

GPU = 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU}"
setproctitle.setproctitle("gujen1 - bachelorthesis")

from training.utils.config import get_steps_per_epoch
from training.utils.custom_callbacks import GPUMemoryUsageCallback
from training.utils.printing import (
    print_welcome_message,
    print_with_heading,
)
from training.utils.gpu import print_gpu_support
import numpy as np
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    AutoModelForSequenceClassification,
)
import evaluate
import pandas
from sklearn.metrics import f1_score, precision_score, recall_score

from datacorpus.aggregation.agg_bronco import aggregate_bronco_label_classification

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

# TODO ist das wirklich richtig so? mit spez. daten holen, idk
classification_docs, label2id, id2label = aggregate_bronco_label_classification()

# Number of labels
NUM_LABELS = len(label2id)


print_welcome_message()
print_gpu_support(f"{GPU}")


# Preprocess function for tokenizing the text and converting labels to the required format
def preprocess_function(examples):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding=True, max_length=512)
    labels = []
    for label_dict in examples["labels"]:
        label = [0] * NUM_LABELS
        for label_name, present in label_dict.items():
            if present:
                label[label2id[label_name]] = 1
        labels.append(label)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Load tokenizer
print_with_heading("Load fast tokenizer")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Load dataset from the classification documents
print_with_heading("Load dataset")

# Convert classification_docs to a Dataset object
dataset = Dataset.from_pandas(pandas.DataFrame(classification_docs))
dataset = dataset.train_test_split(test_size=TEST_SIZE)
dataset_tokenized = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load model
print_with_heading("Load model")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased",
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
)

# Load metrics
def compute_metrics(eval_pred: EvalPrediction):
    predictions, label_ids = eval_pred
    predictions = (predictions > 0.5).astype(int)
    f1 = f1_score(label_ids, predictions, average="weighted")
    precision = precision_score(label_ids, predictions, average="weighted")
    recall = recall_score(label_ids, predictions, average="weighted")
    return {"f1": f1, "precision": precision, "recall": recall}

# Training arguments
print_with_heading("Training arguments")
training_args = TrainingArguments(
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    optim="adamw_torch_fused",
    learning_rate=LEARNING_RATE,
    report_to=["none"],
    logging_strategy="steps",
    output_dir="bert_classification_model",
    save_strategy="epoch",
    save_total_limit=2,
    evaluation_strategy="steps",
)

EVAL_STEPS, LOGGING_STEPS = get_steps_per_epoch(
    len(dataset_tokenized["train"]),
    BATCH_SIZE,
    1,
    EVALS_PER_EPOCH,
    LOGS_PER_EPOCH,
)
training_args.eval_steps = EVAL_STEPS
training_args.logging_steps = LOGGING_STEPS

if DEBUG:
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

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_tokenized["train"],
    eval_dataset=dataset_tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=custom_callbacks,
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation: {eval_results}")

# Save the model if required
if SAVE_MODEL:
    trainer.save_model("bert_label_classification_model")