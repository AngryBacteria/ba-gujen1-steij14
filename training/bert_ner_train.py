import os

import setproctitle
from transformers.trainer_utils import HubStrategy

GPU = 2
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU}"
setproctitle.setproctitle("gujen1 - bachelorthesis")

from training.utils.config import get_steps_per_epoch
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)
from training.utils.custom_callbacks import GPUMemoryUsageCallback
from training.utils.printing import (
    print_welcome_message,
    print_with_heading,
)
from training.utils.gpu import print_gpu_support

EPOCHS = 7
BATCH_SIZE = 16
LEARNING_RATE = 4e-4
DEBUG = True
WANDB = True
RUN_NAME = "GELECTRA_NER_BRONCO_CARDIO_V01"
SAVE_MODEL = True
UPLOAD_MODEL = True
EVALS_PER_EPOCH = 4
LOGS_PER_EPOCH = 16
MODEL = "deepset/gelectra-large"

ID2LABEL = {
    0: "O",
    1: "B-MED",
    2: "I-MED",
    3: "B-TREAT",
    4: "I-TREAT",
    5: "B-DIAG",
    6: "I-DIAG",
}
LABEL2ID = {
    "O": 0,
    "B-MED": 1,
    "I-MED": 2,
    "B-TREAT": 3,
    "I-TREAT": 4,
    "B-DIAG": 5,
    "I-DIAG": 6,
}
LABEL_LIST = [
    "O",
    "B-MED",
    "I-MED",
    "B-TREAT",
    "I-TREAT",
    "B-DIAG",
    "I-DIAG",
]

print_welcome_message()
print_gpu_support(f"{GPU}")

# Possible modesl
# FacebookAI/xlm-roberta-large
# deepset/gelectra-large
# GerMedBERT/medbert-512
# SciBERT
# BioBert
# MedBert
# ClinicalBert
# PubMedBert

# Load tokenizer and model
print_with_heading("Load fast tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
print_with_heading("Load fast tokenizer")
model = AutoModelForTokenClassification.from_pretrained(
    MODEL,
    num_labels=7,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

# Load Data
print_with_heading("Load data")
dataset = load_dataset("json", data_files={"data": "ner.jsonl"})[
    "data"
].train_test_split(test_size=0.15, shuffle=True, seed=42)
seqeval = evaluate.load("seqeval")


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["words"], truncation=True, is_split_into_words=True, max_length=512
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(
            batch_index=i
        )  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif (
                word_idx != previous_word_idx
            ):  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p: EvalPrediction):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return seqeval.compute(predictions=true_predictions, references=true_labels)


tokenized_dataset = dataset.map(
    tokenize_and_align_labels, batched=True, remove_columns=["source"]
)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# TODO: weight decay and grad norm, warmup ratio verstehen und anwenden
print_with_heading("Train model")
training_args = TrainingArguments(
    # training setup
    output_dir=RUN_NAME,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    # optimization setup
    optim="adamw_torch_fused",
    learning_rate=LEARNING_RATE,
    # logging
    report_to=["none"],
    logging_strategy="steps",
    # saving
    save_strategy="epoch",
    save_total_limit=2,
    hub_private_repo=True,
    hub_strategy=HubStrategy.END,
    hub_model_id=f"BachelorThesis/{RUN_NAME}",
    # evaluation
    evaluation_strategy="steps",
)

# setup steps for logging and evaluation
EVAL_STEPS, LOGGING_STEPS = get_steps_per_epoch(
    len(tokenized_dataset["train"]),
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
    training_args.run_name = RUN_NAME


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=custom_callbacks,
)

trainer.train()

eval_results = trainer.evaluate()
print(f"Evaluation: {eval_results}")

if SAVE_MODEL:
    trainer.save_model(RUN_NAME)

if UPLOAD_MODEL:
    trainer.push_to_hub()
