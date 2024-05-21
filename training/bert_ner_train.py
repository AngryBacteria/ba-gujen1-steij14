import os

import setproctitle

GPU = 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU}"
setproctitle.setproctitle("gujen1 - bachelorthesis")

from transformers.trainer_utils import HubStrategy
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
from training.utils.gpu import print_cuda_support

EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
DEBUG = True
WANDB = True
RUN_NAME = "GerMedBERT_NER_BRONCO_CARDIO_V02"
SAVE_MODEL = True
UPLOAD_MODEL = True
EVALS_PER_EPOCH = 2
LOGS_PER_EPOCH = 16
MODEL = "GerMedBERT/medbert-512"

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
print_cuda_support(f"{GPU}")

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


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)
    return new_labels


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["words"], truncation=True, is_split_into_words=True, max_length=512
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
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
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return results


tokenized_dataset = dataset.map(
    tokenize_and_align_labels, batched=True, remove_columns=["source"]
)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

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
    weight_decay=0.01,
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
    len(tokenized_dataset["train"]), BATCH_SIZE, 1, EVALS_PER_EPOCH, LOGS_PER_EPOCH, 1
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
