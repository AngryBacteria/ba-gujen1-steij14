import os

import setproctitle

GPU = 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU}"
setproctitle.setproctitle("gujen1 - bachelorthesis")

from research.training.utils.utils_config import get_steps_per_epoch
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
from research.training.utils.custom_callbacks import GPUMemoryUsageCallback
from research.training.utils.printing_utils import (
    print_welcome_message,
    print_with_heading,
)
from research.training.utils.utils_gpu import print_gpu_support

EPOCHS = 2
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
DEBUG = True
WANDB = False
RUN_NAME = ""
SAVE_MODEL = False
EVALS_PER_EPOCH = 2
LOGS_PER_EPOCH = 2

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
    "O": 1,
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

# Load tokenizer and model
print_with_heading("Load fast tokenizer")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
print_with_heading("Load fast tokenizer")
model = AutoModelForTokenClassification.from_pretrained(
    "google-bert/bert-base-uncased",
    num_labels=7,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

# Load Data
print_with_heading("Load data")
dataset = load_dataset("json", data_files={"data": "ner.json"})[
    "data"
].train_test_split(test_size=0.1, shuffle=True, seed=42)
seqeval = evaluate.load("seqeval")


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["words"], truncation=True, is_split_into_words=True
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

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


tokenized_dataset = dataset.map(
    tokenize_and_align_labels, batched=True, remove_columns=["source"]
)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

print_with_heading("Train model")
training_args = TrainingArguments(
    # training setup
    output_dir="bert_ner_model",
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
    if RUN_NAME != "":
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
    trainer.save_model("bert_ner_model")
