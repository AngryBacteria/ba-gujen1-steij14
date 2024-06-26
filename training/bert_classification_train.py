import os
import setproctitle

GPU = 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU}"
setproctitle.setproctitle("gujen1 - bachelorthesis")

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers.trainer_utils import HubStrategy

from datacorpus.aggregation.aggregate import get_all_classification_annotations
from shared.gpu_utils import print_cuda_support

from training.utils.config import get_steps_per_epoch
from training.utils.printing import (
    print_welcome_message,
    print_with_heading,
)
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
)

# Config
MODEL_NAME = "GerMedBERT/medbert-512"
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
TEST_SIZE = 0.1
DEBUG = True
WANDB = False
RUN_NAME = "GerMedBert_NORM_V01_BRONCO"
SAVE_MODEL = True
UPLOAD_MODEL = True
EVALS_PER_EPOCH = 4
LOGS_PER_EPOCH = 2

data, label2id, id2label, NUM_LABELS = get_all_classification_annotations()

print_welcome_message()
print_cuda_support()


def preprocess_function(examples):
    return tokenizer(examples["origin"], truncation=True, padding=True, max_length=512)


# Load tokenizer
print_with_heading("Load fast tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load dataset
print_with_heading("Load dataset")
dataset = Dataset.from_pandas(data).train_test_split(test_size=TEST_SIZE)
dataset_tokenized = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load model
print_with_heading("Load model")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
    problem_type="multi_label_classification",
)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = sigmoid(predictions)
    predictions = (predictions > 0.5).astype(int)

    f1 = f1_score(labels, predictions, average="micro", zero_division=1)
    precision = precision_score(labels, predictions, average="micro", zero_division=1)
    recall = recall_score(labels, predictions, average="micro", zero_division=1)
    accuracy = accuracy_score(labels, predictions)

    return {
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


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
    # logging
    report_to=["none"],
    logging_strategy="steps",
    # saving
    output_dir="bert_classification_model",
    save_strategy="epoch",
    save_total_limit=1,
    hub_private_repo=True,
    hub_strategy=HubStrategy.END,
    hub_model_id=f"BachelorThesis/{RUN_NAME}",
    # evaluation
    evaluation_strategy="steps",
)

# setup steps for logging and evaluation
EVAL_STEPS, LOGGING_STEPS = get_steps_per_epoch(
    len(dataset_tokenized["train"]), BATCH_SIZE, 1, EVALS_PER_EPOCH, LOGS_PER_EPOCH, 1
)
training_args.eval_steps = EVAL_STEPS
training_args.logging_steps = LOGGING_STEPS

if DEBUG:  # setup logging and debugging
    training_args.include_tokens_per_second = True
    training_args.include_num_input_tokens_seen = True
if WANDB:
    training_args.report_to = ["wandb"]
    os.environ["WANDB_PROJECT"] = "bachelor-thesis-testing"
    if RUN_NAME != "":
        training_args.run_name = RUN_NAME

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_tokenized["train"],
    eval_dataset=dataset_tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

eval_results = trainer.evaluate()
print(f"Evaluation: {eval_results}")

if SAVE_MODEL:
    trainer.save_model("bert_classification_model")

if UPLOAD_MODEL:
    trainer.push_to_hub()
