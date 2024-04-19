import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, Trainer, \
    TrainingArguments
import evaluate


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Load dataset
imdb = load_dataset("imdb")
tokenized_imdb = imdb.map(preprocess_function, batched=True)
tokenized_imdb = {
    "train": tokenized_imdb["train"].select(range(1000)),
    "test": tokenized_imdb["test"].select(range(100)),
}
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# Train
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    evaluation_strategy="epoch",
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
