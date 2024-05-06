import os

import setproctitle
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
setproctitle.setproctitle("gujen1 - bachelorthesis")

from datasets import load_dataset

from shared.model_utils import get_tokenizer_with_template, patch_model
from transformers import AutoModelForCausalLM


def test_with_file():
    tokenizer = get_tokenizer_with_template(
        tokenizer_name="LeoLM/leo-mistral-hessianai-7b"
    )
    model = AutoModelForCausalLM.from_pretrained(
        "mistral_instruction_low_precision",
        torch_dtype=torch.bfloat16,
        # load_in_8bit=True,
        # load_in_4bit=True,
    )
    model = patch_model(model, tokenizer).to("cuda:0")

    _dataset = load_dataset("json", data_files={"data": "prompts.jsonl"})[
        "data"
    ].train_test_split(test_size=0.1, shuffle=True, seed=42)
    data = _dataset["test"]
    for i, example in enumerate(data):
        # remove first element from messages array (we dont want asisstant response)
        messages = example["messages"][:-1]
        only_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(only_prompt, return_tensors="pt").to("cuda:0")
        outputs = model.generate(**inputs, max_length=512)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        print("-----------------------------------------")


def test_metrics():
    _dataset = load_dataset("json", data_files={"data": "prompts.jsonl"})[
        "data"
    ].train_test_split(test_size=0.1, shuffle=True, seed=42)

    data = _dataset["test"]
    for i, example in enumerate(data):
        message = example["messages"][-1]
        message = message["content"].strip().lower()
        annotations = message.split("|")
        print(calculate_extraction_metrics(annotations, annotations))


def calculate_extraction_metrics(truth_labels: list[str], prediction_labels: list[str]):
    """
    Calculate precision, recall and f1 score for the extraction task. Takes in two lists, the truth labels and the
    predicted labels and returns the metrics.
    """
    truth_set = set(truth_labels)
    prediction_set = set(prediction_labels)

    true_positives = len(truth_set & prediction_set)
    false_positives = len(prediction_set - truth_set)
    false_negatives = len(truth_set - prediction_set)

    if true_positives == 0:
        precision = 0.0
        recall = 0.0
    else:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)

    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score
