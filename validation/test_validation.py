import os
import re

import setproctitle
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
setproctitle.setproctitle("gujen1 - bachelorthesis")

from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd


def get_tokenizer(precision: int, model_name: str):
    # TODO: make usable for models without the LeoLm tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
    )
    if precision == 4:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            load_in_4bit=True,
        )
    elif precision == 8:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
        )
    elif precision == 16:
        model = AutoModelForCausalLM.from_pretrained(
            "BachelorThesis/Mistral_V03_BRONCO_CARDIO",
            torch_dtype=torch.bfloat16,
        )
        model.to("cuda:0")
    else:
        raise ValueError("Precision has to be 4, 8 or 16")

    return tokenizer, model


def calculate_metrics_from_prompts(precision: int, model_name: str):
    # TODO: make usable for models without the LeoLm tokenizer
    tokenizer, model = get_tokenizer(precision, model_name)

    _dataset = load_dataset("json", data_files={"data": "prompts.jsonl"})[
        "data"
    ].train_test_split(test_size=0.1, shuffle=True, seed=42)
    data = _dataset["test"]

    output = []
    for i, example in enumerate(data):
        # create the instruction for the model
        instruction = tokenizer.apply_chat_template(
            example["messages"][:-1], tokenize=False, add_generation_prompt=True
        )
        # get the whole input to save later
        prompt = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=True
        )
        print(prompt)
        # get the ground truth from the prompt
        truth = example["messages"][-1]["content"]

        # get the model output
        inputs = tokenizer(instruction, return_tensors="pt").to("cuda:0")
        outputs = model.generate(**inputs, max_new_tokens=1000)
        model_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = model_output.split("### Antwort:")[1].strip()

        # get extractions
        truth_extractions = get_extractions_only(truth)
        prediction_extractions = get_extractions_only(prediction)
        print(f"Truth     : {truth_extractions}")
        print(f"Prediction: {prediction_extractions}")

        # calculate metrics
        metrics = calculate_validation_metrics(
            truth_extractions, prediction_extractions
        )
        print(f"Metrics   : {metrics}")

        print("-----------------------------------------")
        # TODO save params such as temp
        output.append(
            {
                "model": model_name,
                "text": prompt,
                "instruction": instruction,
                "response": model_output,
                "truth": truth_extractions,
                "prediction": prediction_extractions,
                "precision": metrics[0],
                "recall": metrics[1],
                "f1_score": metrics[2],
                "task": example["task"],
                "type": example["type"],
                "source": example["source"],
            }
        )
    # convert to pandas df and save to json
    df = pd.DataFrame(output)
    df.to_json(f"validation_results_{precision}bit.json", orient="records")
    return output


def get_extractions_only(string_input: str):
    """
    Get all extractions from a string input. The string has to be in the typical form of a prompt output:
    extraction1 [atrribute1|attribute2] | extraction2 [attribute3|attribute4] | ...
    """
    string_input = string_input.strip().lower()
    annotations = string_input.split("|")
    extractions = [remove_brackets(x) for x in annotations]
    extractions = list(set(extractions))

    return extractions


def get_attributes_only(string_input: str):
    """
    Get all attributes from a string input. The string has to be in the typical form of a prompt output:
    extraction1 [atrribute1|attribute2] | extraction2 [attribute3|attribute4] | ...
    """
    string_input = string_input.strip().lower()
    annotations = string_input.split("|")
    attributes = [extract_from_brackets(x) for x in annotations]

    return attributes


def remove_brackets(string_input: str):
    """
    Remove substrings enclosed in brackets from a string.
    """
    pattern = r"\[[^]]*\]"
    cleaned_string = re.sub(pattern, "", string_input)
    cleaned_string = cleaned_string.strip()
    return cleaned_string


def extract_from_brackets(string_input: str):
    """
    Extract substrings enclosed in brackets from a string.
    """
    pattern = r"\[(.*?)\]"
    matches = re.findall(pattern, string_input)
    return matches


def calculate_validation_metrics(truth_labels: list[str], prediction_labels: list[str]):
    """
    Calculate precision, recall and f1 score for the extraction task. Takes in two lists, the truth labels and the
    predicted labels and returns the metrics.
    """
    # only take unique
    # TODO: discuss if this is cheating
    truth_set = set(truth_labels)
    prediction_set = set(prediction_labels)
    # make them lower case
    truth_set = {x.lower() for x in truth_set}
    prediction_set = {x.lower() for x in prediction_set}

    # calculate metrics
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


def aggregate_metrics(file_name):
    """TODO"""
    df = pd.read_json(file_name)
    # group by task
    grouped = df.groupby(["task", "source"])
    for name, group in grouped:
        print(name)
        print(f"Precision: {group["precision"].mean()}")
        print(f"Recall: {group["recall"].mean()}")
        print(f"F1 Score: {group["f1_score"].mean()}")
        print("----------------------------------------------------")


# calculate_metrics_from_prompts(4, "BachelorThesis/Mistral_V03_BRONCO_CARDIO")
# aggregate_metrics("validation_results_4bit.json")
