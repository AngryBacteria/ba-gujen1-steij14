import datetime
import os
import re

import setproctitle

from shared.logger import logger
from shared.model_utils import (
    load_model_and_tokenizer,
    ModelPrecision,
    parse_model_output_only,
    ChatTemplate,
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
setproctitle.setproctitle("gujen1 - bachelorthesis")

from datasets import load_dataset
import pandas as pd


def calculate_metrics_from_prompts(precision: ModelPrecision, model_name: str):
    # TODO: make usable for models without the LeoLm tokenizer
    tokenizer, model = load_model_and_tokenizer(model_name, precision)

    _dataset = load_dataset("json", data_files={"data": "prompts.jsonl"})[
        "data"
    ].train_test_split(test_size=0.1, shuffle=True, seed=42)
    test_data = _dataset["test"]

    output = []
    for i, example in enumerate(test_data):
        # create the model_instruction for the model
        instruction = tokenizer.apply_chat_template(
            example["messages"][:-1], tokenize=False, add_generation_prompt=True
        )
        if instruction is None:
            logger.error("No instruction found")
            continue

        # get the whole input to save later
        prompt = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=True
        )
        logger.debug(prompt)
        # get the ground truth_string from the full_prompt
        truth_string = example["messages"][-1]["content"]
        if truth_string is None:
            logger.error("No truth string found")
            continue

        # get the model output
        start_time = datetime.datetime.now()
        _inputs = tokenizer(instruction, return_tensors="pt").to("cuda:0")
        _outputs = model.generate(**_inputs, max_new_tokens=1000)
        output_string = tokenizer.decode(_outputs[0], skip_special_tokens=True)
        end_time = datetime.datetime.now()
        execution_time = end_time - start_time
        prediction_string = parse_model_output_only(
            output_string, ChatTemplate.ALPACA_MISTRAL
        )
        if not prediction_string:
            logger.error("No extractions found")
            continue

        # get extractions
        # TODO: replace with helper functions
        truth = get_extractions_only(truth_string)
        prediction = get_extractions_only(prediction_string)
        logger.debug(f"Truth         : {truth}")
        logger.debug(f"Prediction    : {prediction}")
        logger.debug(f"Execution time: {execution_time.microseconds}")

        # calculate metrics
        metrics = calculate_validation_metrics(truth, prediction)
        logger.debug(f"Metrics   : {metrics}")

        logger.debug("-----------------------------------------")
        # TODO save params such as temp
        output.append(
            {
                "model": model_name,
                "model_precision": precision,
                "execution_time": execution_time.microseconds,
                "prompt": prompt,
                "instruction": instruction,
                "output_string": output_string,
                "truth_string": truth_string,
                "truth": truth,
                "prediction": prediction,
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


def remove_brackets(string_input: str):
    """
    Remove substrings enclosed in brackets from a string.
    """
    pattern = r"\[[^]]*\]"
    cleaned_string = re.sub(pattern, "", string_input)
    cleaned_string = cleaned_string.strip()
    return cleaned_string


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
        logger.debug(name)
        logger.debug(f"Precision: {group["precision"].mean()}")
        logger.debug(f"Recall: {group["recall"].mean()}")
        logger.debug(f"F1 Score: {group["f1_score"].mean()}")
        logger.debug("----------------------------------------------------")


calculate_metrics_from_prompts(
    ModelPrecision.FOUR_BIT, "BachelorThesis/Mistral_V03_BRONCO_CARDIO"
)
