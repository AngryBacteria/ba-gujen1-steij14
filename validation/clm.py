import datetime
import os
import re

import setproctitle

from shared.logger import logger
from shared.model_utils import (
    load_model_and_tokenizer,
    ModelPrecision,
    get_model_output_only,
    ChatTemplate, get_extractions_only,
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
setproctitle.setproctitle("gujen1 - bachelorthesis")

from datasets import load_dataset
import pandas as pd


def calculate_metrics_from_prompts(precision: ModelPrecision, model_name: str):
    tokenizer, model = load_model_and_tokenizer(model_name, precision)

    _dataset = load_dataset("json", data_files={"data": "prompts.jsonl"})[
        "data"
    ].train_test_split(test_size=0.1, shuffle=True, seed=42)
    test_data = _dataset["test"]
    test_date = datetime.datetime.now()

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
        _start_time = datetime.datetime.now()
        _inputs = tokenizer(instruction, return_tensors="pt").to("cuda:0")
        _outputs = model.generate(**_inputs, max_new_tokens=1000)
        output_string = tokenizer.decode(_outputs[0], skip_special_tokens=True)
        _end_time = datetime.datetime.now()
        execution_time = (_end_time - _start_time).microseconds

        # get extractions
        prediction_string = get_model_output_only(
            output_string, ChatTemplate.ALPACA_MISTRAL
        )
        if not prediction_string:
            logger.error("No extractions found")
            continue
        # TODO: replace with helper functions
        truth = get_extractions_only(truth_string)
        prediction = get_extractions_only(prediction_string)
        logger.debug(f"Truth         : {truth}")
        logger.debug(f"Prediction    : {prediction}")
        logger.debug(f"Execution time: {execution_time}")

        # calculate metrics
        metrics = calculate_extraction_validation_metrics(truth, prediction)
        logger.debug(f"Metrics   : {metrics}")

        logger.debug("-----------------------------------------")
        # TODO save params such as temp
        output.append(
            {
                "model": model_name,
                "model_precision": precision,
                "execution_time": execution_time,
                "date": test_date,
                "prompt": prompt,
                "instruction": instruction,
                "truth_string": truth_string,
                "truth": truth,
                "output_string": output_string,
                "prediction_string": prediction_string,
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


def calculate_extraction_validation_metrics(
    truth_labels: list[str], prediction_labels: list[str]
):
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