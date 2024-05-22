import datetime
import os

import evaluate
import setproctitle

from shared.logger import logger
from shared.model_utils import (
    load_model_and_tokenizer,
    ModelPrecision,
    get_model_output_only,
    ChatTemplate,
    get_extractions_without_attributes,
    get_extractions_with_attributes,
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
setproctitle.setproctitle("gujen1 - bachelorthesis")

from datasets import load_dataset
import pandas as pd


def calculate_metrics_from_prompts(
    precision: ModelPrecision,
    model_name: str,
    trained_sequence_length: int,
    tasks_to_eval=None,
):
    """
    Calculates the metrics precision, recall and f1 score for the extraction task. The model makes a precision which
    is then compared to the ground truth (the full prompt with answer). The model is evaluated on the prompts.jsonl
    data. The results are saved to a file. The extraction/normalization task is evaluated with and without attributes.
    Right now the attribute metrics are not saved in an intelligent way, this should be done (TODO: do it).
    :param tasks_to_eval: The tasks that should be evaluated
    :param precision: The precision to load the model in (16bit recommended if hardware supports it)
    :param model_name: The name of the model to load
    :param trained_sequence_length: The sequence length the model was trained on
    :return: None the data is saved to a file
    """
    if tasks_to_eval is None:
        tasks_to_eval = ["extraction", "normalization", "summary", "catalog"]

    tokenizer, model = load_model_and_tokenizer(model_name, precision)
    model_name = model_name.split("\\")[-1]
    _dataset = load_dataset("json", data_files={"data": "prompts.jsonl"})[
        "data"
    ].train_test_split(test_size=0.1, shuffle=True, seed=42)
    _test_data = _dataset["test"]
    test_date = datetime.datetime.now()

    output = []
    for i, example in enumerate(_test_data):
        # skip examples not in the right task
        if example["task"] not in tasks_to_eval:
            logger.warning(
                f"Skipping example {i} with task {example['task']}, because task is not supported yet"
            )
            continue

        # skip examples that are too long
        _tokenized = tokenizer.apply_chat_template(
            example["messages"], tokenize=True, add_generation_prompt=True
        )
        if len(_tokenized) >= trained_sequence_length:
            logger.warning(
                f"Skipping example {i} with length {len(_tokenized)}, because it is too long"
            )
            continue

        # create the model_instruction for the model
        instruction: str = tokenizer.apply_chat_template(
            example["messages"][:-1], tokenize=False, add_generation_prompt=True
        )
        if instruction is None or instruction == "":
            logger.error("No instruction found")
            continue

        # get the whole input to save later
        prompt: str = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=True
        )
        if prompt is None or prompt == "":
            logger.error("No prompt found")
            continue
        logger.debug(f"Prompt: {prompt}")

        # get the ground truth_string from the full_prompt
        truth_string: str = example["messages"][-1]["content"]
        if truth_string is None or truth_string == "":
            logger.error("No truth string found")
            continue

        # get the model output
        _start_time = datetime.datetime.now()
        _inputs = tokenizer(instruction, return_tensors="pt").to("cuda:0")
        _outputs = model.generate(**_inputs, max_new_tokens=9999)
        output_string = tokenizer.decode(_outputs[0], skip_special_tokens=True)
        output_string_raw = tokenizer.decode(_outputs[0], skip_special_tokens=False)
        _end_time = datetime.datetime.now()
        execution_time = (_end_time - _start_time).microseconds

        # get model prediction
        prediction_string = get_model_output_only(
            output_string, ChatTemplate.ALPACA_MISTRAL, lower=False
        )
        if not prediction_string or prediction_string == "":
            logger.error("No extractions found")
            continue

        # SUMMARIZATION
        if example["task"] == "summary":
            truth = truth_string.strip().lower()
            metrics = calculate_rogue_metrics(prediction_string, truth)
            logger.debug(f"Truth (summary)      : {truth_string}")
            logger.debug(f"Prediction (summary) : {prediction_string}")
            logger.debug(f"Metrics (summary)    : {metrics}")

            output.append(
                {
                    "model": model_name,
                    "model_precision": precision.value,
                    "execution_time": execution_time,
                    "date": test_date,
                    "prompt": prompt,
                    "instruction": instruction,
                    "truth_string": truth_string,
                    "truth": truth,
                    "output_string": output_string,
                    "output_string_raw": output_string_raw,
                    "prediction_string": prediction_string,
                    "prediction": prediction_string,
                    "rouge1": metrics["rouge1"],
                    "rouge2": metrics["rouge2"],
                    "rougeL": metrics["rougeL"],
                    "rougeLsum": metrics["rougeLsum"],
                    "task": example["task"],
                    "type": example["type"],
                    "source": example["source"],
                    "na_prompt": example["na_prompt"],
                }
            )

        # EXTRACTION AND NORMALIZATION
        if example["task"] == "extraction" or example["task"] == "normalization":
            truth = get_extractions_without_attributes(truth_string)
            prediction = get_extractions_without_attributes(prediction_string)
            metrics = calculate_string_f1_validation_metrics(truth, prediction)
            logger.debug(f"Truth (extraction/normalization)       : {truth}")
            logger.debug(f"Prediction (extraction/normalization)  : {prediction}")
            logger.debug(f"Metrics (extraction/normalization)     : {metrics}")
            output.append(
                {
                    "model": model_name,
                    "model_precision": precision.value,
                    "execution_time": execution_time,
                    "date": test_date,
                    "prompt": prompt,  # the full prompt that was used (with answers)
                    "instruction": instruction,  # the instruction that was used (prompt without answers)
                    "truth_string": truth_string,  # The full truth (prompt output) as a string
                    "truth": truth,  # The extraction seperated from the truth string
                    "output_string": output_string,  # The full model output as a string
                    "output_string_raw": output_string_raw,
                    # The full model output (with prompt) as with special tokens
                    "prediction_string": prediction_string,  # The model output as a string
                    "prediction": prediction,  # The extraction seperated from the model output
                    "precision": metrics[0],
                    "recall": metrics[1],
                    "f1_score": metrics[2],
                    "task": example[
                        "task"
                    ],  # The task of the example (extraction, normalization, catalog, summary)
                    "type": example[
                        "type"
                    ],  # Type of annotation (DIAGNOSIS, MEDICATION, TREATMENT)
                    "source": example["source"],  # data source
                    "na_prompt": example["na_prompt"],  # if the example is empty or not
                }
            )

        # ATTRIBUTES
        if example["task"] == "extraction":
            truth = get_extractions_with_attributes(truth_string)
            prediction = get_extractions_with_attributes(prediction_string)
            metrics = calculate_string_f1_validation_metrics(truth, prediction)
            logger.debug(f"Metrics (attribute)      : {metrics}")
            logger.debug(f"Truth (attributes)       : {truth}")
            logger.debug(f"Prediction (attributes)  : {prediction}")
            output.append(
                {
                    "model": model_name,
                    "model_precision": precision.value,
                    "execution_time": execution_time,
                    "date": test_date,
                    "prompt": prompt,
                    "instruction": instruction,
                    "truth_string": truth_string,
                    "truth": truth,
                    "output_string": output_string,
                    "output_string_raw": output_string_raw,
                    "prediction_string": prediction_string,
                    "prediction": prediction,
                    "precision": metrics[0],
                    "recall": metrics[1],
                    "f1_score": metrics[2],
                    "task": "attributes",
                    "type": example["type"],
                    "source": example["source"],
                    "na_prompt": example["na_prompt"],
                }
            )

        # CATALOG
        if example["task"] == "catalog":
            metrics = calculate_string_f1_validation_metrics(
                [truth_string], [prediction_string]
            )
            logger.debug(f"Truth (catalog)          : {truth_string}")
            logger.debug(f"Prediction (catalog)     : {prediction_string}")
            logger.debug(f"Metrics (catalog)         : {metrics}")
            output.append(
                {
                    "model": model_name,
                    "model_precision": precision.value,
                    "execution_time": execution_time,
                    "date": test_date,
                    "prompt": prompt,
                    "instruction": instruction,
                    "truth_string": truth_string,
                    "truth": truth_string,
                    "output_string": output_string,
                    "output_string_raw": output_string_raw,
                    "prediction_string": prediction_string,
                    "prediction": prediction_string,
                    "precision": metrics[0],
                    "recall": metrics[1],
                    "f1_score": metrics[2],
                    "task": "catalog",
                    "type": example["type"],
                    "source": example["source"],
                    "na_prompt": example["na_prompt"],
                }
            )

        logger.debug(f"{150 * '-'}")
    # convert to pandas df and save to json
    df = pd.DataFrame(output)
    df.to_json(
        f"validation_results_{precision.value}bit_{model_name}.json", orient="records"
    )
    return output


def calculate_string_f1_validation_metrics(
    truth_labels: list[str], prediction_labels: list[str]
):
    """
    Calculate precision, recall and f1 score for the extraction task. Takes in two lists, the truth labels and the
    predicted labels and returns the metrics.
    :param truth_labels: The truth labels as a list
    :param prediction_labels: The predicted labels as a list
    """
    # TODO: maybe to make results better: filter out strings that are not in original text --> no false positives
    # lower case and strip
    truth_set = {x.lower().strip() for x in truth_labels}
    prediction_set = {x.lower().strip() for x in prediction_labels}

    # get unique
    truth_set = set(truth_set)
    prediction_set = set(prediction_set)

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


def calculate_rogue_metrics(prediction: str, desired: str):
    """
    Calculate the ROUGE metrics for the given prediction and desired string. More info can be found on the
    transformers documentation https://huggingface.co/spaces/evaluate-metric/rouge
    :param prediction: The text that was predicted from the model
    :param desired: The text that should have been predicted by the model
    :return: rogue1, rogue2, rogueL and rogueLsum in a dictionary
    """
    rouge = evaluate.load("rouge")
    results = rouge.compute(
        predictions=[prediction.lower().strip()], references=[desired.lower().strip()]
    )
    return results


def aggregate_metrics(file_name: str):
    df = pd.read_json(file_name)
    grouped = df.groupby(["task", "source", "type", "na_prompt"])
    for name, group in grouped:
        logger.debug(name)
        logger.debug(f"Precision: {group['precision'].mean()}")
        logger.debug(f"Recall: {group['recall'].mean()}")
        logger.debug(f"F1 Score: {group['f1_score'].mean()}")
        logger.debug(f"{60 * '-'}")


if __name__ == "__main__":
    calculate_metrics_from_prompts(
        ModelPrecision.SIXTEEN_BIT,
        "BachelorThesis/Gemma2b_V02_BRONCO_CARDIO_SUMMARY_CATALOG",
        4096,
    )

    calculate_metrics_from_prompts(
        ModelPrecision.FOUR_BIT,
        "BachelorThesis/Gemma2b_V02_BRONCO_CARDIO_SUMMARY_CATALOG",
        4096,
    )
    # aggregate_metrics("validation_results_8bit_Gemma2b_V01_BRONCO_CARDIO_SUMMARY.json")
