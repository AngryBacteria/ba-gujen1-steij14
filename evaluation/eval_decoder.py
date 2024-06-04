import os

import setproctitle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
setproctitle.setproctitle("gujen1 - bachelorthesis")

from shared.gpu_utils import get_cuda_memory_usage
import time
import torch
import evaluate
from pandas import DataFrame
from sklearn.metrics import recall_score, f1_score, precision_score
from sklearn.preprocessing import MultiLabelBinarizer
import datetime
from shared.logger import logger
from shared.decoder_utils import (
    load_model_and_tokenizer,
    ModelPrecision,
    get_model_output_only,
    get_extractions_without_attributes,
    get_extractions_with_attributes_grouped,
    get_attributes_only,
)

from datasets import load_dataset
import pandas as pd


def get_eval_data_from_models(
    precision: ModelPrecision,
    model_name: str,
    max_sequence_length: int,
    tasks_to_eval=None,
    only_eval_resources=False,
):
    """
    The model makes a prediction which is then stored with the ground truth (the full prompt with answer).
    The prompts.jsonl data needs to be present in the directory. The results are saved to a file to later analyze
    it with aggregate_metrics()
    :param model_name: The name of the model. Used to save the file
    :param tasks_to_eval: The tasks that should be evaluated
    :param precision: The precision to load the model in (16bit recommended if hardware supports it)
    :param max_sequence_length: The max sequence length the model should be evaluated on. The full prompt (with answer)
    is taken as the filter for the max sequence length.
    :param only_eval_resources: If True, only the VRAM consumption and time in milliseconds is evaluated. Cache is
    cleared after each example to be accurate. Only around 10 examples per task is evaluated to save time.
    :return: None, the data is saved to a file
    """
    if tasks_to_eval is None:
        tasks_to_eval = ["extraction", "normalization", "summary"]

    tokenizer, model = load_model_and_tokenizer(precision=precision)
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
        if len(_tokenized) >= max_sequence_length:
            logger.warning(
                f"Skipping example {i} with length {len(_tokenized)}, because it is too long"
            )
            continue

        # create the instruction (prompt) for the model
        instruction: str = tokenizer.apply_chat_template(
            example["messages"][:-1], tokenize=False, add_generation_prompt=True
        )
        if instruction is None or instruction == "":
            logger.error("No instruction found")
            continue

        # get the whole input text (prompt + output) to save later
        prompt: str = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=True
        )
        if prompt is None or prompt == "":
            logger.error("No prompt found")
            continue
        logger.debug(f"Prompt [{i}]: {prompt}")

        # get the ground truth_string from the full_prompt
        truth_string: str = example["messages"][-1]["content"]
        if truth_string is None or truth_string == "":
            logger.error("No truth string found")
            continue

        # get the model output
        if only_eval_resources:
            _start_time = time.perf_counter_ns()
        _inputs = tokenizer(instruction, return_tensors="pt").to("cuda:0")
        _outputs = model.generate(**_inputs, max_new_tokens=2500)
        if only_eval_resources:
            _end_time = time.perf_counter_ns()
            allocated, capacity = get_cuda_memory_usage(0)
            execution_time = (_end_time - _start_time) / 10**6
            output.append(
                {
                    "model": model_name,
                    "model_precision": precision.value,
                    "execution_time": execution_time,
                    "allocated": allocated,
                    "capacity": capacity,
                    "tokens": len(_outputs[0]),
                }
            )

            logger.debug(
                f"Execution time: {execution_time} ms "
                f"used VRAM: {allocated}/{capacity} GB "
                f"for {len(_outputs[0])} tokens"
            )
            torch.cuda.empty_cache()
            continue

        output_string = tokenizer.decode(_outputs[0], skip_special_tokens=True)
        output_string_raw = tokenizer.decode(_outputs[0], skip_special_tokens=False)

        # get model prediction
        prediction_string = get_model_output_only(output_string, lower=False)
        if not prediction_string or prediction_string == "":
            logger.error("No output from model found")
            continue

        # log the results
        logger.debug(f"Truth: {truth_string}")
        logger.debug(f"Prediction: {prediction_string}")

        output.append(
            {
                "model": model_name,
                "model_precision": precision.value,
                "execution_time": execution_time,
                "date": test_date,
                "prompt": prompt,  # the full prompt that was used (with answers)
                "instruction": instruction,  # the instruction that was used (prompt without answers)
                "truth_string": truth_string,  # The full truth (prompt output) as a string
                "output_string": output_string,  # The full model output as a string
                "output_string_raw": output_string_raw,
                # The full model output (with prompt) as with special tokens
                "prediction_string": prediction_string,  # The model output as a string
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

        logger.debug(f"{150 * '-'}")
    # convert to pandas df and save to json
    df = pd.DataFrame(output)
    if only_eval_resources:
        df.to_csv(
            f"validation_results_resources_{precision.value}bit_{model_name}.csv",
        )
    else:
        df.to_json(
            f"validation_results_{precision.value}bit_{model_name}.json",
            orient="records",
        )
    return output


def calculate_string_validation_metrics(
    truth_labels_list: list[list[str]], prediction_labels_list: list[list[str]]
):
    """
    Calculate precision, recall and f1 score for all datasets combined. Takes in two lists of lists,
    the truth labels and the predicted labels for each dataset and returns the combined metrics.
    :param truth_labels_list: A list of lists, where each sublist is the truth labels for a datapoint.
    :param prediction_labels_list: A list of lists, where each sublist is the predicted labels for a datapoint.
    :return: A tuple containing the combined precision, recall, and f1 score.
    """
    # lower case and strip
    truth_labels_list = [
        [x.lower().strip() for x in labels] for labels in truth_labels_list
    ]
    prediction_labels_list = [
        [x.lower().strip() for x in labels] for labels in prediction_labels_list
    ]

    mlb = MultiLabelBinarizer()
    all_labels = sorted(
        set(
            label
            for labels in truth_labels_list + prediction_labels_list
            for label in labels
        )
    )
    mlb.fit([all_labels])

    y_true = mlb.transform(truth_labels_list)
    y_pred = mlb.transform(prediction_labels_list)

    precision = precision_score(y_true, y_pred, average="micro", zero_division=1)
    recall = recall_score(y_true, y_pred, average="micro", zero_division=1)
    f1 = f1_score(y_true, y_pred, average="micro", zero_division=1)

    return precision, recall, f1


def calculate_rouge_metrics(truths: list[str], predictions: list[str]):
    """
    Calculate the ROUGE metrics for the given prediction and desired string. More info can be found on the
    transformers documentation https://huggingface.co/spaces/evaluate-metric/rouge
    :param predictions: The texts that were predicted from the model
    :param truths: The texts that should have been predicted by the model
    :return: rogue1, rogue2, rogueL and rogueLsum in a dictionary
    """
    rouge = evaluate.load("rouge")

    predictions = [prediction.lower().strip() for prediction in predictions]
    truths = [truth.lower().strip() for truth in truths]

    results = rouge.compute(predictions=predictions, references=truths)
    return results


def get_rouge_mean_from_df(df: DataFrame):
    """
    Get the mean ROUGE score mean from a grouped DataFrame
    """
    truths = []
    predictions = []
    for _, row in df.iterrows():
        truths.append(row["truth_string"])
        predictions.append(row["prediction_string"])

    return calculate_rouge_metrics(truths, predictions)


def get_extraction_normalization_mean_f1(
    df: DataFrame, ignore_na=False, detailed_normalization=True
):
    """
    Get the mean F1 (and co) score for the tasks extraction and normalization from a grouped DataFrame.
    Examples without extractions can be ignored.
    :param df: The DataFrame with the grouped data
    :param ignore_na: Ignores examples without extractions
    :param detailed_normalization: If True, the normalization will be evaluated in more detail, meaning the icd10 and
    ops codes will be evaluated until their last digit. False means only 3 digits for icd10 and 4 for ops codes.
    """

    truths = []
    predictions = []
    for _, row in df.iterrows():
        truth = get_extractions_without_attributes(row["truth_string"])
        prediction = get_extractions_without_attributes(row["prediction_string"])
        if (
            ignore_na
            and "keine vorhanden" in truth
            and row["na_prompt"]
            and "keine vorhanden" in prediction
        ):
            continue

        if row["task"] == "normalization" and not detailed_normalization:
            truth = [current_truth.split(".")[0] for current_truth in truth]
            prediction = [
                current_prediction.split(".")[0] for current_prediction in prediction
            ]

        truths.append(truth)
        predictions.append(prediction)

    return calculate_string_validation_metrics(truths, predictions), len(truths)


def get_attribute_mean_f1(
    df: DataFrame, ignore_na=False, ignore_positive=False, only_check_existence=False
):
    """
    Get the meanF1 (and co) score from a grouped DataFrame for the task of assigning attributes. Examples without
    attributes can be ignored and the positive label can be ignored as well.
    :param df:
    :param ignore_na: Ignores examples without attributes
    :param ignore_positive: Ignores the "positiv" label because that is the most frequent label
    :param only_check_existence: Only check if the attribute exists in the truth string and no complicated F1 score
    calculation by grouping entity and attribute
    """
    if only_check_existence:
        truths = []
        predictions = []
        for index, row in df.iterrows():

            truth = get_attributes_only(row["truth_string"])
            prediction = get_attributes_only(row["prediction_string"])
            if ignore_positive:
                prediction = [x for x in prediction if x != "POSITIV"]
                truth = [x for x in truth if x != "POSITIV"]
            if ignore_na and len(truth) == 0:
                continue

            truths.append(truth)
            predictions.append(prediction)

        return calculate_string_validation_metrics(truths, predictions), len(truths)

    # ATTENTION THIS IS EXPERIMENTAL (and probably wrong)
    else:
        truths = []
        predictions = []

        for index, row in df.iterrows():
            # get grouped attributes, meaning the attributes are grouped by the entity
            truth = get_extractions_with_attributes_grouped(row["truth_string"])
            prediction = get_extractions_with_attributes_grouped(
                row["prediction_string"]
            )
            # iterate over all entities of the prediction
            for key, value in prediction.items():
                # if the entity (key) ist also in the truth, calculate the metrics
                # this means if not it won't calculate, because this would be the extraction metric
                if key in truth:
                    if ignore_positive:
                        value = [x for x in value if x != "POSITIV"]
                        truth[key] = [x for x in truth[key] if x != "POSITIV"]
                    if ignore_na and len(truth[key]) == 0:
                        continue

                    predictions.append(value)
                    truths.append(truth[key])

        for i, tru in enumerate(truths):
            logger.debug(f"Truth {i}: {tru}")
            logger.debug(f"Prediction {i}: {predictions[i]}")
        return calculate_string_validation_metrics(truths, predictions), len(truths)


def aggregate_task_metrics(
    file_name: str,
    write_to_csv=True,
    write_to_excel=True,
    excel_sheet_name="Results",
):
    """
    Aggregates the metrics from the evaluation for the tasks extraction, normalization, and summary. Takes in a file
    as input and writes the results to a csv and/or an excel file.
    :param file_name: The name/path of the file to read the data from
    :param write_to_csv: If an csv file should be written
    :param write_to_excel: If a excel file should be created or not
    :param excel_sheet_name: The name of the sheet in the excel file
    """
    df = pd.read_json(file_name)
    grouped = df.groupby(["task", "source", "type"])
    # grouped = df.groupby(["task"])

    metrics_output = []
    for name, group in grouped:
        # Summary
        if name[0] == "summary":
            rouge_scores = get_rouge_mean_from_df(group)
            logger.debug(f"{name} -- {rouge_scores}")
            metrics_output.append(
                {
                    "source": name[1] if len(name) > 1 else None,
                    "type": name[2] if len(name) > 2 else None,
                    "task": name[0],
                    "rouge1": rouge_scores["rouge1"],
                    "rouge2": rouge_scores["rouge2"],
                    "rougeL": rouge_scores["rougeL"],
                    "rougeLsum": rouge_scores["rougeLsum"],
                    "n": len(group),
                    "execution_time": group["execution_time"].mean(),
                }
            )

        # Extraction and Normalization
        if name[0] == "extraction" or name[0] == "normalization":
            f1_scores, extraction_normalization_amount = (
                get_extraction_normalization_mean_f1(group, True, True)
            )
            logger.debug(f"{name} -- {f1_scores}")
            metrics_output.append(
                {
                    "source": name[1] if len(name) > 1 else None,
                    "type": name[2] if len(name) > 2 else None,
                    "task": name[0],
                    "precision": f1_scores[0],
                    "recall": f1_scores[1],
                    "f1_score": f1_scores[2],
                    "n": extraction_normalization_amount,
                    "execution_time": group["execution_time"].mean(),
                }
            )

        # Attributes
        if name[0] == "extraction":
            f1_scores, attribute_amount = get_attribute_mean_f1(group, False, True, True)
            logger.debug(f"ATTRIBUTE:{name} -- {f1_scores}")
            metrics_output.append(
                {
                    "source": name[1] if len(name) > 1 else None,
                    "type": name[2] if len(name) > 2 else None,
                    "task": "attribute",
                    "precision": f1_scores[0],
                    "recall": f1_scores[1],
                    "f1_score": f1_scores[2],
                    "n": attribute_amount,
                    "execution_time": group["execution_time"].mean(),
                }
            )

    metrics_df = pd.DataFrame(metrics_output)
    metrics_df = metrics_df.sort_values(by="task")

    if write_to_csv:
        metrics_df.to_csv("results.csv", index=False)
    if write_to_excel:
        excel_path = "results.xlsx"
        if os.path.exists(excel_path):
            with pd.ExcelWriter(
                excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
            ) as writer:
                metrics_df.to_excel(writer, index=False, sheet_name=excel_sheet_name)
        else:
            metrics_df.to_excel(excel_path, index=False, sheet_name=excel_sheet_name)


if __name__ == "__main__":
    aggregate_task_metrics(
        r"S:\documents\onedrive_bfh\OneDrive - Berner Fachhochschule\Dokumente\UNI\Bachelorarbeit\Training\Resultate\model_outputs\validation_results_16bit_Gemma-2b_V03.JSON",
        write_to_csv=False,
        write_to_excel=True,
        excel_sheet_name="Gemma_V03",
    )
    aggregate_task_metrics(
        r"S:\documents\onedrive_bfh\OneDrive - Berner Fachhochschule\Dokumente\UNI\Bachelorarbeit\Training\Resultate\model_outputs\validation_results_16bit_LLama-3-8b_V03.json",
        write_to_csv=False,
        write_to_excel=True,
        excel_sheet_name="LLama_V03",
    )
    aggregate_task_metrics(
        r"S:\documents\onedrive_bfh\OneDrive - Berner Fachhochschule\Dokumente\UNI\Bachelorarbeit\Training\Resultate\model_outputs\validation_results_16bit_LeoMistral-7b_V06.json",
        write_to_csv=False,
        write_to_excel=True,
        excel_sheet_name="LeoMistral_V06",
    )
