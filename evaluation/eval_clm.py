import os

import setproctitle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
setproctitle.setproctitle("gujen1 - bachelorthesis")

import evaluate
from pandas import DataFrame
from statistics import mean
from sklearn.metrics import recall_score, f1_score, precision_score
from sklearn.preprocessing import MultiLabelBinarizer
import datetime
from shared.logger import logger
from shared.clm_model_utils import (
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

        # get the model output and vram usage
        _start_time = datetime.datetime.now()
        _inputs = tokenizer(instruction, return_tensors="pt").to("cuda:0")
        _outputs = model.generate(**_inputs, max_new_tokens=2500)
        output_string = tokenizer.decode(_outputs[0], skip_special_tokens=True)
        output_string_raw = tokenizer.decode(_outputs[0], skip_special_tokens=False)
        _end_time = datetime.datetime.now()
        execution_time = (_end_time - _start_time).microseconds

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
    df.to_json(
        f"validation_results_{precision.value}bit_{model_name}.json", orient="records"
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


def calculate_rouge_metrics(truth: str, prediction: str):
    """
    Calculate the ROUGE metrics for the given prediction and desired string. More info can be found on the
    transformers documentation https://huggingface.co/spaces/evaluate-metric/rouge
    :param prediction: The text that was predicted from the model
    :param truth: The text that should have been predicted by the model
    :return: rogue1, rogue2, rogueL and rogueLsum in a dictionary
    """
    rouge = evaluate.load("rouge")
    results = rouge.compute(
        predictions=[prediction.lower().strip()], references=[truth.lower().strip()]
    )
    return results


def get_rouge_mean_from_df(df: DataFrame):
    """
    Get the mean ROUGE score mean from a grouped DataFrame
    """
    rogue1 = []
    rogue2 = []
    rogueL = []
    rogueLsum = []

    for _, row in df.iterrows():
        rouge_metrics = calculate_rouge_metrics(
            row["truth_string"], row["prediction_string"]
        )
        rogue1.append(rouge_metrics["rouge1"])
        rogue2.append(rouge_metrics["rouge2"])
        rogueL.append(rouge_metrics["rougeL"])
        rogueLsum.append(rouge_metrics["rougeLsum"])

    return mean(rogue1), mean(rogue2), mean(rogueL), mean(rogueLsum)


def get_extraction_normalization_mean_f1(df: DataFrame, ignore_na=False):
    """
    Get the mean F1 (and co) score for the tasks extraction and normalization from a grouped DataFrame.
    Examples without extractions can be ignored.
    """

    truths = []
    predictions = []
    for _, row in df.iterrows():
        if ignore_na and row["na_prompt"]:
            continue

        truth = get_extractions_without_attributes(row["truth_string"])
        prediction = get_extractions_without_attributes(row["prediction_string"])
        truths.append(truth)
        predictions.append(prediction)

    return calculate_string_validation_metrics(truths, predictions)


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
    truths = []
    predictions = []
    if only_check_existence:
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

        return calculate_string_validation_metrics(truths, predictions)

    # ATTENTION THIS IS EXPERIMENTAL (and probably wrong)
    else:
        raise NotImplementedError("This is not implemented correctly yet")
        precision = []
        recall = []
        f1 = []

        for index, row in df.iterrows():
            truths = get_extractions_with_attributes_grouped(row["truth_string"])
            predictions = get_extractions_with_attributes_grouped(
                row["prediction_string"]
            )
            metrics_temp_precision = []
            metrics_temp_recall = []
            metrics_temp_f1 = []
            for key, value in predictions.items():
                if key in truths:
                    if ignore_positive:
                        value = [x for x in value if x != "POSITIV"]
                        truths[key] = [x for x in truths[key] if x != "POSITIV"]
                    if ignore_na and len(truths[key]) == 0:
                        continue

                    metrics_temp = calculate_string_validation_metrics(
                        value, truths[key]
                    )
                    metrics_temp_precision.append(metrics_temp[0])
                    metrics_temp_recall.append(metrics_temp[1])
                    metrics_temp_f1.append(metrics_temp[2])

            precision.append(
                mean(metrics_temp_precision) if metrics_temp_precision else 1
            )
            recall.append(mean(metrics_temp_recall) if metrics_temp_recall else 1)
            f1.append(mean(metrics_temp_f1) if metrics_temp_f1 else 1)

        return mean(f1), mean(precision), mean(recall)


# ATTRIBUTE: Ignore positive prompts, maybe use sklearn
def aggregate_metrics(
    file_name: str,
    write_to_csv=True,
    write_to_new_excel=True,
    excel_sheet_name="Results",
):
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
                    "rouge1": rouge_scores[0],
                    "rouge2": rouge_scores[1],
                    "rougeL": rouge_scores[2],
                    "rougeLsum": rouge_scores[3],
                    "n": len(group),
                }
            )

        # Extraction and Normalization
        if name[0] == "extraction" or name[0] == "normalization":
            f1_scores = get_extraction_normalization_mean_f1(group, True)
            logger.debug(f"{name} -- {f1_scores}")
            metrics_output.append(
                {
                    "source": name[1] if len(name) > 1 else None,
                    "type": name[2] if len(name) > 2 else None,
                    "task": name[0],
                    "precision": f1_scores[0],
                    "recall": f1_scores[1],
                    "f1_score": f1_scores[2],
                    "n": len(group),
                }
            )

        # Attributes
        if name[0] == "extraction":
            f1_scores = get_attribute_mean_f1(group, False, True, True)
            logger.debug(f"ATTRIBUTE:{name} -- {f1_scores}")
            metrics_output.append(
                {
                    "source": name[1] if len(name) > 1 else None,
                    "type": name[2] if len(name) > 2 else None,
                    "task": "attribute",
                    "precision": f1_scores[0],
                    "recall": f1_scores[1],
                    "f1_score": f1_scores[2],
                    "n": len(group),
                }
            )

    metrics_df = pd.DataFrame(metrics_output)
    metrics_df = metrics_df.sort_values(by="task")

    if write_to_csv:
        metrics_df.to_csv("results.csv", index=False)
    if write_to_new_excel:
        metrics_df.to_excel("results.xlsx", index=False, sheet_name=excel_sheet_name)


if __name__ == "__main__":
    # get_eval_data_from_models(
    #     ModelPrecision.SIXTEEN_BIT,
    #     "LeoMistral_V06",
    #     4096,
    # )
    aggregate_metrics(
        "S:\\documents\\onedrive_bfh\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\Training\\Resultate\\validation_results_16bit_Gemma2b_V03.json",
        write_to_csv=False,
        write_to_new_excel=True,
    )