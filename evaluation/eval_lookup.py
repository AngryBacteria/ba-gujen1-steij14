import datetime
import re

import pandas as pd
from datasets import load_dataset

from shared.logger import logger
from shared.mongodb import get_collection


def calculate_simple_db_lookup_accuracy():
    """
    Calculate the accuracy for the normalization task with a simple database lookup as prediction.
    """
    atc_collection = get_collection("catalog", "atc")
    icd_collection = get_collection("catalog", "icd10gm")
    ops_collection = get_collection("catalog", "ops")
    _dataset = load_dataset("json", data_files={"data": "prompts.jsonl"})[
        "data"
    ].train_test_split(test_size=0.1, shuffle=True, seed=42)
    test_data = _dataset["test"]

    test_date = datetime.datetime.now()
    outputs = []
    for example in test_data:

        if example["task"] != "normalization":
            logger.warning(
                f"Skipping example with task {example['task']} because it is not a normalization."
            )
            continue

        lookup_term = example["context_entity"].lower().strip()
        truth = example["output"].lower().strip()
        regx = re.compile(f"^{lookup_term}", re.IGNORECASE)

        _start_time = datetime.datetime.now()
        if example["type"] == "MEDICATION":
            doc = atc_collection.find_one({"title": regx})
            if doc is None:
                prediction = "404"
            else:
                prediction = doc["code"].lower().strip()
        elif example["type"] == "DIAGNOSIS":
            doc = icd_collection.find_one(
                {
                    "$or": [
                        {"title": regx},
                        {"synonyms": {"$elemMatch": {"$regex": regx}}},
                    ]
                }
            )
            if doc is None:
                prediction = "404"
            else:
                prediction = doc["code"].lower().strip()
        elif example["type"] == "TREATMENT":
            doc = ops_collection.find_one(
                {
                    "$or": [
                        {"title": regx},
                        {"synonyms": {"$elemMatch": {"$regex": regx}}},
                    ]
                }
            )
            if doc is None:
                prediction = "404"
            else:
                prediction = doc["code"].lower().strip()
        else:
            continue
        _end_time = datetime.datetime.now()
        execution_time = (_end_time - _start_time).microseconds

        metric = 1 if truth == prediction else 0
        logger.debug(f"Prediction: {prediction}, Truth: {truth}, Metric: {metric}")
        outputs.append(
            {
                "model": "db_lookup",
                "model_precision": "",
                "execution_time": execution_time,
                "date": test_date,
                "prompt": "",
                "instruction": "",
                "truth_string": "",
                "truth": truth,
                "output_string": "",
                "output_string_raw": "",
                "prediction_string": "",
                "prediction": prediction,
                "precision": metric,
                "recall": metric,
                "f1": metric,
                "task": example["task"],
                "type": example["type"],
                "source": example["source"],
                "na_prompt": example["na_prompt"],
            }
        )

    # save to csv
    df = pd.DataFrame(outputs)
    df.to_json(
        "validation_results_normalization_db_lookup.json", index=False, orient="records"
    )


def aggregate_simple_db_lookup():
    df = pd.read_json(
        "S:\\documents\\onedrive_bfh\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\Training\\RESULTATE\\validation_results_normalization_db_lookup.json"
    )
    grouped = df.groupby(["type"])
    for name, group in grouped:
        logger.debug(name)
        logger.debug(group["precision"].mean())
        logger.debug(f"{60 * '-'}")


if __name__ == "__main__":
    # calculate_simple_db_lookup_accuracy()
    aggregate_simple_db_lookup()
