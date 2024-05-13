import pandas as pd
from datasets import load_dataset

from shared.mongodb import get_collection


def calculate_lookup_accuracy():
    """
    Calculate the accuracy for the normalization task with a simple database lookup.
    """
    atc_collection = get_collection("catalog", "atc")
    icd_collection = get_collection("catalog", "icd10gm")
    ops_collection = get_collection("catalog", "ops")
    _dataset = load_dataset("json", data_files={"data": "prompts.jsonl"})[
        "data"
    ].train_test_split(test_size=0.1, shuffle=True, seed=42)
    test_data = _dataset["test"]

    outputs = []
    for example in test_data:

        if example["task"] != "normalization":
            continue

        lookup_term = example["normalization_labels"].lower().strip()
        truth = example["annotation_labels"].lower().strip()

        if example["type"] == "MEDICATION":
            doc = atc_collection.find_one({"name": lookup_term})
            if doc is None:
                continue
            prediction = doc["atc"].lower().strip()
        elif example["type"] == "DIAGNOSIS":
            print("DIAGNOSIS")
            doc = icd_collection.find_one({"title": lookup_term})
            if doc is None:
                continue
            prediction = doc["atc"].lower().strip()
        elif example["type"] == "TREATMENT":
            print("TREATMENT")
            doc = ops_collection.find_one({"title": lookup_term})
            if doc is None:
                continue
            prediction = doc["code"].lower().strip()
        else:
            continue

        outputs.append(
            {
                "lookup_term": lookup_term,
                "truth": truth,
                "prediction": prediction,
                "correct": truth == prediction,
            }
        )

    # save to csv
    df = pd.DataFrame(outputs)
    df.to_csv("normalization_lookup_validation.csv", index=False)


calculate_lookup_accuracy()
