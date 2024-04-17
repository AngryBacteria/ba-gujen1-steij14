import re

import pandas as pd
from pandas import DataFrame

from research.datacorpus.utils.utils_mongodb import upload_data_to_mongodb
from research.logger import logger

# DATA SOURCE: https://www.nlm.nih.gov/research/umls/Snomed/core_subset.html

CORE_PATH = "F:\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\catalog\\snomed\\SNOMEDCT_CORE_SUBSET_202311.txt"
snomed_classes = [
    "situation",
    "person",
    "event",
    "disorder",
    "morphologic abnormality",
    "finding",
    "navigational concept",
    "procedure",
    "regime/therapy",
]


def extract_and_remove_class(row: str) -> tuple[str, str]:
    """Extracts the class from the SNOMED_FSN column and removes it from the row."""
    matches = re.findall(r"\(([^)]+)\)", row)
    if matches:
        last_match = matches[-1]
        if last_match.lower() in snomed_classes:
            modified_fsn = re.sub(r"\(" + re.escape(last_match) + r"\)$", "", row)
            return last_match.strip(), modified_fsn.strip()
    return "", row.strip()


def load_snomed_core(print_unique_classes=False) -> DataFrame:
    """Loads the SNOMED core subset and extracts the class as a new column."""
    headers = [
        "SNOMED_CID",
        "SNOMED_FSN",
        "SNOMED_CONCEPT_STATUS",
        "UMLS_CUI",
        "OCCURRENCE",
        "USAGE",
        "FIRST_IN_SUBSET",
        "IS_RETIRED_FROM_SUBSET",
        "LAST_IN_SUBSET",
        "REPLACED_BY_SNOMED_CID",
    ]
    data = pd.read_csv(CORE_PATH, sep="|", header=1, names=headers)
    data = data.map(lambda x: x.strip() if isinstance(x, str) else x)

    # used for creating the snomed_classes list
    if print_unique_classes:
        pattern = re.compile(r"\((.*?)\)")
        unique_classes = set()
        for index, row in data.iterrows():
            class_matches = pattern.findall(row["SNOMED_FSN"])
            for class_match in class_matches:
                unique_classes.add(class_match)
        logger.debug(f"Unique classes: {unique_classes}")

    # extract the class from the SNOMED_FSN
    data[["SNOMED_CLASS", "SNOMED_FSN"]] = (
        data["SNOMED_FSN"].apply(lambda x: extract_and_remove_class(x)).apply(pd.Series)
    )
    if print_unique_classes:
        missing = data[data["SNOMED_CLASS"] == ""]
        logger.debug(f"Missing classes: {missing}")

    # filter and rename
    filtered_data = data[data["SNOMED_CLASS"] != ""]
    filtered_data.rename(
        columns={
            "SNOMED_CID": "cid",
            "SNOMED_FSN": "name",
            "SNOMED_CONCEPT_STATUS": "status",
            "UMLS_CUI": "umls_cui",
            "OCCURRENCE": "occurrence",
            "USAGE": "usage",
            "FIRST_IN_SUBSET": "in_subset",
            "IS_RETIRED_FROM_SUBSET": "retired",
            "LAST_IN_SUBSET": "last_in_subset",
            "REPLACED_BY_SNOMED_CID": "replaced_by_snomed_cid",
            "SNOMED_CLASS": "class",
        },
        inplace=True,
    )

    return filtered_data


def create_snomed_db() -> None:
    # Upload to MongoDB
    data = load_snomed_core().to_dict(orient="records")
    upload_data_to_mongodb(data, "catalog", "snomed_core", True, ["cid"])


# TODO: translate to german
create_snomed_db()
