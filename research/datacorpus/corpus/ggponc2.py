import json
import os.path

import pandas as pd

from research.datacorpus.utils.utils_ner import parse_ner_dataset
from research.datacorpus.utils.utils_mongodb import (
    upload_data_to_mongodb,
    rename_dict_keys,
)
from research.logger import logger

# DATA SOURCE: https://www.leitlinienprogramm-onkologie.de/projekte/ggponc-deutsch

fine_annotations_folder = "Bachelorarbeit\\datensets\\corpus\\ggponc2\\json\\fine"
fine_annotations_ner_folder = "Bachelorarbeit\\datensets\\corpus\\ggponc2\\conll\\fine"


def load_json(filename) -> dict:
    """Load json file."""
    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)

    logger.debug(f"Parsed {len(data)} GGPONC json annotations from file {filename}")
    return data


def transform_ggponc_annotations(data):
    """Transform annotations into unified format."""
    output = []
    for document in data:
        annotations = []
        passage_text = ""
        # loop over passages and find all related entities
        for passage in document["passages"]:
            passage_start = passage["offsets"][0][0]
            passage_end = passage["offsets"][0][1]
            passage_text += passage["text"].strip() + " "

            # for all related entities save annotations into passage_annotations
            passage_annotations = []
            for entity in document["entities"]:
                entity_start = entity["offsets"][0][0]
                entity_end = entity["offsets"][0][1]
                if entity_start >= passage_start and entity_end <= passage_end:
                    passage_annotations.append(
                        {
                            "type": entity["type"].strip(),
                            "origin": passage["text"].strip(),
                            "text": entity["text"][0].strip(),
                            "start": entity["offsets"][0][0],
                            "end": entity["offsets"][0][1],
                        }
                    )

            if len(passage_annotations) == 0:
                annotations.append(
                    {
                        "type": "None",
                        "origin": passage["text"].strip(),
                        "text": [],
                        "start": [],
                        "end": [],
                    }
                )
            else:
                annotations.extend(passage_annotations)

        # group annotations by type and origin. Takes a long time...
        df = pd.DataFrame(annotations)
        grouped_df = (
            df.groupby(["type", "origin"])
            .agg(
                {
                    "text": lambda x: x.tolist(),
                    "start": lambda x: x.tolist(),
                    "end": lambda x: x.tolist(),
                }
            )
            .reset_index()
        )
        grouped_df = grouped_df.to_dict(orient="records")
        output.append(
            {
                "document": document["document"],
                "annotations": grouped_df,
                "full_text": passage_text.strip(),
            }
        )

    return output


# TODO: refactor data to resemble bronco
def get_ggponc_json(refactor=True):
    """Load the GGPONC json data from the fine annotations folder and return the short and long documents."""
    # Load data from both files
    data_short = load_json(os.path.join(fine_annotations_folder, "short", "all.json"))
    data_short = rename_dict_keys(data_short, "document_id", "document")

    data_long = load_json(os.path.join(fine_annotations_folder, "long", "all.json"))
    data_long = rename_dict_keys(data_long, "document_id", "document")

    if refactor:
        data_short = transform_ggponc_annotations(data_short)
        data_long = transform_ggponc_annotations(data_long)

    return data_short, data_long


def get_ggponc_ner():
    """Load the GGPOC NER data from the fine annotations folder and return the short and long documents."""
    # get and combine short data
    data_short_dev = parse_ner_dataset(
        os.path.join(fine_annotations_ner_folder, "short", "dev_fine_short.conll")
    )
    data_short_test = parse_ner_dataset(
        os.path.join(fine_annotations_ner_folder, "short", "test_fine_short.conll")
    )
    data_short_train = parse_ner_dataset(
        os.path.join(fine_annotations_ner_folder, "short", "train_fine_short.conll")
    )
    data_short_ner = data_short_dev + data_short_test + data_short_train

    # get and combine long data
    data_long_dev = parse_ner_dataset(
        os.path.join(fine_annotations_ner_folder, "long", "dev_fine_long.conll")
    )
    data_long_test = parse_ner_dataset(
        os.path.join(fine_annotations_ner_folder, "long", "test_fine_long.conll")
    )
    data_long_train = parse_ner_dataset(
        os.path.join(fine_annotations_ner_folder, "long", "train_fine_long.conll")
    )
    data_long_ner = data_long_dev + data_long_test + data_long_train

    return data_short_ner, data_long_ner


def build_ggponc_db():
    data_short_json, data_long_json = get_ggponc_json()
    data_short_ner, data_long_ner = get_ggponc_ner()

    # json data
    upload_data_to_mongodb(data_short_json, "corpus", "ggponc_short", True, [])
    upload_data_to_mongodb(data_long_json, "corpus", "ggponc_long", True, [])

    # ner data
    upload_data_to_mongodb(data_short_ner, "corpus", "ggponc_short_ner", True, [])
    upload_data_to_mongodb(data_long_ner, "corpus", "ggponc_long_ner", True, [])


build_ggponc_db()
