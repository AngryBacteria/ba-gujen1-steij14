import json
import os.path

from datacorpus.utils.ner import parse_ner_dataset
from shared.decoder_utils import count_tokens
from shared.mongodb import (
    upload_data_to_mongodb,
    rename_dict_keys,
)
from shared.logger import logger

# DATA SOURCE: https://www.leitlinienprogramm-onkologie.de/projekte/ggponc-deutsch

fine_annotations_folder = "S:\\documents\\onedrive_bfh\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\corpus\\ggponc2\\json\\fine"
fine_annotations_ner_folder = "F:\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\corpus\\ggponc2\\conll\\fine"


# TODO: unify anonymization (PATIENT, etc...)
def load_json(filename) -> dict:
    """Load json file."""
    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)
    logger.debug(f"Parsed {len(data)} GGPONC json annotations from file {filename}")
    return data


def get_normalized_entity_type(entity_type: str) -> str:
    """
    Normalize the entity types to be compatible with the other datasets.
    :param entity_type: The entity type to normalize.
    :return: The normalized entity type.
    """
    if entity_type == "Diagnosis_or_Pathology":
        return "DIAGNOSIS"
    elif entity_type == "Clinical_Drug":
        return "MEDICATION"
    elif entity_type == "Therapeutic":
        return "TREATMENT"
    else:
        return entity_type.strip()


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
            passage_annotations = [
                {
                    "type": get_normalized_entity_type(entity["type"]),
                    "origin": passage["text"].strip(),
                    "text": entity["text"][0].strip(),
                    "start": entity["offsets"][0][0],
                    "end": entity["offsets"][0][1],
                }
                for entity in document["entities"]
                if (
                    passage_start <= entity["offsets"][0][0]
                    and entity["offsets"][0][1] <= passage_end
                )
            ]
            if len(passage_annotations) == 0:
                annotations.append(
                    {
                        "type": "NA",
                        "origin": passage["text"].strip(),
                        "text": [],
                        "start": [],
                        "end": [],
                    }
                )
            else:
                grouped_annotations = {}
                for annotation in passage_annotations:
                    key = (annotation["type"], annotation["origin"])
                    if key not in grouped_annotations:
                        grouped_annotations[key] = {
                            "type": annotation["type"],
                            "origin": annotation["origin"],
                            "text": [],
                            "start": [],
                            "end": [],
                        }
                    grouped_annotations[key]["text"].append(annotation["text"])
                    grouped_annotations[key]["start"].append(annotation["start"])
                    grouped_annotations[key]["end"].append(annotation["end"])
                annotations.extend(list(grouped_annotations.values()))

        output.append(
            {
                "document": document["document"],
                "annotations": annotations,
                "full_text": passage_text.strip(),
            }
        )
    return output


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


def get_normalized_ner_tags(ner_tags: list[str]):
    new_tags = []
    for tag in ner_tags:
        if tag == "B-Diagnosis_or_Pathology":
            new_tags.append("B-DIAG")
        elif tag == "I-Diagnosis_or_Pathology":
            new_tags.append("I-DIAG")
        elif tag == "B-Therapeutic":
            new_tags.append("B-TREAT")
        elif tag == "I-Therapeutic":
            new_tags.append("I-TREAT")
        elif tag == "B-Clinical_Drug":
            new_tags.append("B-MED")
        elif tag == "I-Clinical_Drug":
            new_tags.append("I-MED")
        else:
            new_tags.append(tag.strip())

    return new_tags


def get_ggponc_ner(refactor=True):
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
    if refactor:
        for entry in data_short_ner:
            entry["ner_tags"] = get_normalized_ner_tags(entry["ner_tags"])

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
    if refactor:
        for entry in data_long_ner:
            entry["ner_tags"] = get_normalized_ner_tags(entry["ner_tags"])

    return data_short_ner, data_long_ner


def count_ggponc_tokens():
    """
    Count the tokens in the GGPONC dataset.
    """
    texts = []
    data_short = load_json(os.path.join(fine_annotations_folder, "short", "all.json"))
    for document in data_short:
        for passage in document["passages"]:
            texts.append(passage["text"])

    return count_tokens(texts)


def build_ggponc_db():
    data_short_json, data_long_json = get_ggponc_json()
    data_short_ner, data_long_ner = get_ggponc_ner()

    # json data
    upload_data_to_mongodb(data_short_json, "corpus", "ggponc_short", True, [])
    upload_data_to_mongodb(data_long_json, "corpus", "ggponc_long", True, [])

    # ner data
    upload_data_to_mongodb(data_short_ner, "corpus", "ggponc_short_ner", True, [])
    upload_data_to_mongodb(data_long_ner, "corpus", "ggponc_long_ner", True, [])


if __name__ == "__main__":
    build_ggponc_db()
    # print(count_ggponc_tokens())
