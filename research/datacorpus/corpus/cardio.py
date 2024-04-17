import os.path
import re

import pandas as pd

from research.datacorpus.utils.utils_mongodb import upload_data_to_mongodb
from research.logger import logger

# DATA SOURCE: https://heidata.uni-heidelberg.de/dataset.xhtml?persistentId=doi:10.11588/data/AFYQDY

TSV_FOLDER_PATH = "Bachelorarbeit\\datensets\\corpus\\cardiode\\tsv"
TXT_FOLDER_PATH = "Bachelorarbeit\\datensets\\corpus\\cardiode\\txt"
TXT_HELDOUT_FOLDER_PATH = "Bachelorarbeit\\datensets\\corpus\\cardiode\\txt_heldout"


# TODO: unify anonymization (PATIENT, etc...) and labels / types
# TODO: remove the pseudo text parts <B-PER> etc...
def clean_cardio_string(text: str) -> str:
    cleaned_text = re.sub(r"<\[Pseudo] ([^>]*)>", r"\1", text)
    return cleaned_text.strip()


# TODO: also parse the duration, form, frequency, strength
def transform_cardio_annotation(annotation: dict):
    """Transform annotations into unified format."""
    drugs = []
    # get drugs and active ingredients
    for index, tag in enumerate(annotation["ner_tags"]):
        if tag == "DRUG" or tag == "ACTIVEING":
            offsets = annotation["offsets"][index].split("-")
            drugs.append(
                {
                    "type": "MEDICATION",
                    "origin": annotation["text"].strip(),
                    "id": annotation["ids"][index].strip(),
                    "text": annotation["words"][index].strip(),
                    "in_narrative": annotation["in_narrative"][index],
                    "start": int(offsets[0]),
                    "end": int(offsets[1]),
                }
            )

    if len(drugs) == 0:
        return {
            "id": [],
            "text": [],
            "origin": annotation["text"].strip(),
            "type": "None",
            "end": [],
            "start": [],
        }
    else:
        df = pd.DataFrame(drugs)
        grouped_df = (
            df.groupby(["type", "origin"])
            .agg(
                {
                    "id": lambda x: x.tolist(),
                    "text": lambda x: x.tolist(),
                    "in_narrative": lambda x: x.tolist(),
                    "start": lambda x: x.tolist(),
                    "end": lambda x: x.tolist(),
                }
            )
            .reset_index()
        )
        return grouped_df.to_dict(orient="records")[0]


def transform_ner_cardio_annotations(annotations: list[str]) -> list[str]:
    # remove the ids from the annotations
    cleaned = []
    for anno in annotations:
        cleaned.append(re.sub(r"\[.*]", "", anno).strip())

    # transform cardio annotations to the same format of ggponc2 and bronco
    transformed = []
    last_seen = None
    for anno in cleaned:
        if anno == "_":
            transformed.append("O")
        elif anno == last_seen:
            # If same type, it's inside a sequence
            transformed.append(f"I-{anno}")
        else:
            # If different type, it's the beginning of a sequence
            transformed.append(f"B-{anno}")
            last_seen = anno

    return transformed


def parse_ner_annotations():
    # get all filenames in the folder TSV_FOLDER_PATH
    tsv_files = [file for file in os.listdir(TSV_FOLDER_PATH) if file.endswith(".tsv")]
    annotations_ner = []

    current_words = []
    current_ner_tags = []
    current_in_narrative = []
    for tsv_file in tsv_files:
        annotations_file_ner = []
        with open(
            os.path.join(TSV_FOLDER_PATH, tsv_file), "r", encoding="utf-8"
        ) as file:
            for line in file:
                line = line.strip()

                # skip empty and the lines in the beginning of the file
                if not line or (line.startswith("#") and not line.startswith("#Text=")):
                    continue

                # save the previous annotation and reset the values
                if line.startswith("#Text="):
                    if current_words:
                        annotations_file_ner.append(
                            {
                                "words": current_words,
                                "ner_tags": transform_ner_cardio_annotations(
                                    current_ner_tags
                                ),
                                "in_narrative": current_in_narrative,
                            }
                        )

                    current_words = []
                    current_ner_tags = []
                    current_in_narrative = []
                    continue

                # get the values from the line
                parts = [part.strip() for part in line.split("\t")]
                word = parts[2]
                annotation = parts[3]
                in_narrative = parts[4]

                # save the values
                current_words.append(word)
                current_ner_tags.append(annotation)
                current_in_narrative.append(in_narrative)

        annotations_ner.append(
            {"document": tsv_file, "annotations": annotations_file_ner}
        )
        logger.debug(
            f"Processed {len(annotations_file_ner)} ner-annotations from the file: {tsv_file}"
        )

    return annotations_ner


def parse_annotations():
    # get all filenames in the folder TSV_FOLDER_PATH
    tsv_files = [file for file in os.listdir(TSV_FOLDER_PATH) if file.endswith(".tsv")]

    annotations = []

    current_ids = []
    current_words = []
    current_ner_tags = []
    current_offsets = []
    current_in_narrative = []
    current_relations = []
    current_text = []
    for tsv_file in tsv_files:
        annotations_file = []
        with open(
            os.path.join(TSV_FOLDER_PATH, tsv_file), "r", encoding="utf-8"
        ) as file:
            for line in file:
                line = line.strip()

                # skip empty and the lines in the beginning of the file
                if not line or (line.startswith("#") and not line.startswith("#Text=")):
                    continue

                # save the previous annotation and reset the values
                if line.startswith("#Text="):
                    if current_words:
                        annotations_file.append(
                            transform_cardio_annotation(
                                {
                                    "ids": current_ids,
                                    "words": current_words,
                                    "ner_tags": current_ner_tags,
                                    "offsets": current_offsets,
                                    "in_narrative": current_in_narrative,
                                    "relations": current_relations,
                                    "text": current_text,
                                }
                            )
                        )

                    current_ids = []
                    current_words = []
                    current_ner_tags = []
                    current_offsets = []
                    current_in_narrative = []
                    current_relations = []
                    current_text = line.split("#Text=")[1].strip()
                    continue

                # get the values from the line
                parts = [part.strip() for part in line.split("\t")]
                anno_id = parts[0]
                offset = parts[1]
                word = parts[2]
                annotation = parts[3]
                in_narrative = parts[4]
                relation = parts[5]

                # save the values
                current_ids.append(anno_id)
                current_words.append(word)
                current_ner_tags.append(annotation)
                current_offsets.append(offset)
                current_in_narrative.append(in_narrative)
                current_relations.append(relation)

        # read the txt file to get the full text
        txt_file = tsv_file.replace(".tsv", ".txt")
        with open(
            os.path.join(TXT_FOLDER_PATH, txt_file), "r", encoding="utf-8"
        ) as file:
            full_text = file.read().strip()

        # save the last file annotation
        annotations.append(
            {
                "document": tsv_file,
                "annotations": annotations_file,
                "full_text": full_text,
            }
        )

        logger.debug(
            f"Processed {len(annotations_file)} annotations from the file: {tsv_file}"
        )

    return annotations


def parse_cardio_heldout_text():
    txt_files = [
        file for file in os.listdir(TXT_HELDOUT_FOLDER_PATH) if file.endswith(".txt")
    ]

    heldout_text = []
    for filename in txt_files:
        with open(
            os.path.join(TXT_HELDOUT_FOLDER_PATH, filename), "r", encoding="utf-8"
        ) as file:
            full_text = file.read().strip()
            heldout_text.append({"document": filename, "full_text": full_text})

    logger.debug(f"Processed {len(heldout_text)} heldout texts")
    return heldout_text


def build_cardio_db():
    annotations = parse_annotations()
    annotations_ner = parse_ner_annotations()
    heldout_text = parse_cardio_heldout_text()
    upload_data_to_mongodb(annotations, "corpus", "cardio", True, [])
    upload_data_to_mongodb(annotations_ner, "corpus", "cardio_ner", True, [])
    upload_data_to_mongodb(heldout_text, "corpus", "cardio_heldout", True, [])


build_cardio_db()
