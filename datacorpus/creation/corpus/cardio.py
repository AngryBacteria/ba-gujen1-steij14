import os.path
import re

import pandas as pd

from shared.clm_model_utils import count_tokens
from shared.mongodb import upload_data_to_mongodb
from shared.logger import logger

# DATA SOURCE: https://heidata.uni-heidelberg.de/dataset.xhtml?persistentId=doi:10.11588/data/AFYQDY

TSV_FOLDER_PATH = r"S:\documents\onedrive_bfh\OneDrive - Berner Fachhochschule\Dokumente\UNI\Bachelorarbeit\Daten\corpus\cardiode\tsv"
TXT_FOLDER_PATH = r"S:\documents\onedrive_bfh\OneDrive - Berner Fachhochschule\Dokumente\UNI\Bachelorarbeit\Daten\corpus\cardiode\txt"
TXT_HELDOUT_FOLDER_PATH = r"S:\documents\onedrive_bfh\OneDrive - Berner Fachhochschule\Dokumente\UNI\Bachelorarbeit\Daten\corpus\cardiode\txt_heldout"


# TODO: unify anonymization (PATIENT, etc...)
def clean_cardio_string(text: str) -> str:
    """
    Clean the cardio text from pseudo tags (extract date and remove parentheses).
    :param text: The text to clean
    :return: The cleaned text
    """
    cleaned_text = re.sub(r"<\[Pseudo] ([^>]*)>", r"\1", text)
    return cleaned_text.strip()


def print_unique_anonymization():
    """
    Prints all unique anonymization tags from the cardio documents.
    :return: None, prints the unique anonymization tags
    """
    tsv_files = [file for file in os.listdir(TSV_FOLDER_PATH) if file.endswith(".tsv")]

    unique_matches = set()
    for tsv_file in tsv_files:
        with open(
            os.path.join(TSV_FOLDER_PATH, tsv_file), "r", encoding="utf-8"
        ) as file:
            pattern = r"[BI]-[A-Z]{3,}"
            matches = re.findall(pattern, file.read())
            unique_matches.update(matches)

    unique_matches = sorted(unique_matches)
    for match in unique_matches:
        print(match)


def get_normalized_ner_tag(ner_tag: str):
    """
    Function to normalize the NER tags of the cardio dataset. Useful to later combine multiple datasets together.
    :param ner_tag: The ner tag to normalize
    :return: Normalized ner tag
    """
    if ner_tag == "B-DRUG" or ner_tag == "B-ACTIVEING":
        return "B-MED"
    elif ner_tag == "I-DRUG" or ner_tag == "I-ACTIVEING":
        return "I-MED"
    else:
        return ner_tag.strip()


def transform_ner_cardio_annotations(annotations: list[str]) -> list[str]:
    """
    Transform the cardio NER annotations into a unified format.
    :param annotations: List of annotation strings.
    :return: Unified format of the annotations.
    """
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
            transformed.append(get_normalized_ner_tag(f"I-{anno}"))
        else:
            # If different type, it's the beginning of a sequence
            transformed.append(get_normalized_ner_tag(f"B-{anno}"))
            last_seen = anno

    return transformed


def get_medication_relations(annotation: dict, relation_type: str):
    """
    Get relations of type relation_type from the annotation. These could be the duration or frequency of the medication.
    :param annotation: The annotation object
    :param relation_type: The type of relation to get
    :return: List of relations of type relation_type
    """
    # get relation mentions
    relations = []
    current_relation_text = []
    current_relation = None
    for index, tag in enumerate(annotation["ner_tags"]):
        relation = annotation["relations"][index]
        word = annotation["words"][index]

        # Check if the current tag is relation of type relation_type
        if relation_type in tag:
            # Start a new relation if not already started
            if current_relation_text == []:
                current_relation = relation
            current_relation_text.append(word)
        else:
            # If a relation just ended, save it
            if current_relation_text:
                # If there are multiple relations, split and store duration under each
                if "|" in current_relation:
                    for rel in current_relation.split("|"):
                        if relation_type in ["FREQUENCY", "STRENGTH"]:
                            text = "".join(current_relation_text)
                        else:
                            text = " ".join(current_relation_text)
                        relations.append({"relation": rel, "relation_text": text})
                else:
                    if relation_type in ["FREQUENCY", "STRENGTH"]:
                        text = "".join(current_relation_text)
                    else:
                        text = " ".join(current_relation_text)
                    relations.append(
                        {"relation": current_relation, "relation_text": text}
                    )

                # Reset the current relation
                current_relation_text = []
                current_relation = None

    return relations


def parse_ner_annotations():
    """
    Parse the NER annotations of the cardio dataset.
    :return: NER annotations in a unified format.
    """
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


def transform_cardio_annotation(annotation: dict):
    """
    Transform annotations into unified format.
    :param annotation: The annotation object
    """
    drugs = []
    # get drugs and active ingredients
    for index, tag in enumerate(annotation["ner_tags"]):
        if tag == "DRUG" or tag == "ACTIVEING":
            offsets = annotation["offsets"][index].split("-")
            drugs.append(
                {
                    "type": "MEDICATION",
                    "origin": clean_cardio_string(annotation["text"]),
                    "id": annotation["ids"][index].strip(),
                    "text": annotation["words"][index].strip(),
                    "in_narrative": annotation["in_narrative"][index],
                    "start": int(offsets[0]),
                    "end": int(offsets[1]),
                    "attributes": [],
                }
            )

    for relation_type in ["DURATION", "FREQUENCY", "STRENGTH", "FORM"]:
        for relation in get_medication_relations(annotation, relation_type):
            pattern = r"\[.*?\]"
            search_id = relation["relation"]
            search_id = re.sub(pattern, "", search_id)
            for drug in drugs:
                if drug["id"] == search_id:
                    drug["attributes"].append(
                        {
                            "attribute_label": relation_type,
                            "attribute": relation["relation_text"],
                        }
                    )

    if len(drugs) == 0:
        return {
            "id": [],
            "text": [],
            "origin": clean_cardio_string(annotation["text"]),
            "type": "NA",
            "end": [],
            "start": [],
            "attributes": [],
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
                    "attributes": lambda x: x.tolist(),
                }
            )
            .reset_index()
        )
        return grouped_df.to_dict(orient="records")[0]


def parse_annotations():
    """
    Parse the normal annotations (non NER) of the cardio dataset.
    :return: Annotations in a unified format.
    """
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
                "full_text": clean_cardio_string(full_text),
            }
        )

        logger.debug(
            f"Processed {len(annotations_file)} annotations from the file: {tsv_file}"
        )

    return annotations


def parse_cardio_heldout_text():
    """
    Parse the heldout texts of the cardio dataset. They have no annotations.
    :return: Heldout texts
    """
    txt_files = [
        file for file in os.listdir(TXT_HELDOUT_FOLDER_PATH) if file.endswith(".txt")
    ]

    heldout_text = []
    for filename in txt_files:
        with open(
            os.path.join(TXT_HELDOUT_FOLDER_PATH, filename), "r", encoding="utf-8"
        ) as file:
            full_text = file.read().strip()
            heldout_text.append(
                {"document": filename, "full_text": clean_cardio_string(full_text)}
            )

    logger.debug(f"Processed {len(heldout_text)} heldout texts")
    return heldout_text


def count_cardio_tokens():
    """
    Count the tokens of all the cardio documents.
    :return: The token count of the cardio documents.
    """
    texts = []
    txt_files = [file for file in os.listdir(TXT_FOLDER_PATH) if file.endswith(".txt")]
    for txt_file in txt_files:
        with open(
            os.path.join(TXT_FOLDER_PATH, txt_file), "r", encoding="utf-8"
        ) as file:
            full_text = file.read().strip()
            texts.append(full_text)

    return count_tokens(texts)


def build_cardio_db():
    """
    Build the cardio database.
    :return: None
    """
    annotations = parse_annotations()
    annotations_ner = parse_ner_annotations()
    heldout_text = parse_cardio_heldout_text()
    upload_data_to_mongodb(annotations, "corpus", "cardio", True, [])
    upload_data_to_mongodb(annotations_ner, "corpus", "cardio_ner", True, [])
    upload_data_to_mongodb(heldout_text, "corpus", "cardio_heldout", True, [])


if __name__ == "__main__":
    print(count_cardio_tokens())
    # build_cardio_db()
