# DATA SOURCE: https://heidata.uni-heidelberg.de/dataset.xhtml?persistentId=doi:10.11588/data/AFYQDY
import logging
import os.path

from research.datacorpus.utils.utils_mongodb import upload_data_to_mongodb
from research.logger import logger

TSV_FOLDER_PATH = "F:\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\corpus\\cardiode\\tsv"
TXT_FOLDER_PATH = "F:\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\corpus\\cardiode\\txt"


def parse_annotations():
    # get all filenames in the folder TSV_FOLDER_PATH
    tsv_files = [file for file in os.listdir(TSV_FOLDER_PATH) if file.endswith(".tsv")]

    annotations = []

    current_words = []
    current_ner_tags = []
    current_offsets = []
    current_text = []
    current_in_narrative = []
    current_relations = []
    current_text_categories = []
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
                            {
                                "words": current_words,
                                "ner_tags": current_ner_tags,
                                "offsets": current_offsets,
                                "in_narrative": current_in_narrative,
                                "relations": current_relations,
                                "text_categories": current_text_categories,
                                "text": current_text,
                            }
                        )
                    current_text = line.split("#Text=")[1]
                    current_words = []
                    current_ner_tags = []
                    current_offsets = []
                    current_in_narrative = []
                    current_relations = []
                    current_text_categories = []
                    continue

                # get the values from the line
                parts = line.split("\t")
                offset = parts[1]
                word = parts[2]
                annotation = parts[3]
                in_narrative = parts[4]
                relation = parts[5]
                text_category = parts[6]

                # save the values
                current_words.append(word)
                current_ner_tags.append(annotation)
                current_offsets.append(offset)
                current_in_narrative.append(in_narrative)
                current_relations.append(relation)
                current_text_categories.append(text_category)

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
        logger.debug(f"Processed {len(annotations)} annotations from the {tsv_file}")

    return annotations


def build_cardio_db():
    annotations = parse_annotations()
    upload_data_to_mongodb(annotations, "corpus", "cardio", True, [])


# TODO: remove the psuedo text parts (problem is in the annotations) <pseudo> and <B-PER>
build_cardio_db()
