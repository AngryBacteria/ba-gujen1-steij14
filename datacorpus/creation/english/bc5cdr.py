import json
import os.path

from datacorpus.utils.mongodb import upload_data_to_mongodb

# PAPER: https://academic.oup.com/database/article/doi/10.1093/database/baw068/2630414?login=true
# DATA SOURCE: https://huggingface.co/datasets/tner/bc5cdr/tree/main/dataset

JSON_FOLDER_PATH = "F:\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\corpus_en\\BC5CDR"
ANNOTATION_FILE_NAMES = [
    "dataset_test.json",
    "dataset_train.json",
    "dataset_valid.json",
]


def transform_label(label: str):
    """Transforms the label to the format used in the MongoDB database."""
    if label == 0:
        return "O"
    elif label == 1:
        return "B-Chemical"
    elif label == 2:
        return "B-DIAG"
    elif label == 3:
        return "I-DIAG"
    elif label == 4:
        return "I-Chemical"
    else:
        raise ValueError(f"Unknown label: {label}")


def build_bc5cdr_db():
    """Builds the BC5CDR database in MongoDB."""
    data = []
    for name in ANNOTATION_FILE_NAMES:
        path = os.path.join(JSON_FOLDER_PATH, name)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                data.append({"ner-tags": obj["tags"], "words": obj["tokens"]})

    for obj in data:
        obj["ner-tags"] = [transform_label(tag) for tag in obj["ner-tags"]]
    upload_data_to_mongodb(data, "corpus_en", "bc5cdr", True, [])


build_bc5cdr_db()
