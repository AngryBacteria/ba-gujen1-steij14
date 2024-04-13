import json
import os.path

from dotenv import load_dotenv
from pymongo import MongoClient

from research.datacorpus.corpus.utils_ner import parse_ner_dataset
from research.logger import logger

fine_annotations_folder = "F:\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\corpus\\ggponc2\\json\\fine"
fine_annotations_ner_folder = "F:\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\corpus\\ggponc2\\conll\\fine"


def load_json(filename):
    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)

    logger.debug(f"Parsed {len(data)} GGPONC json annotations from file {filename}")
    return data


def get_ggponc_json():
    """Load the GGPONC json data from the fine annotations folder and return the short and long documents."""
    # Load data from both files
    data_short = load_json(os.path.join(fine_annotations_folder, "short", "all.json"))
    data_long = load_json(os.path.join(fine_annotations_folder, "long", "all.json"))
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

    load_dotenv()
    client = MongoClient(os.getenv("MONGO_URL"))
    db = client.get_database("corpus")

    # json data
    db.drop_collection("ggponc_short")
    db.drop_collection("ggponc_long")
    ggponc_short = db.get_collection("ggponc_short")
    ggponc_long = db.get_collection("ggponc_long")
    ggponc_short.insert_many(data_short_json)
    ggponc_long.insert_many(data_long_json)

    logger.debug(
        f"Uploaded {len(data_short_json)} short and {len(data_long_json)} long json annotations to ggponc collection."
    )

    # ner data
    db.drop_collection("ggponc_short_ner")
    db.drop_collection("ggponc_long_ner")
    ggponc_short_ner = db.get_collection("ggponc_short_ner")
    ggponc_long_ner = db.get_collection("ggponc_long_ner")
    ggponc_short_ner.insert_many(data_short_ner)
    ggponc_long_ner.insert_many(data_long_ner)
    logger.debug(
        f"Uploaded {len(data_short_ner)} short and {len(data_long_ner)} long NER annotations to ggponc collection."
    )

    client.close()


build_ggponc_db()
