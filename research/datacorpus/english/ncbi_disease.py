import os

from research.datacorpus.corpus.utils_ner import parse_ner_dataset
from research.datacorpus.utils_mongodb import upload_data_to_mongodb

# INFO: https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/
# DATA SOURCE: https://github.com/spyysalo/ncbi-disease

TSV_FOLDER_PATH = "F:\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\corpus_en\\NCBI_Disease"
ANNOTATION_FILE_NAMES = [
    "devel.tsv",
    "test.tsv",
    "train.tsv",
]


def build_ncbi_disease_db():
    data = []
    for filepath in ANNOTATION_FILE_NAMES:
        path = os.path.join(TSV_FOLDER_PATH, filepath)
        annotations_file = parse_ner_dataset(path)
        data.extend(annotations_file)

    upload_data_to_mongodb(data, "corpus_en", "ncbi_disease", True, [])


build_ncbi_disease_db()
