import json
import os.path

from dotenv import load_dotenv
from pymongo import MongoClient

from research.logger import logger

fine_annotations_folder = "F:\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\ggponc2\\json\\fine"


def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


# Load data from both files
data_short = load_json(os.path.join(fine_annotations_folder, 'short', 'all.json'))
data_long = load_json(os.path.join(fine_annotations_folder, 'long', 'all.json'))

# Upload to MongoDB
load_dotenv()
client = MongoClient(os.getenv("MONGO_URL"))
db = client.get_database("main")
db.drop_collection("ggponc_short")
db.drop_collection("ggponc_long")
ggponc_short = db.get_collection("ggponc_short")
ggponc_long = db.get_collection("ggponc_long")
ggponc_short.insert_many(data_short)
ggponc_long.insert_many(data_long)
client.close()
logger.debug(f"Uploaded {len(data_short)} short and {len(data_long)} long documents to ggponc collection.")
