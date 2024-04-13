import os

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

from research.logger import logger

# DATA SOURCE: Medication_Pharmacode_ATC.xlsx

EXCEL_PATH = "F:\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\catalog\\atc\\Medication_Pharmacode_ATC.xlsx"


def read_atc_data():
    column_names = [
        "pharmacode",
        "productcode",
        "text",
        "name",
        "dose",
        "form",
        "package_type",
        "package_size",
        "atc",
        "ingredient",
        "manufacturer",
        "btm",
        "primary_indication",
    ]
    data = pd.read_excel(EXCEL_PATH, header=0)
    data.columns = column_names
    data = data.map(lambda x: x.strip() if isinstance(x, str) else x)
    data = data[data["pharmacode"] != ""]
    data = data[data["name"] != ""]
    data = data[data["text"] != ""]
    data = data[data["atc"] != ""]
    data = data[data["ingredient"] != ""]

    unique_combinations = data[
        ["name", "atc", "ingredient", "primary_indication"]
    ].drop_duplicates()
    unique_combinations = unique_combinations.to_dict(orient="records")

    load_dotenv()
    client = MongoClient(os.getenv("MONGO_URL"))
    db = client.get_database("catalog")
    collection_name = "atc"
    db.drop_collection(collection_name)
    atc_collection = db.get_collection(collection_name)
    atc_collection.insert_many(unique_combinations)
    logger.debug(f"Uploaded {len(unique_combinations)} rows to MongoDB.")
    client.close()


read_atc_data()
