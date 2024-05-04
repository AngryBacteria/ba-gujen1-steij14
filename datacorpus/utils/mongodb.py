import os

from dotenv import load_dotenv
from pymongo import MongoClient

from shared.logger import logger

load_dotenv()


def upload_data_to_mongodb(
    data: list[dict],
    database_name: str,
    collection_name: str,
    drop_collection: bool,
    index_names: list[str],
):
    """Upload data to MongoDB."""
    client = MongoClient(os.getenv("MONGO_URL"))
    db = client.get_database(database_name)
    if drop_collection:
        db.drop_collection(collection_name)
    collection = db.get_collection(collection_name)
    if index_names is None:
        index_names = []
    for index_name in index_names:
        collection.create_index(index_name, unique=True)
    collection.insert_many(data)

    client.close()
    logger.debug(
        f"Uploaded {len(data)} rows to MongoDB database {database_name} collection {collection_name}"
    )


def get_collection(database_name: str, collection_name: str):
    """Get a collection from MongoDB."""
    client = MongoClient(os.getenv("MONGO_URL"))
    db = client.get_database(database_name)
    collection = db.get_collection(collection_name)
    return collection


def rename_dict_keys(data_list, old_key, new_key):
    """Rename a key in a list of dictionaries."""
    for item in data_list:
        if old_key in item:
            item[new_key] = item.pop(old_key)
    return data_list