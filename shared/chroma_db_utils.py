import chromadb

from shared.logger import logger
from shared.mongodb import get_collection

chroma_client = chromadb.PersistentClient()


def init_chroma_db(mongo_collections=None):
    if mongo_collections is None:
        mongo_collections = []

    chroma_client.get_or_create_collection("icd10gm")
    chroma_client.get_or_create_collection("atc")
    chroma_client.get_or_create_collection("ops")
    logger.debug("Initialized ChromaDB.")

    # TODO: add synonyms from icd10gm / ops
    for mcollection in mongo_collections:
        mongo_collection = get_collection("catalog", mcollection)
        mongo_docs = list(mongo_collection.find({}))
        mongo_values = [doc["title"] for doc in mongo_docs]
        mongo_metadata = [{"code": doc["code"]} for doc in mongo_docs]
        mongo_ids = [doc["code"] for doc in mongo_docs]
        fill_collection(mcollection, mongo_values, mongo_metadata, mongo_ids)


def get_chroma_collection(name: str):
    return chroma_client.get_or_create_collection(name)


def fill_collection(
    collection_name: str, values: list[str], metadata: list[dict], ids: list[str]
):
    collection = get_chroma_collection(collection_name)
    collection.add(documents=values, metadatas=metadata, ids=ids)
    logger.debug(f"Inserted {len(values)} documents into collection {collection_name}.")


def find_chroma_data(collection_name: str, query: str):
    collection = get_chroma_collection(collection_name)
    return collection.query(query_texts=[query], n_results=2)


if __name__ == "__main__":
    init_chroma_db([])
