import chromadb

from shared.logger import logger

chroma_client = chromadb.PersistentClient()


def init_chroma_db(fill_mongodb=False):
    chroma_client.get_or_create_collection("icd10")
    chroma_client.get_or_create_collection("atc")
    chroma_client.get_or_create_collection("ops")
    logger.debug("Initialized ChromaDB.")


def get_collection(name: str):
    return chroma_client.get_or_create_collection(name)


def fill_collection(
    collection_name: str, values: list[str], metadata: list[dict], ids: list[str]
):
    collection = get_collection(collection_name)
    collection.add(documents=values, metadatas=metadata, ids=ids)
    logger.debug(f"Inserted {len(values)} documents into collection {collection_name}.")


if __name__ == "__main__":
    init_chroma_db()
