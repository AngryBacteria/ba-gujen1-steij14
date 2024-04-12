import os

import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from pymongo import MongoClient

from research.logger import logger

corpus_path = "F:\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\jsyncc\\corpus.xml"


def find_with_default(element, tag, default_value=""):
    """Attempt to find the tag in the element, return text or a default value if not found or empty."""
    found_element = element.find(tag)
    if found_element is not None and found_element.text is not None and found_element.text.strip():
        return found_element.text.strip()
    return default_value


def build_jsyncc_db():
    tree = ET.parse(corpus_path)
    root = tree.getroot()
    documents = root.findall(".//document")
    # Convert XML to dictionary
    corpus = []
    for document in documents:
        doc_id = find_with_default(document, "id")
        id_long = find_with_default(document, "idLong")
        text = find_with_default(document, "text")
        doc_type = find_with_default(document, "type")
        heading = find_with_default(document, "heading")
        source = find_with_default(document, "source")

        corpus.append({
            "doc_id": doc_id,
            "id_long": id_long,
            "text": text,
            "doc_type": doc_type,
            "heading": heading,
            "source": source
        })

    # Upload to MongoDB
    load_dotenv()
    client = MongoClient(os.getenv("MONGO_URL"))
    db = client.get_database("main")
    db.drop_collection("jsyncc")
    jsyncc = db.get_collection("jsyncc")
    jsyncc.insert_many(corpus)
    logger.debug(f"Uploaded {len(corpus)} documents to jsyncc collection.")
    client.close()


build_jsyncc_db()
