import os
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pymongo import MongoClient

from research.logger import logger

load_dotenv()


def get_bund_text(code: str):
    """Get the text for an ICD10 code from gesund.bund.de"""
    encoded_code = quote(code.replace(".", "-"))
    link = f"https://gesund.bund.de/icd-code-suche/{encoded_code}"
    response = requests.get(link)
    if not response.ok:
        logger.error(
            f"Request failed with status code ({response.status_code}) for url: {response.url}"
        )
        return None

    soup = BeautifulSoup(response.content, "html.parser")
    article = soup.find("article")
    if article is None:
        logger.warning(f"No article found for: {code}")
        return None

    target_div = article.find("div", {"data-text-key": code})
    if target_div is None:
        logger.warning(f"Article text not found for code: {code}")
        return None

    return target_div.text.strip()


# TODO: check if doc exists first
def build_bund_db():
    """Update the MongoDB database with the text from gesund.bund.de"""
    client = MongoClient(os.getenv("MONGO_URL"))
    db = client.get_database("main")
    icd10gm_collection = db.get_collection("icd10gm")
    gesundbund_collection = db.get_collection("gesundbund")

    for doc in icd10gm_collection.find({"type": "category"}):
        try:
            code = doc["code"]
            text = get_bund_text(code)
            if text is not None:
                existing_doc = gesundbund_collection.find({"code": code})
                if existing_doc is None:
                    gesundbund_collection.insert_one({"code": code, "text": text})
                    logger.debug(f"Uploaded text from bund.de for {code}")
                else:
                    logger.debug(f"Document already exists for {code}")
        except Exception as e:
            logger.error(f"Error for {doc['code']}: {e}")
    client.close()


build_bund_db()
