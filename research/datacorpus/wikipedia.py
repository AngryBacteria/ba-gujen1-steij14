import os

import requests
import re
from bs4 import BeautifulSoup, ResultSet
from dotenv import load_dotenv
from pymongo import MongoClient

from research.logger import logger

ignore_list = [
    "Kategorie:Liste (Krankheiten nach ICD-10)",
    "Kategorie:Liste (Krankheitsopfer)",
    "Krankheitsbild in der Sportmedizin",
    "Krankheitsbild in der Tiermedizin",
    "Krankheitsbild in der Wehrmedizin",
    "Kategorie:Krankheitsbild in der Tiermedizin",
]


# WIKI API
def get_category_members(category):
    """Get all members of a wikipedia category."""
    S = requests.Session()
    URL = f"https://de.wikipedia.org/w/api.php"

    titles = []
    last_continue = {}

    while True:
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": f"Kategorie:{category}",
            "cmtype": "page|subcat",
            "cmlimit": 500,
            **last_continue,
        }

        response = S.get(url=URL, params=params)
        logger.debug(response.url)
        data = response.json()

        for item in data["query"]["categorymembers"]:
            titles.append(item["title"])
        if "continue" not in data:
            break
        else:
            last_continue = data["continue"]

    return titles


def get_articles_in_category(category, lang="de"):
    """Get all articles in a wikipedia category. This includes subcategories."""
    articles = set()
    members = get_category_members(category)

    for member in members:
        if member in ignore_list:
            continue
        if member.startswith("Kategorie:"):
            logger.debug(f"Parsing subcategory: {member}")
            subcategory = member.split("Kategorie:")[1]
            articles.update(get_articles_in_category(subcategory, lang))
        else:
            articles.add(member)

    return articles


def get_all_article_names(category_name, save_to_file=False):
    output = get_articles_in_category(category_name)
    logger.debug(f"{len(output)} Articles found for the category: {category_name}")

    if save_to_file:
        content = "\n".join(output)
        file_path = "wikipedia_articles.txt"
        logger.debug(f"Saving articles into {file_path}")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)

    return output


# WEB SCRAPING
def get_disease_info_from_article(name: str):
    """Get the data for a disease from a wikipedia article. Return None if no data is found."""
    link = f"https://de.wikipedia.org/wiki/{name}"
    response = requests.get(link)
    soup = BeautifulSoup(response.content, "html.parser")

    # get icd10 codes from the infobox
    icd10_infos = soup.find_all("div", class_="float-right")
    if icd10_infos is None or len(icd10_infos) == 0:
        logger.debug(f"No ICD-10 codes found for: {name}")
        return None
    codes = parse_icd10_table(icd10_infos)
    if codes is None or len(codes) == 0:
        logger.debug(f"No ICD-10 codes found for: {name}")
        return None

    # get the disease text
    disease_text = soup.find("div", class_="mw-parser-output").find("p").text
    if disease_text is None:
        logger.debug(f"No disease text found for: {name}")
        return None

    logger.debug(f"Found ICD-10 {codes} codes for: {name}")
    return {"icd10": codes, "text": disease_text, "link": link, "name": name}


def parse_icd10_table(icd10_infos: ResultSet):
    """Parse the ICD-10 table data from a wikipedia article. The infobox component is used."""
    output = []
    for icd10_info in icd10_infos:
        table = icd10_info.find("table")
        if table is None:
            continue
        rows = table.find_all("tr")
        if rows is None or len(rows) == 0:
            continue

        # check if icd10-who is used for article
        if rows[0] is not None:
            title = rows[0].text.replace("\n", " ").strip()
            if title != "Klassifikation nach ICD-10":
                continue
        if rows[len(rows) - 1] is not None:
            title = rows[len(rows) - 1].text.replace("\n", " ").strip()
            if title != "ICD-10 online (WHO-Version 2019)":
                continue

        for row in rows:
            row_text = row.text.replace("\n", " ").strip()
            pattern = r"\b([A-Z][0-9]{2}(?:\.[0-9]{1,2})?[\+\*]?)\b"
            icd10_code = re.search(pattern, row_text)
            if icd10_code is not None:
                output.append(icd10_code.group())
    return output


def upload_to_mongodb(data):
    load_dotenv()
    client = MongoClient(os.getenv("MONGO_URL"))
    db = client.get_database("main")
    db.drop_collection("wikipedia")
    collection = db.get_collection("wikipedia")
    collection.insert_many(data)
    logger.debug(f"Uploading {len(data)} rows to MongoDB.")
    client.close()
