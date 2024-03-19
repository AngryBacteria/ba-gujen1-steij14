import csv
import os

import requests
import re
from bs4 import BeautifulSoup, ResultSet, Tag, NavigableString
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


# WIKI PHP API
def get_category_members(category):
    """Get all members of a wikipedia category, including their page IDs."""
    S = requests.Session()
    URL = f"https://de.wikipedia.org/w/api.php"

    members = set()  # Storing tuples of (title, pageid)
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
            members.add((item["title"], item["pageid"]))  # Add a tuple to the set
        if "continue" not in data:
            break
        else:
            last_continue = data["continue"]

    return members


def get_articles_in_category(category):
    """Get all articles in a wikipedia category, including subcategories. Each article is represented
    by a tuple containing the title and page ID."""

    articles = set()
    members = get_category_members(category)

    for title, pageid in members:
        if title in ignore_list:
            continue
        if title.startswith("Kategorie:"):
            logger.debug(f"Parsing subcategory: {title}")
            subcategory = title.split("Kategorie:")[1]
            articles.update(get_articles_in_category(subcategory))
        else:
            articles.add((title, pageid))  # Add a tuple to the set

    return articles


def save_articles_by_category(category_name):
    output = get_articles_in_category(category_name)
    logger.debug(f"{len(output)} Articles found for the category: {category_name}")

    file_path = f"wikipedia_article_titles_{category_name}.txt"
    logger.debug(f"Saving articles into: {file_path}")
    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter="|")
        writer.writerow(["Title", "Page ID"])
        for article in output:
            writer.writerow(article)

    return output


# WEB SCRAPING
def clean_wikipedia_string(text: str):
    text = text.strip()
    pattern = r"\[.*\]"
    text = re.sub(pattern, "", text)
    return text


def get_disease_info_from_article(name: str, full_text: bool = False):
    """Get the data for a disease from a wikipedia article. Return None if no data is found."""
    link = f"https://de.wikipedia.org/wiki/{name}"
    response = requests.get(link)
    soup = BeautifulSoup(response.content, "html.parser")

    # get icd10 codes from the infobox
    icd10_infos = soup.find_all("div", class_="float-right")
    if icd10_infos is None or len(icd10_infos) == 0:
        logger.warning(f"No ICD-10 codes found for: {name}")
        return None
    codes = parse_icd10_table(icd10_infos)
    if codes is None or len(codes) == 0:
        logger.warning(f"No ICD-10 codes found for: {name}")
        return None

    # get the introduction text
    content_div = soup.find("div", class_="mw-content-ltr mw-parser-output")
    introduction_text = get_introduction_text(content_div)
    if introduction_text == "":
        logger.warning(f"No introduction text found for: {name}")
        return None

    if full_text:
        full_text = get_all_text(content_div)
    else:
        full_text = None

    logger.debug(f"Found ICD-10 {codes} codes for: {name}")
    return {
        "icd10": codes,
        "link": link,
        "name": name,
        "text": clean_wikipedia_string(introduction_text),
        "full_text": clean_wikipedia_string(full_text),
    }


def get_introduction_text(content_div: BeautifulSoup | NavigableString | None):
    introduction_text = ""
    for sibling in content_div.find("p").find_next_siblings():
        if isinstance(sibling, Tag):
            # If the sibling is a <p> tag, append its text to the introduction_text variable
            if sibling.name == "p":
                introduction_text += sibling.text + "\n"
            # If the sibling is an <h2> tag, break the loop as we've reached the end of the introduction
            elif sibling.name == "h2":
                break

    return introduction_text


def get_all_text(content_div: BeautifulSoup | NavigableString | None):
    text = ""
    for sibling in content_div.find("p").find_next_siblings():
        if isinstance(sibling, Tag):
            # If the sibling is a <p> tag, append its text to the introduction_text variable
            if sibling.name == "p":
                text += sibling.text + "\n"

    return text


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
