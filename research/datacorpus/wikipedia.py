import csv
import os
from urllib.parse import quote_plus

import requests
import re
from bs4 import BeautifulSoup, ResultSet, Tag, NavigableString
from dotenv import load_dotenv
from pymongo import MongoClient
from research.logger import logger

ignore_list = [
    "Kategorie:Liste (Krankheiten nach ICD-10)",
    "Kategorie:Liste (Krankheitsopfer)",
    "Krankheitsbild in der Tiermedizin",
    "Krankheitsbild in der Wehrmedizin",
    "Kategorie:Krankheitsbild in der Tiermedizin",
    "Kategorie:Hypovitaminose",
]

disease_list = [
    "Kategorie:Krankheit",
]

general_medicine_list = ["Kategorie:Medizinische_Fachsprache"]


# WIKI PHP API
def get_category_members_ids(category):
    """Get all ids/titles of all members for a wikipedia category, including their page IDs.
    Category shoul be the name of the category without the "Kategorie:" prefix."""

    members = set()
    last_continue = {}

    encoded_category = quote_plus(category.replace(" ", "_"))
    while True:
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": f"Kategorie:{encoded_category}",
            "cmtype": "page|subcat",
            "cmlimit": 500,
            **last_continue,
        }

        response = requests.get(url="https://de.wikipedia.org/w/api.php", params=params)
        if not response.ok:
            logger.error(
                f"Request failed with status code ({response.status_code}) for url: {response.url}"
            )
            continue

        logger.debug(f"Getting category members from: {response.url}")
        data = response.json()

        for item in data["query"]["categorymembers"]:
            members.add((item["title"], item["pageid"]))  # Add a tuple to the set
        if "continue" not in data:
            break
        else:
            last_continue = data["continue"]

    return members


def get_article_ids_of_category(category):
    """Get all article ids/titles in a wikipedia category, including subcategories. Each article is represented
    by a tuple containing the title and page ID.
    Category should be the name of the category without the "Kategorie:" prefix."""

    articles = set()
    members = get_category_members_ids(category)

    for title, pageid in members:
        if title in ignore_list:
            continue
        if title.startswith("Kategorie:"):
            logger.debug(f"Parsing subcategory: {title}")
            subcategory = title.split("Kategorie:")[1]
            articles.update(get_article_ids_of_category(subcategory))
        else:
            articles.add((title, pageid))

    return articles


def get_articles_views(name: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    encoded_name = quote_plus(name.replace(" ", "_"))
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/de.wikipedia/all-access/all-agents/{encoded_name}/monthly/2015100100/2024030100"
    response = requests.get(url, headers=headers)
    if not response.ok:
        logger.error(
            f"Request failed with status code ({response.status_code}) for url: {response.url}"
        )
        return None
    else:
        output = 0
        data = response.json()
        for entry in data["items"]:
            output += entry["views"]
        return output


def save_article_ids_by_category(category_name):
    """Save the articles of a category into a file.
    The file will be named 'wikipedia_article_titles_{category_name}.txt'.
    The data is saved as a CSV in the format: Title|Page ID"""
    output = get_article_ids_of_category(category_name)

    file_path = f"wikipedia_article_titles_{category_name}.txt"
    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter="|")
        writer.writerow(["Title", "Page ID"])
        for article in output:
            writer.writerow(article)

    logger.debug(
        f"Saved {len(output)} articles from the {category_name} category into: {file_path}"
    )
    return output


def read_article_ids_from_file(category_name):
    """Read the articles of a category from a file generated by the save_article_ids_by_category function."""
    file_path = f"wikipedia_article_titles_{category_name}.txt"
    articles = set()
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="|")
        next(reader)
        for row in reader:
            articles.add((row[0], row[1]))
    logger.debug(f"Read {len(articles)} articles from: {file_path}")
    return articles


# WEB SCRAPING
def clean_wikipedia_string(text: str):
    """Clean a string from wikipedia by removing references and other unwanted elements."""
    text = text.strip()
    pattern = r"\[.*\]"
    text = re.sub(pattern, "", text)
    return text


# TODO: is there a way to handle "redirection" cases such as https://de.wikipedia.org/wiki/Ariboflavinose
def get_disease_info_from_article(
    name: str, pageid: str = "", get_full_text: bool = False
):
    """Get the data for a disease from a wikipedia article by its name. Returns None if no data is found.
    The data includes the ICD-10 codes, the introduction text and the link to the article.
    If the get_full_text is set to True, the full text of the article is also returned.
    """
    encoded_name = quote_plus(name.replace(" ", "_"))
    link = f"https://de.wikipedia.org/wiki/{encoded_name}"
    response = requests.get(link)
    if not response.ok:
        logger.error(
            f"Request failed with status code ({response.status_code}) for url: {response.url}"
        )
        return None

    print(response.history)

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

    if get_full_text:
        full_text = get_all_text(content_div)
    else:
        full_text = ""

    logger.debug(f"Found ICD-10 {codes} codes for: {name}")
    return {
        "icd10": codes,
        "name": name,
        "text": clean_wikipedia_string(introduction_text),
        "full_text": clean_wikipedia_string(full_text),
        "pageid": pageid,
    }


def get_introduction_text(
    content_div: BeautifulSoup | Tag | NavigableString | None,
) -> str:
    """Get the introduction text of a wikipedia article from the content div."""
    introduction_text = ""
    for child in content_div.descendants:
        if isinstance(child, Tag):
            if child.name == "p":
                introduction_text += child.text + "\n"
            elif child.name == "h2":
                break

    return introduction_text


def get_all_text(content_div: BeautifulSoup | Tag | NavigableString | None) -> str:
    """Get all text of a wikipedia article from the content div."""
    text = ""
    for child in content_div.descendants:
        if isinstance(child, Tag):
            if child.name == "p":
                text += child.text + "\n"

    return text


# TODO: also parse the following pattern: T20.- bis T32
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
            icd10_codes = re.findall(pattern, row_text)
            if icd10_codes is not None and len(icd10_codes) > 0:
                output.extend(icd10_codes)
    return output


def add_views_to_db(overwrite=False):
    """Add the views of the articles to the articles already present in the MongoDB database."""
    load_dotenv()
    client = MongoClient(os.getenv("MONGO_URL"))
    db = client.get_database("main")
    wikipedia_icd10_collection = db.get_collection("wikipedia_icd10")
    for doc in wikipedia_icd10_collection.find():
        try:
            if "views" in doc and not overwrite:
                continue
            title = doc["name"]
            views = get_articles_views(title)
            if views is not None:
                wikipedia_icd10_collection.update_one(
                    {"name": title}, {"$set": {"views": views}}
                )
                logger.debug(f"Updated Article ({title}) with {views} views.")
        except Exception as e:
            logger.error(f"Error for when getting views for article: {e}")
    client.close()


def build_wikipedia_icd10_db(file_exists=True, get_full_text=True, add_views=False):
    load_dotenv()
    client = MongoClient(os.getenv("MONGO_URL"))
    db = client.get_database("main")
    db.drop_collection("wikipedia_icd10")
    wikipedia_icd10_collection = db.get_collection("wikipedia_icd10")

    if not file_exists:
        save_article_ids_by_category("Krankheit")
    articles = read_article_ids_from_file("Krankheit")

    for title, _id in articles:
        try:
            data = get_disease_info_from_article(title, _id, get_full_text)
            if data is not None:
                wikipedia_icd10_collection.insert_one(data)
                logger.debug(f"Uploaded {title}({_id}) to MongoDB.")
        except Exception as e:
            logger.error(f"Failed to upload {title}({_id}): {e}")
    client.close()

    if add_views:
        add_views_to_db()
