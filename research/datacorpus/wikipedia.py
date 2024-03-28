import csv
import os
from urllib.parse import quote_plus

import requests
import re
from bs4 import BeautifulSoup, ResultSet
from dotenv import load_dotenv
from pymongo import MongoClient

from research.datacorpus.scraping_utils import (
    process_ul,
    process_ol,
    process_dl,
    ignore_list,
)
from research.logger import logger

article_ignore_list = [
    "Kategorie:Liste (Krankheiten nach ICD-10)",
    "Kategorie:Liste (Krankheitsopfer)",
    "Krankheitsbild in der Tiermedizin",
    "Krankheitsbild in der Wehrmedizin",
    "Kategorie:Krankheitsbild in der Tiermedizin",
    "Kategorie:Hypovitaminose",
]


# WIKI PHP API
def get_category_members_ids(category):
    """Get all ids/titles of all members for a wikipedia category, including their page IDs.
    Category shoul be the name of the category without the "Kategorie:" prefix."""

    members = set()
    last_continue = {}
    encoded_category = category.replace(" ", "_")
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
            members.add((item["title"], item["pageid"]))
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
        if title in article_ignore_list:
            continue
        if title.startswith("Kategorie:"):
            logger.debug(f"Parsing subcategory: {title}")
            subcategory = title.split("Kategorie:")[1]
            articles.update(get_article_ids_of_category(subcategory))
        else:
            articles.add((title, pageid))

    return articles


def get_articles_views(title: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    encoded_title = quote_plus(title.replace(" ", "_"))
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/de.wikipedia/all-access/all-agents/{encoded_title}/monthly/2015100100/2024030100"
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
    pattern = r"\[.*?\]"
    text = re.sub(pattern, "", text)
    return text


def get_wikipedia_article_data(
    title: str, get_full_text: bool = True, get_icd10: bool = False
):
    """Get the data of a wikipedia article by title.
    If get_full_text is set to True, the full text of the article is returned.
    If get_icd10 is set to True, the ICD-10 codes are also returned."""
    # make request
    encoded_title = quote_plus(title.replace(" ", "_"))
    link = (
        f"https://de.wikipedia.org/api/rest_v1/page/html/{encoded_title}?redirect=false"
    )
    response = requests.get(link, allow_redirects=False)
    if not response.ok:
        logger.error(
            f"Request failed with status code ({response.status_code}) for url: {response.url}"
        )
        return None

    # get content div
    soup = BeautifulSoup(response.content, "html.parser")
    content_div = soup.find("body", class_=["mw-content-ltr", "mw-parser-output"])
    if content_div is None:
        logger.warning(f"No content div found for: {title}")
        return None

    # get the introduction text
    sections = content_div.find_all(name="section")
    introduction_text = get_all_text(sections, False)
    if introduction_text == "":
        logger.warning(f"No introduction text found for: {title}")
        return None

    # get the full text
    if get_full_text:
        sections = content_div.find_all(name="section")
        full_text = get_all_text(sections, True)
        if full_text == "":
            logger.warning(f"No full text found for: {title}")
            return None
    else:
        full_text = ""

    # get icd10 codes and return data
    if get_icd10:
        icd10_infos = soup.find_all("div", class_="float-right")
        if icd10_infos is None or len(icd10_infos) == 0:
            logger.warning(f"No ICD-10 codes found for: {title}")
            return None
        codes = parse_icd10_table(icd10_infos)
        if codes is None or len(codes) == 0:
            logger.warning(f"No ICD-10 codes found for: {title}")
            return None
        logger.debug(f"Found ICD-10 {codes} codes for: {title}")

        return {
            "title": title,
            "text": clean_wikipedia_string(introduction_text),
            "full_text": clean_wikipedia_string(full_text),
            "icd10": codes,
        }
    # return data without icd10 codes
    else:

        return {
            "title": title,
            "text": clean_wikipedia_string(introduction_text),
            "full_text": clean_wikipedia_string(full_text),
        }


def get_all_text(sections: ResultSet, full_text: bool) -> str:
    """Get text for a wikipedia article from the content div.
    If full_text is set to True, the full text of the article is returned.
    Otherwise, only the introduction text is returned."""
    text = ""
    if not full_text:
        sections = [sections[0]]

    stop_processing = False
    for section in sections:
        if stop_processing:
            break

        for child in section.children:
            if child.name in ["h2", "h3", "h4", "h5", "h6"]:
                if child.text.strip() in ignore_list:
                    stop_processing = True
                    break
                else:
                    text = text + "\n" + child.text.strip() + "\n"
            if child.name == "p":
                text += child.text.strip() + "\n"
            if child.name == "ul":
                text += process_ul(child)
            if child.name == "ol":
                text += process_ol(child)
            if child.name == "dl":
                text += process_dl(child)

    if text.startswith("- Wikidata:"):
        return ""
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
            title = doc["title"]
            views = get_articles_views(title)
            if views is not None:
                wikipedia_icd10_collection.update_one(
                    {"title": title}, {"$set": {"views": views}}
                )
                logger.debug(f"Updated Article ({title}) with {views} views.")
        except Exception as e:
            logger.error(f"Error for when getting views for article: {e}")
    client.close()


def build_wikipedia_icd10_db(
    category="Krankheit", file_exists=True, get_full_text=True
):
    """Build a MongoDB collection with the articles from a wikipedia category."""
    load_dotenv()
    client = MongoClient(os.getenv("MONGO_URL"))
    db = client.get_database("main")
    wikipedia_collection = db.get_collection("wikipedia")

    if not file_exists:
        save_article_ids_by_category(category)
    articles = read_article_ids_from_file(category)

    for title, _id in articles:
        try:
            existing_article = wikipedia_collection.find_one({"title": title})
            if existing_article is None:
                data = get_wikipedia_article_data(title, get_full_text, True)
                if data is not None:
                    data["category"] = category
                    wikipedia_collection.insert_one(data)
                    logger.debug(f"Uploaded {title}({_id}) to MongoDB.")
            else:
                logger.debug(f"Article {title}({_id}) already exists in MongoDB.")
        except Exception as e:
            logger.error(f"Failed to upload {title}({_id}): {e}")
    client.close()


# build_wikipedia_icd10_db("Krankheit", file_exists=True, get_full_text=True)
print(
    get_wikipedia_article_data("Kopfschmerz", get_full_text=True, get_icd10=True)[
        "full_text"
    ]
)
