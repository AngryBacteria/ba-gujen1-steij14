import os

import requests
import re
from bs4 import BeautifulSoup, ResultSet
from dotenv import load_dotenv
from pymongo import MongoClient

ignore_list = [
    "Kategorie:Liste (Krankheiten nach ICD-10)",
    "Kategorie:Liste (Krankheitsopfer)",
    "Krankheitsbild in der Sportmedizin",
    "Krankheitsbild in der Tiermedizin",
    "Krankheitsbild in der Wehrmedizin",
    "Kategorie:Krankheitsbild in der Tiermedizin",
]


def get_category_members(category, print_url=False):
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

        if print_url:
            req = requests.Request("GET", URL, params=params).prepare()
            print(req.url)

        response = S.get(url=URL, params=params)
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
            subcategory = member.split("Kategorie:")[1]
            articles.update(get_articles_in_category(subcategory, lang))
        else:
            articles.add(member)

    return articles


def get_all_article_names(category_name, save_to_file=False):
    output = get_articles_in_category(category_name)
    print(f"{len(output)} Artikel Gefunden in der Kategorie {category_name}:")

    if save_to_file:
        content = "\n".join(output)
        with open("wikipedia_articles.txt", "w", encoding="utf-8") as file:
            file.write(content)

    return output


def get_disease_info_from_article(name: str):
    """Get the data for a disease from a wikipedia article. Return None if no data is found."""
    link = f"https://de.wikipedia.org/wiki/{name}"
    response = requests.get(link)
    soup = BeautifulSoup(response.content, "html.parser")

    # get icd10 codes from the infobox
    icd10_infos = soup.find_all("div", class_="float-right")
    if icd10_infos is None or len(icd10_infos) == 0:
        return None
    codes = parse_icd10_table(icd10_infos)
    if codes is None or len(codes) == 0:
        return None

    # get the disease text
    disease_text = soup.find("div", class_="mw-parser-output").find("p").text
    if disease_text is None:
        return None

    return {"icd10": codes, "text": disease_text, "link": link, "name": name}


def parse_icd10_table(icd10_infos: ResultSet):
    """Parse the ICD-10 table data from a wikipedia article. The infobox component is used."""
    output = []
    for icd10_info in icd10_infos:
        table = icd10_info.find("table")
        rows = table.find_all("tr")

        # check if icd10 who is used
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
