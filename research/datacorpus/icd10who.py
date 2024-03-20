import os
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pymongo import MongoClient

from research.logger import logger

icd10_alphabet_path1 = "icd10who2019alpha_edvtxt_teil1_20180824.txt"
icd10_alphabet_path2 = "icd10who2019alpha_edvtxt_teil2_20180824.txt"
icd10_metadata_path = "icd10who2019syst_kodes.txt"


# Official CSV-Files
def parse_icd10who_alphabet(path, filter_special_classes=False):
    """Parse the ICD-10-WHO alphabet files and return a dataframe"""
    columns = [
        "coding_type",
        "dimdi_id",
        "print_flag",
        "code",
        "asterisk_code",
        "code2",
        "title",
    ]
    data = pd.read_csv(
        path,
        header=None,
        names=columns,
        sep="|",
        usecols=["code", "asterisk_code", "code2", "title"],
    )

    data = data[data["code"].notnull()]
    if filter_special_classes:
        filter_condition = ~data["code"].str.contains("[+*]", na=False)
        data = data[filter_condition]

    logger.debug(f"Parsed data from {path} with {len(data)} rows.")

    return data.groupby("code").agg(lambda x: list(x.dropna())).reset_index()


def parse_icd10who_metadata():
    """Parse the ICD-10-WHO metadata file and return a dataframe"""
    columns = [
        "classification_level",
        "position_in_tree",
        "type_of_four_five_digit_code",
        "chapter_number",
        "first_three_digit_code",
        "code_without_cross",
        "code_without_dash_star",
        "code_without_dot_star",
        "class_title",
        "three_digit_code_title",
        "four_digit_code_title",
        "five_digit_code_title",
        "mortality_list_1_reference",
        "mortality_list_2_reference",
        "mortality_list_3_reference",
        "mortality_list_4_reference",
        "morbidity_list_reference",
        "gender_association",
        "gender_error_type",
        "min_age",
        "max_age",
        "age_error_type",
        "rare_in_central_europe",
        "allowed_as_underlying_cause",
    ]
    data = pd.read_csv(
        icd10_metadata_path,
        header=None,
        names=columns,
        sep=";",
        usecols=["class_title", "code_without_dash_star"],
    )
    data = data[data["code_without_dash_star"].notnull()]

    logger.debug(f"Parsed data from {icd10_metadata_path} with {len(data)} rows.")

    return data


def merge_fields(group):
    """
    Custom function to merge fields within a group of rows having the same 'code'.
    """
    # Example of merging lists for 'asterisk_code' and 'code2', and concatenating titles
    asterisk_codes = sum(group["asterisk_code"], [])
    unique_asterisk_codes = list(set(asterisk_codes))

    codes2 = sum(group["code2"], [])
    unique_code2 = list(set(codes2))

    titles = sum(group["title"], [])
    unique_titles = list(set(titles))

    merged_row = {
        "asterisk_code": unique_asterisk_codes,
        "code2": unique_code2,
        "title": unique_titles,
    }
    return pd.Series(merged_row)


# WEBSCRAPING
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
        logger.warning("Article text not found")
        return None

    return target_div.text.strip()


def update_db_with_bund_texts(overwrite=False):
    """Update the MongoDB database with the text from gesund.bund.de"""
    load_dotenv()
    client = MongoClient(os.getenv("MONGO_URL"))
    db = client.get_database("main")
    collection = db.get_collection("icd10who")

    for doc in collection.find():
        try:
            code = doc["code"]
            if "bund_text" in doc and not overwrite:
                continue
            text = get_bund_text(code)
            if text is not None:
                collection.update_one({"code": code}, {"$set": {"bund_text": text}})
                logger.debug(f"Updated {code} with text from bund.de")
        except Exception as e:
            logger.error(f"Error for {doc['code']}: {e}")
    client.close()


def create_icd10_db(add_bund_text=False):
    # parse both alphabet files
    alphabet1 = parse_icd10who_alphabet(icd10_alphabet_path1)
    alphabet2 = parse_icd10who_alphabet(icd10_alphabet_path2)
    output = pd.concat([alphabet1, alphabet2])

    # parse metadata file
    metadata = parse_icd10who_metadata()
    metadata = metadata.rename(
        columns={"code_without_dash_star": "code", "class_title": "title"}
    )
    metadata["asterisk_code"] = [[] for _ in range(len(metadata))]
    metadata["code2"] = [[] for _ in range(len(metadata))]
    metadata["title"] = metadata["title"].apply(lambda x: [x])

    # Combine alphabet and metadata and merge rows
    output = pd.concat([output, metadata])
    merged_output = (
        output.groupby("code").apply(merge_fields, include_groups=False).reset_index()
    )

    # Upload to MongoDB
    load_dotenv()
    client = MongoClient(os.getenv("MONGO_URL"))
    db = client.get_database("main")
    db.drop_collection("icd10who")
    collection = db.get_collection("icd10who")
    collection.insert_many(merged_output.to_dict(orient="records"))
    logger.debug(f"Uploaded {len(merged_output)} rows to MongoDB.")
    client.close()

    # Update with bund.de text
    if add_bund_text:
        update_db_with_bund_texts()
        logger.debug("Updated with bund.de text")
