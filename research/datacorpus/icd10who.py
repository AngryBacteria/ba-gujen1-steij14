import os

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
import xml.etree.ElementTree as et

from research.logger import logger

icd10_alphabet_path1 = "icd10who2019alpha_edvtxt_teil1_20180824.txt"
icd10_alphabet_path2 = "icd10who2019alpha_edvtxt_teil2_20180824.txt"
icd10_metadata_path = "icd10who2019syst_kodes.txt"
icd10_xml = "icd10who2019syst_claml_20180824.xml"
load_dotenv()


# Official CSV-Files. I recommend you use the XML-File instead.
def parse_icd10who_alphabet(path):
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


def create_icd10_db_from_csv():
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
    client = MongoClient(os.getenv("MONGO_URL"))
    db = client.get_database("main")
    db.drop_collection("icd10who")
    collection = db.get_collection("icd10who")
    collection.insert_many(merged_output.to_dict(orient="records"))
    logger.debug(f"Uploaded {len(merged_output)} rows to MongoDB.")
    client.close()


# XML-File
def parse_xml_icd10():
    tree = et.parse(icd10_xml)
    root = tree.getroot()

    # find all icd10 classes
    icd10_entries = []
    for class_element in root.findall(".//Class"):
        # pre-checks
        kind = class_element.attrib.get('kind', None)
        if kind != 'category':
            continue
        code = class_element.attrib.get('code', None)
        if code is None:
            logger.error("No code info found")
            continue

        # parse code
        usage = class_element.attrib.get('usage', None)
        if usage == 'dagger':
            code = code + "+"
        if usage == 'aster':
            code = code + "*"

        # parse labels
        # TODO: handle "Nicht belegte Schlüsselnummer"
        preferred_label = class_element.find(".//Rubric[@kind='preferred']/Label")
        if preferred_label is None:
            logger.error(f"No preferred label found for class {code}")
            continue

        # parse metadata
        rare_disease_element = class_element.find(".//Meta[@name='RareDisease']")
        sex_code_element = class_element.find(".//Meta[@name='SexCode']")
        sex_reject_element = class_element.find(".//Meta[@name='SexReject']")
        age_low_element = class_element.find(".//Meta[@name='AgeLow']")
        age_high_element = class_element.find(".//Meta[@name='AgeHigh']")

        rare_disease = rare_disease_element.get('value', None)
        sex_code = sex_code_element.get('value', None)
        sex_reject = sex_reject_element.get('value', None)
        age_low = age_low_element.get('value', None)
        age_high = age_high_element.get('value', None)

        icd10_entries.append({
            "code": code,
            "title": preferred_label.text,
            "rare_disease": rare_disease,
            "sex_code": sex_code,
            "sex_reject": sex_reject,
            "age_low": age_low,
            "age_high": age_high
        })

    for entry in icd10_entries:
        print(entry)


parse_xml_icd10()
