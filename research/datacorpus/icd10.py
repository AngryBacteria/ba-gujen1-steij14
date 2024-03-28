import os

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
import xml.etree.ElementTree as et

from research.logger import logger

icd10who_alphabet_csv_path1 = "icd10who2019alpha_edvtxt_teil1_20180824.txt"
icd10who_alphabet_csv_path2 = "icd10who2019alpha_edvtxt_teil2_20180824.txt"
icd10who_xml_path = "icd10who2019syst_claml_20180824.xml"

icd10gm_alphabet_csv_path1 = "icd10gm2024alpha_edvtxt_20230929.txt"
icd10gm_xml_path = "icd10gm2024syst_claml_20230915.xml"
load_dotenv()


# Official CSV-Files. I recommend you use the XML-File instead.
def parse_csv_icd10_alphabet(icd10gm=False):
    """Parse an ICD-10 alphabet files and return a dataframe"""
    if icd10gm:
        columns = [
            "coding_type",
            "bfarm_id",
            "print_flag",
            "code",
            "asterisk_code",
            "additional_code",
            "code2",
            "title",
        ]
        data = pd.read_csv(
            icd10gm_alphabet_csv_path1,
            header=None,
            names=columns,
            sep="|",
        )
    else:
        columns = [
            "coding_type",
            "dimdi_id",
            "print_flag",
            "code",
            "asterisk_code",
            "code2",
            "title",
        ]
        alphabet1 = pd.read_csv(
            icd10who_alphabet_csv_path1,
            header=None,
            names=columns,
            sep="|",
        )
        alphabet2 = pd.read_csv(
            icd10who_alphabet_csv_path2,
            header=None,
            names=columns,
            sep="|",
        )
        data = pd.concat([alphabet1, alphabet2])
    data = data[data["code"].notnull()]

    logger.debug(f"Parsed icd10 alphabet data with {len(data)} rows.")
    return data.groupby("code").agg(lambda x: list(x.dropna())).reset_index()


# XML-File
def get_title_from_xml(element: et.Element, code: str) -> str | None:
    """Get the title from the preferred or preferredLong label of an ICD-10 element."""
    label_element = element.find(".//Rubric[@kind='preferredLong']/Label")
    if label_element is None or label_element.text is None:
        label_element = element.find(".//Rubric[@kind='preferred']/Label")
        if label_element is None or label_element.text is None:
            logger.error(f"No preferred label found for chapter {code}")
            return None
    if "Nicht belegte Schlüsselnummer" in label_element.text:
        return None

    return label_element.text


def get_super_class_from_xml(element: et.Element) -> str | None:
    """Get the super class code from the SuperClass element of an ICD-10 element."""
    super_class_element = element.find(".//SuperClass")
    super_class_code = super_class_element.get("code", None)
    return super_class_code


def get_text_from_xml(element: et.Element) -> str | None:
    """Get the text of the text-element from the Introduction element of an ICD-10 element."""
    output = ""
    text_rubrics = element.findall(".//Rubric[@kind='text']/Label/Para")
    for para in text_rubrics:
        output += para.text

    if output == "":
        return None
    return output


def parse_xml_icd10_categories(icd10gm=False, add_alphabet=False):
    """Parse the ICD-10 XML file and return a list of dictionaries with category data."""
    if icd10gm:
        tree = et.parse(icd10gm_xml_path)
    else:
        tree = et.parse(icd10who_xml_path)
    root = tree.getroot()

    # find all icd10 codes
    icd10_categories = []
    for category_element in root.findall(".//Class"):
        # pre-checks
        kind = category_element.attrib.get("kind", None)
        if kind != "category":
            continue
        code = category_element.attrib.get("code", None)
        if code is None:
            logger.error("No code found for category")
            continue

        # parse code
        usage = category_element.attrib.get("usage", None)
        if usage == "dagger":
            code = code + "†"
        if usage == "aster":
            code = code + "*"
        if icd10gm:
            if usage == "optional":
                code = code + "!"

        # parse labels
        title = get_title_from_xml(category_element, code)
        if title is None:
            continue

        # parse block (or superclass)
        super_class_code = get_super_class_from_xml(category_element)
        if super_class_code is None:
            logger.error(f"No block (or superclass) code found for category: {code}")
            continue

        # parse metadata
        meta_dict = {}
        for meta in category_element.findall("Meta"):
            meta_dict[meta.get("name", "")] = meta.get("value", "")

        # append to list
        icd10_categories.append(
            {
                "code": code,
                "super_class_code": super_class_code,
                "title": title,
                "synonyms": [],
                "type": "category",
                "meta": meta_dict,
            }
        )

    # add synonyms from the alphabet file
    if add_alphabet:
        alphabet_codes = parse_csv_icd10_alphabet(icd10gm)
        for category in icd10_categories:
            code = category["code"]
            alphabet_entries = alphabet_codes[alphabet_codes["code"] == code]
            if len(alphabet_entries) == 1:
                alphabet_entry = alphabet_entries.iloc[0]
                titles = alphabet_entry["title"]
                category_title = category["title"]
                synonyms = [title for title in titles if title != category_title]
                category["synonyms"] = synonyms

    logger.debug(
        f"Parsed category data from icd10_xml with {len(icd10_categories)} rows."
    )
    return icd10_categories


def parse_xml_icd10_blocks(icd10gm=False):
    if icd10gm:
        tree = et.parse(icd10gm_xml_path)
    else:
        tree = et.parse(icd10who_xml_path)
    root = tree.getroot()

    # find all icd10 blocks
    icd10_blocks = []
    for block_element in root.findall(".//Class"):
        # pre-checks
        kind = block_element.attrib.get("kind", None)
        if kind != "block":
            continue
        code = block_element.attrib.get("code", None)
        if code is None:
            logger.error("No code found for block")
            continue

        # parse labels
        title = get_title_from_xml(block_element, code)
        if title is None:
            continue

        # parse chapter (or superclass)
        super_class_code = get_super_class_from_xml(block_element)
        if super_class_code is None:
            logger.error(f"No chapter (or superclass) code found for block: {code}")
            continue

        # append to list
        icd10_blocks.append(
            {
                "code": code,
                "super_class_code": super_class_code,
                "title": title,
                "type": "block",
            }
        )

    logger.debug(f"Parsed block data from icd10_xml with {len(icd10_blocks)} rows.")
    return icd10_blocks


# todo: parse introduction text
# todo: parse note text
# todo: parse text text
def parse_xml_icd10_chapters(icd10gm=False):
    if icd10gm:
        tree = et.parse(icd10gm_xml_path)
    else:
        tree = et.parse(icd10who_xml_path)
    root = tree.getroot()

    # find all icd10 codes
    icd10_chapters = []
    for chapter_element in root.findall(".//Class"):
        # pre-checks
        kind = chapter_element.attrib.get("kind", None)
        if kind != "chapter":
            continue
        code = chapter_element.attrib.get("code", None)
        if code is None:
            logger.error("No code for chapter found")
            continue

        # parse labels
        title = get_title_from_xml(chapter_element, code)
        if title is None:
            continue

        # append to list
        icd10_chapters.append(
            {
                "code": code,
                "title": title,
                "type": "chapter",
            }
        )

    logger.debug(f"Parsed chapter data from icd10_xml with {len(icd10_chapters)} rows.")
    return icd10_chapters


# TODO: icd10gm : parse subcategory, for example M08.4 --> M08.43
def create_icd10_db_from_xml(icd10gm=True, add_alphabet=False):
    # parse xml files and combine
    categories = parse_xml_icd10_categories(add_alphabet)
    blocks = parse_xml_icd10_blocks(icd10gm)
    chapters = parse_xml_icd10_chapters(icd10gm)
    merged_output = categories + blocks + chapters

    # sort by code
    merged_output = sorted(merged_output, key=lambda x: x["code"])

    # Upload to MongoDB
    client = MongoClient(os.getenv("MONGO_URL"))
    db = client.get_database("main")
    collection_name = "icd10gm" if icd10gm else "icd10who"
    db.drop_collection(collection_name)
    icd10_collection = db.get_collection(collection_name)
    icd10_collection.insert_many(merged_output)

    logger.debug(f"Uploaded {len(merged_output)} rows to MongoDB.")
    client.close()


# entries = parse_xml_icd10_categories(icd10gm=True, add_alphabet=True)
# entries = parse_xml_icd10_blocks(icd10gm=True)
entries = parse_xml_icd10_chapters(icd10gm=True)
for entry in entries:
    print(entry)
