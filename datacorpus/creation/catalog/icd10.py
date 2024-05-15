import xml.etree.ElementTree as et

import pandas as pd
from pandas import DataFrame

from shared.mongodb import upload_data_to_mongodb
from shared.logger import logger

# DATA SOURCE: https://www.bfarm.de/DE/Kodiersysteme/Klassifikationen/ICD/ICD-10-GM/_node.html
# DATA SOURCE: https://www.bfarm.de/DE/Kodiersysteme/Klassifikationen/ICD/ICD-10-WHO/_node.html

# Paths to the various ICD10gm and ICD10who files. Includes alphabet files as well
icd10who_alphabet_csv_path1 = "icd10who2019alpha_edvtxt_teil1_20180824.txt"
icd10who_alphabet_csv_path2 = "icd10who2019alpha_edvtxt_teil2_20180824.txt"
icd10who_xml_path = "icd10who2019syst_claml_20180824.xml"
icd10gm_alphabet_csv_path1 = "icd10gm2024alpha_edvtxt_20230929.txt"
icd10gm_xml_path = "icd10gm2024syst_claml_20230915.xml"


def parse_csv_icd10_alphabet(icd10gm=False) -> DataFrame:
    """Parse an ICD-10 alphabet files and return a dataframe
    :param icd10gm: If true, parse the ICD-10-GM , otherwise the ICD-10-WHO
    """
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
        data = data.map(lambda x: x.strip() if isinstance(x, str) else x)
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
        data = data.map(lambda x: x.strip() if isinstance(x, str) else x)
    data = data[data["code"].notnull()]

    if icd10gm:
        logger.debug(f"Parsed icd10gm alphabet data with {len(data)} rows.")
    else:
        logger.debug(f"Parsed icd10 alphabet data with {len(data)} rows.")
    return data.groupby("code").agg(lambda x: list(x.dropna())).reset_index()


def get_title_from_xml(element: et.Element, code: str) -> str | None:
    """Get the title from the preferred or preferredLong label of an ICD-10 element.
    :param element: The ICD-10 element to extract the title from
    :param code: The code of the ICD-10 element (only used for logging)
    """
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
    """Get the super class code from the SuperClass element of an ICD-10 element.
    :param element: The ICD-10 element to extract the super class from
    """
    super_class_element = element.find(".//SuperClass")
    super_class_code = super_class_element.get("code", None)
    return super_class_code


def parse_xml_icd10_categories(icd10gm=False, add_alphabet=False) -> list[dict]:
    """Parse the ICD-10 XML file and return a list of dictionaries with category data.
    :param icd10gm: If true, parse the ICD-10-GM , otherwise the ICD-10-WHO
    :param add_alphabet: If true, add synonyms from the alphabet files
    """
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

    if icd10gm:
        logger.debug(
            f"Parsed category data from icd10gm xml with {len(icd10_categories)} rows."
        )
    else:
        logger.debug(
            f"Parsed category data from icd10 xml with {len(icd10_categories)} rows."
        )
    return icd10_categories


def parse_xml_icd10_blocks(icd10gm=False) -> list[dict]:
    """Parse the ICD-10 XML file and return a list of dictionaries with the icd block data.
    :param icd10gm: If true, parse the ICD-10-GM , otherwise the ICD-10-WHO
    """
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

    if icd10gm:
        logger.debug(
            f"Parsed block data from icd10gm xml with {len(icd10_blocks)} rows."
        )
    else:
        logger.debug(f"Parsed block data from icd10 xml with {len(icd10_blocks)} rows.")
    return icd10_blocks


def parse_xml_icd10_chapters(icd10gm=False) -> list[dict]:
    """Parse the ICD-10 XML file and return a list of dictionaries with the icd chapter data.
    :param icd10gm: If true, parse the ICD-10-GM , otherwise the ICD-10-WHO
    """
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

    if icd10gm:
        logger.debug(
            f"Parsed chapter data from icd10gm xml with {len(icd10_chapters)} rows."
        )
    else:
        logger.debug(
            f"Parsed chapter data from icd10 xml with {len(icd10_chapters)} rows."
        )
    return icd10_chapters


def create_icd10_db_from_xml(icd10gm=True, add_alphabet=False) -> None:
    """Parse the ICD-10 XML files and create a MongoDB collection with the data.
    :param icd10gm: If true, parse the ICD-10-GM , otherwise the ICD-10-WHO
    :param add_alphabet: If true, add synonyms from the alphabet files
    """
    # parse xml files and combine
    categories = parse_xml_icd10_categories(add_alphabet=add_alphabet, icd10gm=icd10gm)
    blocks = parse_xml_icd10_blocks(icd10gm=icd10gm)
    chapters = parse_xml_icd10_chapters(icd10gm=icd10gm)
    merged_output = categories + blocks + chapters

    # sort by code
    merged_output = sorted(merged_output, key=lambda x: x["code"])

    # Upload to MongoDB
    collection_name = "icd10gm" if icd10gm else "icd10who"
    upload_data_to_mongodb(merged_output, "catalog", collection_name, True, ["code"])


create_icd10_db_from_xml(icd10gm=True, add_alphabet=True)
create_icd10_db_from_xml(icd10gm=False, add_alphabet=True)
