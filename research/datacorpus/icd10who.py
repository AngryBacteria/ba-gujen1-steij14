import os

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
client = MongoClient(os.getenv('MONGO_URL'))
icd10_alphabet_path = "icd10who2019alpha_edvtxt_teil1_20180824.txt"
icd10_metadata_path = "icd10who2019syst_kodes.txt"


def parse_icd10who_alphabet():
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
        icd10_alphabet_path,
        header=None,
        names=columns,
        sep="|"
    )
    data = data[data['code'].notnull()]
    filter_condition = ~data['code'].str.contains('[+*]', na=False)
    data = data[filter_condition]
    return data.groupby("code")['title'].apply(list).reset_index()


def parse_icd10who_metadata():
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
    )
    return data


def upload_to_mongodb():
    data = parse_icd10who_alphabet()
    db = client.get_database("main")
    collection = db.get_collection("icd10who")
    collection.insert_many(data.to_dict(orient="records"))


print(parse_icd10who_alphabet())