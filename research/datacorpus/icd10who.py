import os

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
client = MongoClient(os.getenv('MONGO_URL'))
icd10_alphabet_path1 = "icd10who2019alpha_edvtxt_teil1_20180824.txt"
icd10_alphabet_path2 = "icd10who2019alpha_edvtxt_teil2_20180824.txt"
icd10_metadata_path = "icd10who2019syst_kodes.txt"


def parse_icd10who_alphabet(path, filter_special_classes=False):
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
        usecols=["code", "asterisk_code", "code2", "title"]
    )

    data = data[data['code'].notnull()]
    if filter_special_classes:
        filter_condition = ~data['code'].str.contains('[+*]', na=False)
        data = data[filter_condition]

    return data.groupby('code').agg(lambda x: list(x.dropna())).reset_index()


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
        usecols=["class_title", "code_without_dash_star"]
    )
    data = data[data['code_without_dash_star'].notnull()]

    return data


def upload_to_mongodb(dataframe):
    db = client.get_database("main")
    db.drop_collection("icd10who")
    collection = db.get_collection("icd10who")
    collection.insert_many(dataframe.to_dict(orient="records"))


pd.set_option('display.max_columns', None)
# parse both alphabet files
alphabet1 = parse_icd10who_alphabet(icd10_alphabet_path1)
alphabet2 = parse_icd10who_alphabet(icd10_alphabet_path2)
output = pd.concat([alphabet1, alphabet2])

# parse metadata file
metadata = parse_icd10who_metadata()
metadata = metadata[~metadata['code_without_dash_star'].str.contains("\.")]
metadata = metadata.rename(columns={"code_without_dash_star": "code"})
metadata = metadata.rename(columns={"class_title": "title"})
metadata['asterisk_code'] = [[] for _ in range(len(metadata))]
metadata['code2'] = [[] for _ in range(len(metadata))]
metadata['title'] = metadata['title'].apply(lambda x: [x])

# combine and upload
output = pd.concat([output, metadata])
output = output.sort_values(by="code")
upload_to_mongodb(output)
