# DATA SOURCE: https://www.bfarm.de/SharedDocs/Downloads/DE/Kodiersysteme/klassifikationen/ops/vorgaenger-bis-2020/
import pandas as pd

from shared.mongodb import upload_data_to_mongodb

OPS_CSV_PATH = "S:\\documents\\onedrive_bfh\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\catalog\\ops\\p1smt2017\\Klassifikationsdateien\\ops2017syst_kodes.txt"
OPS_ALPHABET_CSV_PATH = "S:\\documents\\onedrive_bfh\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\catalog\\ops\\p2set2017\\ops2017alpha_edvtxt_20161028.txt"


def upload_ops_metadata(add_alphabet=True):
    df = pd.read_csv(OPS_CSV_PATH, sep=";")
    df.columns = [
        "Ebene",
        "Ort",
        "Art",
        "KapNr",
        "GrVon",
        "DKode",
        "Kode",
        "Seite",
        "Titel",
        "Viersteller",
        "Fünfsteller",
        "Sechssteller",
        "Para17bd",
        "ZusatzK",
        "EinmalK",
    ]

    output = []
    for _, row in df.iterrows():
        if not pd.isna(row["Titel"]) and not pd.isna(row["Kode"]):
            output.append(
                {
                    "title": row["Titel"],
                    "code": row["Kode"],
                    "chapter_code": row["KapNr"],
                    "level": row["Ebene"],
                    "synonyms": [],
                }
            )

    if add_alphabet:
        df_alphabet = pd.read_csv(OPS_ALPHABET_CSV_PATH, sep="|", encoding="utf-8")
        df_alphabet.columns = ["type", "dimdiid", "code1", "code2", "text"]

        # find by code1 and add to synonyms array
        for _, row in df_alphabet.iterrows():
            code1 = row["code1"]
            text = row["text"]
            for item in output:
                if item["code"] == code1:
                    item["synonyms"].append(text)

    upload_data_to_mongodb(output, "catalog", "ops", True, ["code"])


upload_ops_metadata()
