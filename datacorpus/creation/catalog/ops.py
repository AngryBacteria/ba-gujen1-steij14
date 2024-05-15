# DATA SOURCE: https://www.bfarm.de/SharedDocs/Downloads/DE/Kodiersysteme/klassifikationen/ops/vorgaenger-bis-2020/
import pandas as pd

from shared.mongodb import upload_data_to_mongodb

OPS_CSV_PATH = "S:\\documents\\onedrive_bfh\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\catalog\\ops\\p1smt2017\\Klassifikationsdateien\\ops2017syst_kodes.txt"
OPS_DREISTELLER_CSV_PATH = "S:\\documents\\onedrive_bfh\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\catalog\\ops\\p1smt2017\\Klassifikationsdateien\\ops2017syst_dreisteller.txt"
OPS_GRUPPEN_CSV_PATH = "S:\\documents\\onedrive_bfh\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\catalog\\ops\\p1smt2017\\Klassifikationsdateien\\ops2017syst_gruppen.txt"
OPS_KAPITEL_CSV_PATH = "S:\\documents\\onedrive_bfh\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\catalog\\ops\\p1smt2017\\Klassifikationsdateien\\ops2017syst_kapitel.txt"
OPS_ALPHABET_CSV_PATH = "S:\\documents\\onedrive_bfh\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\catalog\\ops\\p2set2017\\ops2017alpha_edvtxt_20161028.txt"


def upload_ops_metadata(
    add_dreisteller=True, add_groups=True, add_chapters=True, add_alphabet=True
):
    df = pd.read_csv(OPS_CSV_PATH, sep=";", encoding="utf-8")
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
        if pd.isna(row["Titel"]) or pd.isna(row["Kode"]):
            continue
        output.append(
            {
                "title": row["Titel"],
                "code": row["Kode"],
                "chapter_code": row["KapNr"],
                "level": row["Ebene"],
                "synonyms": [],
            }
        )

    if add_dreisteller:
        df_dreisteller = pd.read_csv(
            OPS_DREISTELLER_CSV_PATH, sep=";", encoding="utf-8"
        )
        df_dreisteller.columns = ["KapNr", "GrVon", "Kode", "Titel"]
        for _, row in df_dreisteller.iterrows():
            if pd.isna(row["Titel"]) or pd.isna(row["Kode"]):
                continue
            output.append(
                {
                    "title": row["Titel"],
                    "code": row["Kode"],
                    "chapter_code": row["KapNr"],
                    "level": 3,
                    "synonyms": [],
                }
            )

    if add_groups:
        df_groups = pd.read_csv(OPS_GRUPPEN_CSV_PATH, sep=";", encoding="utf-8")
        df_groups.columns = ["KapNr", "GrVon", "GrBis", "Titel"]
        for _, row in df_groups.iterrows():
            if pd.isna(row["Titel"]) or pd.isna(row["GrVon"]) or pd.isna(row["GrBis"]):
                continue
            output.append(
                {
                    "title": row["Titel"],
                    "code": f"{row['GrVon']} - {row['GrBis']}",
                    "chapter_code": row["KapNr"],
                    "level": 2,
                    "synonyms": [],
                }
            )

    if add_chapters:
        df_chapters = pd.read_csv(OPS_KAPITEL_CSV_PATH, sep=";", encoding="utf-8")
        df_chapters.columns = ["KapNr", "Titel"]
        for _, row in df_chapters.iterrows():
            if pd.isna(row["Titel"]):
                continue
            output.append(
                {
                    "title": row["Titel"],
                    "code": row["KapNr"],
                    "chapter_code": row["KapNr"],
                    "level": 1,
                    "synonyms": [],
                }
            )

    if add_alphabet:
        df_alphabet = pd.read_csv(OPS_ALPHABET_CSV_PATH, sep="|", encoding="ANSI")
        df_alphabet.columns = ["type", "dimdiid", "code1", "code2", "text"]

        # find by code1 and add to synonyms array
        for _, row in df_alphabet.iterrows():
            code1 = row["code1"]
            text = row["text"]
            for item in output:
                if item["code"] == code1:
                    item["synonyms"].append(text)

    upload_data_to_mongodb(output, "catalog", "ops", True, ["code"])


upload_ops_metadata(True, False, False, True)
