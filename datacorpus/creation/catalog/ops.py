# DATA SOURCE: https://www.bfarm.de/SharedDocs/Downloads/DE/Kodiersysteme/klassifikationen/ops/vorgaenger-bis-2020/
import pandas as pd

from shared.mongodb import upload_data_to_mongodb

# Paths to the various OPS classification files. Includes metadata and alphabet
OPS_CSV_PATH = "S:\\documents\\onedrive_bfh\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\catalog\\ops\\ops2024syst_kodes.txt"
OPS_DREISTELLER_CSV_PATH = "S:\\documents\\onedrive_bfh\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\catalog\\ops\\ops2024syst_dreisteller.txt"
OPS_GROUPS_CSV_PATH = "S:\\documents\\onedrive_bfh\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\catalog\\ops\\ops2024syst_gruppen.txt"
OPS_CHAPTERS_CSV_PATH = "S:\\documents\\onedrive_bfh\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\catalog\\ops\\ops2024syst_kapitel.txt"
OPS_ALPHABET_CSV_PATH = "S:\\documents\\onedrive_bfh\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\catalog\\ops\\ops2024alpha_edvtxt_20231103.txt"


def upload_ops_metadata(
    add_dreisteller=True, add_groups=False, add_chapters=False, add_alphabet=True
):
    """
    Parse and upload the OPS metadata and optionally alphabet data (for synonyms) to the MongoDB database.
    :param add_dreisteller: If the dreisteller ops data should be added (recommended)
    :param add_groups: If the groups ops data should be added (not recommended)
    :param add_chapters: If the chapters ops data should be added (not recommended)
    :param add_alphabet: If the alphabet ops data should be added (recommended)
    :return: Nothing, data is uploaded
    """
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
        df_groups = pd.read_csv(OPS_GROUPS_CSV_PATH, sep=";", encoding="utf-8")
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
        df_chapters = pd.read_csv(OPS_CHAPTERS_CSV_PATH, sep=";", encoding="utf-8")
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


if __name__ == "__main__":
    upload_ops_metadata(True, False, False, True)
