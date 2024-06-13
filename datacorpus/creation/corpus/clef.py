import os

from shared.mongodb import upload_data_to_mongodb
from shared.logger import logger

# DATA SOURCE: https://clefehealth.imag.fr/?page_id=215

# path of the file containing the id and annotations
annotation_path = "F:clef ehealth\\anns_train_dev.txt"
# path of the folder containing the text files
text_files_folder_path = "F:clef ehealth\\docs-training"

# Read the annotations. Split by id and codes
with open(annotation_path, "r") as f:
    # split document id and codes
    annotations = [line.strip().split() for line in f]
    # create a dictionary with the document id as key and the codes as value
    annotations = {ann[0]: ann[1].split("|") for ann in annotations}

data = []
for filename in os.listdir(text_files_folder_path):
    if filename.endswith(".txt"):
        with open(
            os.path.join(text_files_folder_path, filename), "r", encoding="utf-8"
        ) as f:
            content = f.read()

        # Parse annotation and document id
        file_id = filename.split(".")[0]
        ann_codes = annotations.get(file_id, [])

        file_data = {
            "text": content.strip(),
            "icd10_block_codes": [],
            "icd10_chapter_codes": [],
            "document": f"{file_id}.txt",
        }
        for code in ann_codes:
            if "-" in code:
                file_data["icd10_block_codes"].append(code)
            else:
                file_data["icd10_chapter_codes"].append(code)

        if (
            len(file_data["icd10_chapter_codes"]) > 0
            or len(file_data["icd10_chapter_codes"]) > 0
        ):
            data.append(file_data)

        logger.debug(f"Parsed {file_id} from CLEF eHealth corpus.")
    logger.debug(f"Parsed {len(data)} documents from CLEF eHealth corpus.")


if __name__ == "__main__":
    upload_data_to_mongodb(data, "corpus", "clef2019", True, ["document"])
