import os

from research.datacorpus.utils_mongodb import upload_data_to_mongodb

# path of the file containing the id and annotations
annotation_path = "F:\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\clef ehealth\\anns_train_dev.txt"
# path of the folder containing the text files
text_files_folder_path = "F:\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\clef ehealth\\docs-training"

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
            "text": content,
            "icd10_block_codes": [],
            "icd10_chapter_codes": [],
            "document_id": file_id,
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


# Upload to MongoDB
upload_data_to_mongodb(data, "corpus", "clef2019", True, ["document_id"])
