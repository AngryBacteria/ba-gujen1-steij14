import os
import json

annotation_path = "F:\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\clef ehealth\\anns_train_dev.txt"
text_files_folder_path = "F:\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\clef ehealth\\docs-training"

# Read the annotations. Split by id and codes
with open(annotation_path, "r") as f:
    annotations = [line.strip().split() for line in f]
    annotations = {ann[0]: ann[1].split("|") for ann in annotations}

data = {}
for filename in os.listdir(text_files_folder_path):
    if filename.endswith(".txt"):
        with open(os.path.join(text_files_folder_path, filename), "r") as f:
            content = f.read()

        # Parse annotation and id
        file_id = filename.split(".")[0]
        ann_codes = annotations.get(file_id, [])

        file_data = {
            "content": content,
            "annotations": {"icd10_block_codes": [], "icd10_chapter_codes": []},
        }
        for code in ann_codes:
            if "-" in code:
                file_data["annotations"]["icd10_block_codes"].append(code)
            else:
                file_data["annotations"]["icd10_chapter_codes"].append(code)

        if (
            len(file_data["annotations"]["icd10_block_codes"]) > 0
            or len(file_data["annotations"]["icd10_chapter_codes"]) > 0
        ):
            data[file_id] = file_data

json_data = json.dumps(data, indent=4)
print(json_data)
