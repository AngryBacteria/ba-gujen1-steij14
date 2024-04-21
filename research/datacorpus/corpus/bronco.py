import os.path

import pandas as pd

from research.datacorpus.utils.utils_mongodb import upload_data_to_mongodb
from research.logger import logger

# DATA SOURCE: https://www2.informatik.hu-berlin.de/~leser/bronco/index.html

path_text = "Bachelorarbeit\\datensets\\corpus\\bronco\\textFiles"
path_annotation_brat = "Bachelorarbeit\\datensets\\corpus\\bronco\\bratFiles"
path_annotation_conll = "Bachelorarbeit\\datensets\\corpus\\bronco\\conllIOBTags"


# TODO: unify anonymization (PATIENT, etc...) and labels / types
# TODO: duplication issue Aldactone 50 mg
def find_sentence_by_word_position(content: str, start_pos: int, end_pos: int) -> str:
    """
    Find the sentence in which the given positions are located.
    Used for finding the sentence of an annotation
    :param start_pos: start position
    :param end_pos: end position
    :param content: the text to search
    :return: the string of the sentence
    """
    # Find the start of the sentence. Finds the last newline before start_pos
    content = content.strip()
    start_of_sentence = content.rfind("\n", 0, start_pos)
    if (
        start_of_sentence == -1
    ):  # If no newline is found, start from the beginning (first sentence)
        start_of_sentence = 0
    else:
        start_of_sentence += 1  # Move past the newline character

    # Find the end of the sentence
    end_of_sentence = content.find("\n", end_pos)
    if (
        end_of_sentence == -1
    ):  # If no newline is found, go to the end of the file (last sentence)
        end_of_sentence = len(content)

    # Extract and return the sentence
    sentence = content[start_of_sentence:end_of_sentence]
    return sentence.strip()


# read the text data
def read_text_file(file_number: int) -> str:
    """
    Read the .txt file that containing raw sentences
    :param file_number: number of the file to read
    :return: the text of the file as string
    """
    with open(os.path.join(path_text, f"randomSentSet{file_number}.txt"), "r") as file:
        annotation_text = file.read()
        return annotation_text.strip()


# parse the annotation data. The id starts either with N, T OR A
def parse_annotation_data_general(file_number: int) -> list[dict]:
    """
    Parse the annotation data from the .ann file
    :param file_number: number of the file to read
    :return: list of dictionaries with the annotation data
    """
    with open(
        os.path.join(path_annotation_brat, f"randomSentSet{file_number}.ann"), "r"
    ) as file:
        annotations_text = file.read()  # all the annotations
        annotation_entries = annotations_text.strip().split(
            "\n"
        )  # individual annotations
        sentences_text = read_text_file(file_number)  # the sentences of the annotations

        output = []
        for entry in annotation_entries:
            parts = [part.strip() for part in entry.split()]
            entry_id = parts[0]
            # Entity labels start with T
            try:
                if entry_id.startswith("T"):
                    # if the entity is split into multiple words, the end position is in a different part
                    modifier = 0
                    for part in parts[3:]:
                        if ";" in part:
                            modifier += 1
                        else:
                            break

                    type_string = parts[1]
                    start = int(parts[2])
                    end = int(parts[3 + modifier])
                    text_part = " ".join(parts[4 + modifier :])
                    origin = find_sentence_by_word_position(sentences_text, start, end)
                    output.append(
                        {
                            "id": f"{file_number}_" + entry_id,
                            "type": type_string,
                            "text": text_part,
                            "origin": origin,
                            "normalizations": [],
                            "attributes": [],
                            "start": start,
                            "end": end,
                        }
                    )

                # Attributes starts with A
                # LevelOfTruth (negative, speculative, possibleFuture)
                # or Localisation (R,L,B) , note Localisation==Laterality
                elif entry_id.startswith("A"):
                    attribute_label = parts[1]
                    attribute_reference = parts[2]
                    attribute = parts[3]

                    found = False
                    for entry_dict in output:
                        if entry_dict["id"] == (
                            f"{file_number}_" + attribute_reference
                        ):
                            entry_dict["attributes"].append(
                                {
                                    "attribute_label": attribute_label,
                                    "attribute": attribute,
                                }
                            )
                            found = True

                    if not found:
                        logger.warning(
                            f"Attribute reference {file_number}_{attribute_reference} not found"
                        )

                # Normalization starts with N
                # DIAGNOSIS (ICD10GM2017)
                # TREATMENT (OPS2017)
                # MEDICATION (ATC2017)
                elif entry_id.startswith("N"):
                    normalization_reference = parts[2]
                    normalization = parts[3]
                    normalization_text = " ".join(parts[4:])

                    found = False
                    for entry_dict in output:
                        if entry_dict["id"] == (
                            f"{file_number}_" + normalization_reference
                        ):
                            entry_dict["normalizations"].append(
                                {
                                    "normalization": normalization,
                                    "normalization_text": normalization_text,
                                }
                            )
                            found = True

                    if not found:
                        logger.warning(
                            f"Normalization reference {file_number}_{normalization_reference} not found"
                        )

            except Exception as e:
                logger.warning(f"Outer Error while parsing {entry} - {e}")

        logger.debug(
            f"Parsed {len(output)} entries from file randomSentSet{file_number}.ann"
        )

    # group the data by type and origin. Every sentence can have multiple annotations of the same type
    df = pd.DataFrame(output)
    grouped_df = (
        df.groupby(["type", "origin"])
        .agg(
            {
                "id": lambda x: x.tolist(),
                "text": lambda x: x.tolist(),
                "normalizations": lambda x: x.tolist(),
                "attributes": lambda x: x.tolist(),
                "start": lambda x: x.tolist(),
                "end": lambda x: x.tolist(),
            }
        )
        .reset_index()
    )

    # get sentences without annotations
    sentences = sentences_text.split("\n")
    for sentence in sentences:
        if sentence.strip() not in grouped_df["origin"].tolist():
            new_rows = [
                {
                    "type": "None",
                    "origin": sentence.strip(),
                    "id": [],
                    "text": [],
                    "normalizations": [],
                    "attributes": [],
                    "start": [],
                    "end": [],
                }
            ]
            new_rows = pd.DataFrame(new_rows)
            grouped_df = pd.concat([grouped_df, new_rows], ignore_index=True)

    return grouped_df.to_dict(orient="records")


def parse_annotation_data_ner(file_number: int) -> list[dict[str, list]]:
    """
    Parse the annotation data in the CoNLL format with the NER tags.
    Useful for BERT and other NER models
    :return: List of CoNLL formatted data
    """
    with open(
        os.path.join(path_annotation_conll, f"randomSentSet{file_number}.CONLL"), "r"
    ) as file:
        text = file.read()
    sentences = []
    # split the text into the data chunks and init empty lists
    lines = text.strip().split("\n")
    words = []
    word_types = []
    ner_tags = []
    for line in lines:
        if line.strip():  # This skips empty lines, they are used to separate sentences
            word, word_type, ner_tag = line.strip().split()
            words.append(word)
            word_types.append(word_type)
            ner_tags.append(ner_tag)
        else:  # End of sentence is reached
            if words:
                sentences.append(
                    {"words": words, "word_types": word_types, "ner_tags": ner_tags}
                )
                words = []
                word_types = []
                ner_tags = []

    logger.debug(
        f"Parsed {len(sentences)} sentences from file randomSentSet{file_number}.CONLL"
    )
    return sentences


def create_bronco_db() -> None:
    """
    Create the bronco database in MongoDB
    """
    # JSON bronco collection
    data = []
    for i in range(1, 6):
        data.extend(parse_annotation_data_general(i))
    logger.debug(f"Parsed {len(data)} entries in total")
    upload_data_to_mongodb(data, "corpus", "bronco", True, [])

    # NER bronco collection
    data_ner = []
    for i in range(1, 6):
        data_ner.extend(parse_annotation_data_ner(i))
    logger.debug(f"Parsed {len(data_ner)} NER entries in total")
    upload_data_to_mongodb(data_ner, "corpus", "bronco_ner", True, [])


create_bronco_db()
