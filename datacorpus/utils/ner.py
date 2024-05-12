from shared.logger import logger


def load_conll_annotations_file(filepath: str) -> str:
    """Load a CoNLL file from disk and return the text as a string."""
    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read()
    return text


def parse_ner_dataset(filepath: str):
    """
    Parse a CoNLL NER dataset from a file and return a list of sentences with words and NER tags.
    :param filepath: The path to the CoNLL NER dataset file.
    """
    text = load_conll_annotations_file(filepath)

    sentences = []
    # split the text into the data chunks and init empty lists
    lines = text.strip().split("\n")
    words = []
    ner_tags = []
    for line in lines:
        if line.strip():  # No end of sentence reached (because line is not empty)
            word, ner_tag = line.split()
            words.append(word)
            ner_tags.append(ner_tag)
        else:  # End of sentence is reached
            if words:
                sentences.append({"words": words, "ner_tags": ner_tags})
                words = []
                ner_tags = []

    logger.debug(f"Parsed {len(sentences)} NER annotations from file {filepath}")
    return sentences


def group_ner_data(ner_docs: list[dict], block_size: int, source: str):
    # Create blocks to reduce the number of datapoints
    grouped_docs = []
    temp_doc = {"words": [], "ner_tags": [], "source": source}
    for doc in ner_docs:
        if len(temp_doc["words"]) + len(doc["words"]) > block_size:
            grouped_docs.append(temp_doc)
            temp_doc = {"words": [], "ner_tags": [], "source": "cardio"}
        temp_doc["words"].extend(doc["words"])
        temp_doc["ner_tags"].extend(doc["ner_tags"])

    if temp_doc["words"]:  # Add the last group if not empty
        grouped_docs.append(temp_doc)

    return grouped_docs
