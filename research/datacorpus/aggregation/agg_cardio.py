from research.datacorpus.aggregation.prompts import (
    MEDICATION_INSTRUCTION,
    SYSTEM_PROMPT,
)
from research.datacorpus.creation.utils.utils_mongodb import get_collection
from research.logger import logger

cardio = get_collection("corpus", "cardio")
cardio_heldout = get_collection("corpus", "cardio_heldout")
cardio_ner = get_collection("corpus", "cardio_ner")


# TODO: duration / frequency prompts?
def get_cardio_medication_prompts():
    """
    Retrieves medication prompts from the cardio corpus based on the MEDICATION_PROMPT.
    :return: List of medication extraction prompts
    """
    prompts = []
    documents = cardio.find({"annotations.type": "MEDICATION"})

    for document in documents:
        extraction_instruction_str = MEDICATION_INSTRUCTION.replace(
            "<<CONTEXT>>", document["full_text"]
        )
        texts = []
        for anno in document["annotations"]:
            # skip certain steps
            if anno["type"] != "MEDICATION":
                continue

            # Concatenate texts to form a single output string
            texts.extend(anno["text"])

        texts = list(set(texts))
        extraction_string = "|".join(texts)
        extraction_instruction_str = extraction_instruction_str.replace(
            "<<OUTPUT>>", extraction_string
        )

        prompts.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": extraction_instruction_str.strip(),
                    },
                    {
                        "role": "assistant",
                        "content": extraction_string.strip(),
                    },
                ],
                "text": extraction_instruction_str.strip(),
                "type": "MEDICATION",
                "task": "extraction",
                "source": "cardio",
                "annotation_labels": (
                    extraction_string if extraction_string != "Keine vorhanden" else ""
                ),
            }
        )

    logger.debug(f"Created {len(prompts)} medication prompts from the cardio corpus.")

    return prompts


def aggregate_cardio_ner():
    """
    Aggregate all NER annotations from the cardio corpus.
    Filter out all NER tags that are not MEDICATION, TREATMENT, or DIAGNOSIS.
    :return: List of NER annotations as dictionaries
    """
    ner_docs = []
    documents = cardio_ner.find({})
    for document in documents:
        for anno in document["annotations"]:
            ner_tags = []
            for tag in anno["ner_tags"]:
                if tag == "B-MED" or tag == "B-ACTIVEING":
                    ner_tags.append(1)
                if tag == "I-MED" or tag == "I-ACTIVEING":
                    ner_tags.append(2)
                else:
                    ner_tags.append(0)

            ner_docs.append(
                {"words": anno["words"], "ner_tags": ner_tags, "source": "cardio"}
            )
    logger.debug(f"Created {len(ner_docs)} ner datapoints from the cardio corpus.")
    return ner_docs


def aggregate_cardio_pretrain_texts():
    """
    Get all cardio pretrain texts
    :return: List of pretrain texts from cardio and cardio_heldout
    """
    cardio_texts = [
        {"text": doc["full_text"], "task": "pretrain", "source": "cardio"}
        for doc in cardio.find({}, {"full_text": 1})
    ]
    cardio_heldout_texts = [
        {"text": doc["full_text"], "task": "pretrain", "source": "cardio"}
        for doc in cardio_heldout.find({}, {"full_text": 1})
    ]
    cardio_texts = cardio_texts + cardio_heldout_texts
    return cardio_texts


def aggregate_cardio_prompts():
    """
    Aggregate all medication-related prompts from the cardio corpus with specified minimal text length.
    :return: List of medication prompts
    """
    medication_prompts_cardio = get_cardio_medication_prompts()
    logger.info(
        f"Aggregated {len(medication_prompts_cardio)} medication prompts from the cardio corpus."
    )
    return medication_prompts_cardio
