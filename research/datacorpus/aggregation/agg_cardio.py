# TODO: aggregation of cardio

from research.datacorpus.aggregation.prompts import MEDICATION_PROMPT
from research.datacorpus.creation.utils.utils_mongodb import get_collection
from research.logger import logger

cardio = get_collection("corpus", "cardio")
cardio_heldout = get_collection("corpus", "cardio_heldout")


# TODO: rausnhemen? nein
# TODO: duration / frequency prompts?
def get_cardio_pretrain_texts():
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


def get_cardio_medication_prompts():
    """
    Retrieves medication prompts from the cardio corpus based on the MEDICATION_PROMPT.
    :param minimal_length: Minimal length of origin texts to include
    :return: List of medication extraction prompts
    """
    prompts = []
    documents = cardio.find({"annotations.type": "MEDICATION"})

    for document in documents:
        simple_prompt_str = MEDICATION_PROMPT.replace("<<CONTEXT>>", document["full_text"])
        texts = []
        for anno in document["annotations"]:
            # skip certain steps
            if anno["type"] != "MEDICATION":
                continue

            # Concatenate texts to form a single output string
            texts.extend(anno["text"])

        texts = list(set(texts))
        texts = "|".join(texts)
        simple_prompt_str = simple_prompt_str.replace("<<OUTPUT>>", texts)

        prompts.append(
            {
                "text": simple_prompt_str.strip(),
                "type": "MEDICATION",
                "task": "extraction",
                "source": "cardio",
            }
        )

    logger.debug(
        f"Created {len(prompts)} medication prompts from the cardio corpus."
    )

    return prompts


def aggregate_cardio_prompts():
    """
    Aggregate all medication-related prompts from the cardio corpus with specified minimal text length.
    :param minimal_length: Minimal length of origin texts to include
    :return: List of medication prompts
    """
    medication_prompts_cardio = get_cardio_medication_prompts()
    logger.info(
        f"Aggregated {len(medication_prompts_cardio)} medication prompts from the cardio corpus."
    )
    return medication_prompts_cardio


# Example: Aggregate medication prompts with a minimal origin text length of 50 characters
medication_prompts = aggregate_cardio_prompts()
for prompt in medication_prompts:
    print(prompt)
