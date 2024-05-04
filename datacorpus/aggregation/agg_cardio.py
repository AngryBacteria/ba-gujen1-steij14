from datacorpus.aggregation.prompts import (
    MEDICATION_INSTRUCTION,
    MEDICATION_INSTRUCTION_ATTRIBUTES,
    SYSTEM_PROMPT,
)
from datacorpus.utils.mongodb import get_collection
from shared.logger import logger

cardio = get_collection("corpus", "cardio")
cardio_heldout = get_collection("corpus", "cardio_heldout")
cardio_ner = get_collection("corpus", "cardio_ner")

def get_cardio_instruction(add_attributes: bool):
    """Helper function for getting the right instruction strings.
    :param add_attributes: Boolean to if the attributes should be added to the instruction string
    """
    if add_attributes:
        extraction_instruction = MEDICATION_INSTRUCTION_ATTRIBUTES
    else:
        extraction_instruction = MEDICATION_INSTRUCTION

    return extraction_instruction

# TODO: duration / frequency prompts? --> Done but now perhaps same meds?
def get_cardio_medication_prompts(
        add_attributes: bool
):
    """
    Retrieves medication prompts from the cardio corpus based on the MEDICATION_PROMPT.
    :param add_attributes: If the attributes of medication should be added to the text
    :return: List of medication extraction prompts
    """
    prompts = []
    documents = cardio.find({"annotations.type": "MEDICATION"})
    extraction_instruction = get_cardio_instruction(add_attributes)

    for document in documents:
        extraction_instruction_str = extraction_instruction.replace(
            "<<CONTEXT>>", document["full_text"]
        )
        texts = []
        for anno in document["annotations"]:
            # skip all with no Medication
            if anno["type"] != "MEDICATION":
                continue

            if add_attributes:
                # are already in same order (first attributes belong to first text)
                names = anno["text"]
                attributes = anno["attributes"]

                for name, attr_list in zip(names, attributes):
                    # Preparing attributes string for each text, skipping 'DURATION' attributes (because of anonymization they are often nonsense)
                    filtered_attributes = [attr for attr in attr_list if attr['attribute_label'] != 'DURATION']
                    if filtered_attributes:
                        attributes_str = ", ".join([f"{attr['attribute_label'].replace('STRENGTH', 'STÄRKE').replace('FREQUENCY', 'HÄUFIGKEIT')}: {attr['attribute']}" for attr in filtered_attributes])
                    else:
                        attributes_str = "Keine Attribute vorhanden" 

                    # combining text and attributes into one string
                    med_str = f"{name} [{attributes_str}]"
                    texts.append(med_str)
            else:
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


def aggregate_cardio_prompts(
        attributes: bool
):
    """
    Aggregate all medication-related prompts from the cardio corpus with specified minimal text length.
    :param attributes: If prompt with attributes should be included 
    :return: List of medication prompts
    """
    medication_prompts_cardio = get_cardio_medication_prompts(attributes)
    logger.info(
        f"Aggregated {len(medication_prompts_cardio)} medication prompts from the cardio corpus."
    )
    return medication_prompts_cardio


print(get_cardio_medication_prompts(True)[10])