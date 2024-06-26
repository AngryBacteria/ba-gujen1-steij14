from datacorpus.utils.ner import group_ner_data
from shared.logger import logger
from shared.mongodb import get_collection
from shared.prompt_utils import (
    get_extraction_messages,
    AttributeFormat,
    EntityType,
    TaskType,
)


# Aggregation code for the cardio corpus. This code is used to aggregate prompts and NER annotations from the cardio
# corpus. Only the MEDICATION annotation_type is included in the cardio dataset. Additional attributes can be added to
# the medication prompts, specifically STRENGTH and FREQUENCY.


def get_cardio_medication_prompts():
    """
    Creates the medication prompts from the cardio corpus.
    :return: List of medication extraction prompts
    """
    prompts = []
    cardio = get_collection("corpus", "cardio")
    documents = cardio.find({"annotations.type": "MEDICATION"})

    for document in documents:
        texts = []
        for anno in document["annotations"]:
            # skip all with no Medication
            if anno["type"] != "MEDICATION":
                continue

            # are already in same order (first attributes belong to first text)
            names = anno["text"]
            attributes = anno["attributes"]

            for name, attr_list in zip(names, attributes):
                # Preparing attributes string for each text, skipping 'DURATION' and 'FORM' attributes (because
                # of anonymization they are often nonsense)
                filtered_attributes = [
                    attr
                    for attr in attr_list
                    if (
                        attr["attribute_label"] == "STRENGTH"
                        or attr["attribute_label"] == "FREQUENCY"
                    )
                ]
                if filtered_attributes:
                    attributes_str = "|".join(
                        [
                            f"{attr['attribute_label'].replace('STRENGTH', 'DOSIERUNG').replace('FREQUENCY', 'FREQUENZ')}: {attr['attribute']}"
                            for attr in filtered_attributes
                        ]
                    )
                else:
                    attributes_str = ""

                    # combining text and attributes into one string
                med_str = f"{name} [{attributes_str}]"
                texts.append(med_str)

        texts = list(set(texts))
        texts = [text.strip() for text in texts]
        extraction_string = "|".join(texts)

        messages = get_extraction_messages(
            document["full_text"], AttributeFormat.CARDIO, EntityType.MEDICATION
        )
        messages.append(
            {
                "role": "assistant",
                "content": extraction_string.strip(),
            }
        )

        prompts.append(
            {
                "messages": messages,
                "type": EntityType.MEDICATION.value,
                "task": TaskType.EXTRACTION.value,
                "source": "cardio",
                "na_prompt": False,
                "context": document["full_text"],
                "context_entity": "",
                "output": extraction_string.strip(),
            }
        )

    return prompts


def aggregate_cardio_ner(block_size: int):
    """
    Aggregate all NER annotations from the cardio corpus into a format usable for training.
    :param block_size: The size of the blocks to create. -1 means no block grouping. A block means a group of examples
    that are concatenated into one example to reduce the number of total examples.
    Aggregate all NER annotations from the cardio corpus into a format usable for training.
    Filters out all NER tags that are not MEDICATION, TREATMENT, or DIAGNOSIS and replaces them with the respective id.
    :return: List of NER annotations as dictionaries
    """
    ner_docs = []
    cardio_ner = get_collection("corpus", "cardio_ner")
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
    if block_size > 1:
        ner_docs = group_ner_data(ner_docs, block_size, "cardio")

    logger.debug(
        f"Created {len(ner_docs)} ner datapoints from the cardio corpus [block_size={block_size})"
    )
    return ner_docs


def aggregate_cardio_prompts():
    """
    Aggregate all medication-related prompts from the cardio corpus with specified minimal text length.
    :return: List of medication prompts
    """
    medication_prompts_cardio = get_cardio_medication_prompts()
    logger.debug(
        f"Aggregated {len(medication_prompts_cardio)} medication prompts from the cardio corpus."
    )
    return medication_prompts_cardio
