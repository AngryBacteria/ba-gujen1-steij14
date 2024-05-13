from datacorpus.aggregation.prompts import (
    SYSTEM_PROMPT,
    MEDICATION_INSTRUCTION,
    DIAGNOSIS_INSTRUCTION,
    TREATMENT_INSTRUCTION,
)
from datacorpus.utils.ner import group_ner_data
from shared.mongodb import get_collection
from shared.logger import logger

# Aggregation of the data in the ggponc datacorpus. Includes all three types of annotations: MEDICATION, TREATMENT,
# DIAGNOSIS. Data is saved as prompts.


def get_ggponc_instruction(annotation_type: str):
    """
    Helper function to get the instruction string for a given annotation type.
    :param annotation_type: Type of annotation to get the instruction string for. Can be
    one of the following: DIAGNOSIS, TREATMENT, MEDICATION
    """
    if annotation_type == "DIAGNOSIS":
        extraction_instruction = DIAGNOSIS_INSTRUCTION
    elif annotation_type == "TREATMENT":
        extraction_instruction = TREATMENT_INSTRUCTION
    elif annotation_type == "MEDICATION":
        extraction_instruction = MEDICATION_INSTRUCTION
    else:
        raise ValueError(
            f"Annotation type {annotation_type} is not supported for ggponc prompts."
        )

    return extraction_instruction


# TODO: add prompts with no annotation for the entity, but that have annotations for other entities
def get_ggponc_prompts(annotation_type: str, na_prompts: bool, minimal_length: int):
    """
    Generic function to get prompts from the ggponc corpus
    :param annotation_type: The type of annotation to get prompts for. Can be
    one of the following: DIAGNOSIS, TREATMENT, MEDICATION
    :param minimal_length: The minimal length of origin texts to include
    :param na_prompts: If the prompts should be created for sentences without annotations
    :return: List of prompts
    """
    extraction_instruction = get_ggponc_instruction(annotation_type)
    # if na_prompt is enabled use "na" as annotation type to get the right examples from the db
    annotation_type_output = annotation_type
    if na_prompts:
        annotation_type = "NA"

    prompts = []
    ggponc_collection = get_collection("corpus", "ggponc_short")
    documents = ggponc_collection.find({"annotations.type": annotation_type})
    for document in documents:
        for anno in document["annotations"]:
            if anno["type"] != annotation_type:
                continue
            if minimal_length > 0 and len(anno["origin"]) < minimal_length:
                continue

            # remove duplicates from text
            texts = list(set(anno["text"]))

            # concatenate
            extraction_string = "|".join(texts)
            if extraction_string == "":
                extraction_string = "Keine vorhanden"

            extraction_prompt_str = extraction_instruction.replace(
                "<<CONTEXT>>", anno["origin"]
            )
            extraction_prompt_str = extraction_prompt_str.replace(
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
                            "content": extraction_prompt_str.strip(),
                        },
                        {
                            "role": "assistant",
                            "content": extraction_string.strip(),
                        },
                    ],
                    "type": annotation_type_output,
                    "task": "extraction",
                    "source": "ggponc",
                    "na_prompt": na_prompts,
                    "annotation_labels": extraction_string,
                }
            )

    return prompts


def aggregate_ggponc_ner(block_size: int):
    """
    :param block_size: The size of the blocks to create. -1 means no block grouping
    Get all NER documents from the ggponc corpus.
    Filter out all NER tags that are not MEDICATION, TREATMENT, or DIAGNOSIS.
    :return: List of NER annotations as dictionaries
    """
    ner_docs = []
    ggponc_collection_ner = get_collection("corpus", "ggponc_short_ner")
    documents = ggponc_collection_ner.find({})
    for document in documents:
        ner_tags = []
        for tag in document["ner_tags"]:
            if tag == "B-MED":
                ner_tags.append(1)
            if tag == "I-MED":
                ner_tags.append(2)
            if tag == "B-TREAT":
                ner_tags.append(3)
            if tag == "I-TREAT":
                ner_tags.append(4)
            if tag == "B-DIAG":
                ner_tags.append(5)
            if tag == "I-DIAG":
                ner_tags.append(6)
            else:
                ner_tags.append(0)

        ner_docs.append(
            {"words": document["words"], "ner_tags": ner_tags, "source": "ggponc"}
        )

    if block_size > 1:
        ner_docs = group_ner_data(ner_docs, block_size, "ggpponc")

    logger.debug(
        f"Created {len(ner_docs)} ner datapoints from the ggponc corpus [block_size={block_size})"
    )
    return ner_docs


def aggregate_ggponc_prompts(
    minimal_length: int,
    diagnosis: bool,
    treatment: bool,
    medication: bool,
    na_prompts: bool,
):
    """
    Get all prompts from ggponc corpus
    :param na_prompts: If prompts without any annotations should be included
    :param medication: If medication prompts should be included
    :param treatment: If treatment prompts should be included
    :param diagnosis: If diagnosis prompts should be included
    :param minimal_length: The minimal length of origin texts to include
    :return: List of prompts
    """
    prompts = []
    # prompts with annotations
    if diagnosis:
        diagnosis_prompts = get_ggponc_prompts("DIAGNOSIS", False, minimal_length)
        prompts.extend(diagnosis_prompts)
    if treatment:
        treatment_prompts = get_ggponc_prompts("TREATMENT", False, minimal_length)
        prompts.extend(treatment_prompts)
    if medication:
        medication_prompts = get_ggponc_prompts("MEDICATION", False, minimal_length)
        prompts.extend(medication_prompts)

    # prompts without annotations
    if diagnosis and na_prompts:
        empty_diagnosis_prompts = get_ggponc_prompts("DIAGNOSIS", True, minimal_length)
        prompts.extend(empty_diagnosis_prompts)
    if treatment and na_prompts:
        empty_treatment_prompts = get_ggponc_prompts("TREATMENT", True, minimal_length)
        prompts.extend(empty_treatment_prompts)
    if medication and na_prompts:
        empty_medication_prompts = get_ggponc_prompts(
            "MEDICATION", True, minimal_length
        )
        prompts.extend(empty_medication_prompts)

    logger.debug(
        f"Created {len(prompts)} prompts from the ggponc corpus [minimal length: {minimal_length}]."
    )

    return prompts
