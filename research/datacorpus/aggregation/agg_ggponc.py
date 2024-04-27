from research.datacorpus.aggregation.prompts import (
    MEDICATION_PROMPT,
    TREATMENT_PROMPT,
    DIAGNOSIS_PROMPT,
)
from research.datacorpus.creation.utils.utils_mongodb import get_collection
from research.logger import logger

ggonc_collection = get_collection("corpus", "ggponc_short")


# TODO: add prompts for sentences with not all 3 types of annotations?


def get_ggponc_prompts(
    annotation_type: str, extraction_prompt: str, minimal_length: int
):
    """
    Generic function to get prompts from ggponc corpus
    :param annotation_type: The type of annotation to get prompts for
    :param extraction_prompt: The prompt format for the extraction
    :param minimal_length: The minimal length of origin texts to include
    :return: List of prompts
    """
    prompts = []
    documents = ggonc_collection.find({"annotations.type": annotation_type})
    for document in documents:
        for anno in document["annotations"]:
            if anno["type"] != annotation_type:
                continue
            if minimal_length > 0 and len(anno["origin"]) < minimal_length:
                continue

            # remove duplicates from text
            anno["text"] = list(set(anno["text"]))

            # concatenate
            texts = "|".join(anno["text"])
            if texts == "":
                texts = "Keine vorhanden"

            simple_prompt_str = extraction_prompt.replace("<<CONTEXT>>", anno["origin"])
            simple_prompt_str = simple_prompt_str.replace("<<OUTPUT>>", texts)
            prompts.append(
                {
                    "text": simple_prompt_str.strip(),
                    "type": annotation_type,
                    "task": "extraction",
                    "source": "ggponc",
                }
            )

    return prompts


def get_all_ggponc_prompts(minimal_length: int):
    """
    Get all prompts from ggponc corpus
    :param minimal_length: The minimal length of origin texts to include
    :return: List of prompts
    """
    prompts = []
    # prompts with annotations
    medication_prompts = get_ggponc_prompts(
        "MEDICATION", MEDICATION_PROMPT, minimal_length
    )
    # diagnosis_prompts = get_ggponc_prompts(
    #     "DIAGNOSIS", DIAGNOSIS_PROMPT, minimal_length
    # )
    # treatment_prompts = get_ggponc_prompts(
    #     "TREATMENT", TREATMENT_PROMPT, minimal_length
    # )
    prompts.extend(medication_prompts)
    # prompts.extend(diagnosis_prompts)
    # prompts.extend(treatment_prompts)

    # prompts without annotations
    empty_medication_prompts = get_ggponc_prompts(
        "NA", MEDICATION_PROMPT, minimal_length
    )
    # empty_diagnosis_prompts = get_ggponc_prompts("NA", DIAGNOSIS_PROMPT, minimal_length)
    # empty_treatment_prompts = get_ggponc_prompts("NA", TREATMENT_PROMPT, minimal_length)
    prompts.extend(empty_medication_prompts)
    # prompts.extend(empty_diagnosis_prompts)
    # prompts.extend(empty_treatment_prompts)

    logger.debug(
        f"Created {len(prompts)} prompts from the ggponc corpus [minimal length: {minimal_length}]."
    )

    return prompts
