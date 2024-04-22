from research.datacorpus.aggregation.prompts import (
    MEDICATION_PROMPT,
    TREATMENT_PROMPT,
    DIAGNOSIS_PROMPT,
)
from research.datacorpus.creation.utils.utils_mongodb import get_collection

ggonc_collection = get_collection("corpus", "ggponc_short")


def get_ggponc_prompts(
    annotation_type: str, extraction_prompt: str, minimal_length: int
) -> list[str]:
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

            texts = "|".join(anno["text"])
            if texts == "":
                texts = "Keine vorhanden"

            simple_prompt_str = extraction_prompt.replace("<<CONTEXT>>", anno["origin"])
            simple_prompt_str = simple_prompt_str.replace("<<OUTPUT>>", texts)
            prompts.append(simple_prompt_str.strip())

    return prompts


def get_all_ggponc_prompts(minimal_length: int) -> list[str]:
    """
    Get all prompts from ggponc corpus
    :param minimal_length: The minimal length of origin texts to include
    :return: List of prompts
    """
    prompts = set()
    # prompts with annotations
    medication_prompts = get_ggponc_prompts(
        "MEDICATION", MEDICATION_PROMPT, minimal_length
    )
    diagnosis_prompts = get_ggponc_prompts(
        "DIAGNOSIS", DIAGNOSIS_PROMPT, minimal_length
    )
    treatment_prompts = get_ggponc_prompts(
        "TREATMENT", TREATMENT_PROMPT, minimal_length
    )
    prompts.update(medication_prompts)
    prompts.update(diagnosis_prompts)
    prompts.update(treatment_prompts)

    # prompts without annotations
    empty_medication_prompts = get_ggponc_prompts(
        "NA", MEDICATION_PROMPT, minimal_length
    )
    empty_diagnosis_prompts = get_ggponc_prompts("NA", DIAGNOSIS_PROMPT, minimal_length)
    empty_treatment_prompts = get_ggponc_prompts("NA", TREATMENT_PROMPT, minimal_length)
    prompts.update(empty_medication_prompts)
    prompts.update(empty_diagnosis_prompts)
    prompts.update(empty_treatment_prompts)

    return list(prompts)
