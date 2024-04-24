from research.datacorpus.aggregation.prompts import (
    MEDICATION_PROMPT,
    DIAGNOSIS_PROMPT,
    TREATMENT_PROMPT,
    MEDICATION_NORMALIZATION_PROMPT,
    DIAGNOSIS_NORMALIZATION_PROMPT,
    TREATMENT_NORMALIZATION_PROMPT,
)
from research.datacorpus.creation.utils.utils_mongodb import get_collection
from research.logger import logger

bronco_collection = get_collection("corpus", "bronco")


# TODO: add attribute prompts
# TODO: add prompts for sentences with not all 3 types of annotations?


def get_bronco_prompts(
    annotation_type: str,
    extraction: str,
    normalization_prompt: str,
    minimal_length: int,
) -> tuple[list[str], list[str]]:
    """
    Generic function to get prompts from bronco corpus
    :param annotation_type: The type of annotation to get prompts for
    :param extraction: The prompt format for the extraction
    :param normalization_prompt: The prompt format for the normalization task
    :param minimal_length: Minimal length of origin texts to include
    :return: List of prompts
    """

    simple_prompts = []
    normalization_prompts = []
    documents = bronco_collection.find({"type": annotation_type})
    for document in documents:
        if minimal_length > 0 and len(document["origin"]) < minimal_length:
            continue

        texts = "|".join(document["text"])
        if texts == "":
            texts = "Keine vorhanden"
        simple_prompt_str = extraction.replace("<<CONTEXT>>", document["origin"])
        simple_prompt_str = simple_prompt_str.replace("<<OUTPUT>>", texts)
        simple_prompts.append(simple_prompt_str.strip())

        for index, text in enumerate(document["text"]):
            if document["normalizations"][index][0] is None:
                continue
            else:
                normalization = document["normalizations"][index][0]
                norm_prompt_str = normalization_prompt.replace("<<CONTEXT>>", text)
                norm_prompt_str = norm_prompt_str.replace(
                    "<<OUTPUT>>", normalization["normalization"].split(":")[1]
                )
                normalization_prompts.append(norm_prompt_str.strip())

    return simple_prompts, normalization_prompts


def get_all_bronco_prompts(minimal_length: int, extraction=True, normalization=True):
    """
    Get all prompts from bronco corpus
    :param minimal_length: The minimal length of origin texts to include
    :param extraction: If extraction prompts should be included
    :param normalization: If normalization prompts should be included
    :return: List of prompts
    """
    prompts = set()

    # prompts with annotations
    medication_prompts, medication_norm_prompts = get_bronco_prompts(
        "MEDICATION", MEDICATION_PROMPT, MEDICATION_NORMALIZATION_PROMPT, minimal_length
    )
    diagnosis_prompts, diagnosis_norm_prompts = get_bronco_prompts(
        "DIAGNOSIS", DIAGNOSIS_PROMPT, DIAGNOSIS_NORMALIZATION_PROMPT, minimal_length
    )
    treatment_prompts, treatment_norm_prompts = get_bronco_prompts(
        "TREATMENT", TREATMENT_PROMPT, TREATMENT_NORMALIZATION_PROMPT, minimal_length
    )
    # prompts without annotations
    empty_medication_prompts, empty_medication_norm_prompts = get_bronco_prompts(
        "NA", MEDICATION_PROMPT, MEDICATION_NORMALIZATION_PROMPT, minimal_length
    )
    empty_diagnosis_prompts, empty_diagnosis_norm_prompts = get_bronco_prompts(
        "NA", DIAGNOSIS_PROMPT, DIAGNOSIS_NORMALIZATION_PROMPT, minimal_length
    )
    empty_treatment_prompts, empty_treatment_norm_prompts = get_bronco_prompts(
        "NA", TREATMENT_PROMPT, TREATMENT_NORMALIZATION_PROMPT, minimal_length
    )

    if extraction:
        prompts.update(medication_prompts)
        prompts.update(diagnosis_prompts)
        prompts.update(treatment_prompts)
        prompts.update(empty_medication_prompts)
        prompts.update(empty_diagnosis_prompts)
        prompts.update(empty_treatment_prompts)
    if normalization:
        prompts.update(medication_norm_prompts)
        prompts.update(diagnosis_norm_prompts)
        prompts.update(treatment_norm_prompts)

    logger.debug(
        f"Created {len(prompts)} prompts from the bronco corpus "
        f"[minimal length: {minimal_length}, extraction {extraction}, normalization {normalization}]."
    )

    return list(prompts)
