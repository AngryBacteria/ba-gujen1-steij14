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


# TODO: add prompts with no annotation for the entity, but that have annotations for other entities
def get_bronco_prompts(
    annotation_type: str,
    extraction: str,
    normalization_prompt: str,
    minimal_length: int,
    add_level_of_truth: bool = False,
    add_localisation: bool = False,
):
    """
    Generic function to get prompts from bronco corpus
    :param annotation_type: The type of annotation to get prompts for
    :param extraction: The prompt format for the extraction
    :param normalization_prompt: The prompt format for the normalization task
    :param minimal_length: Minimal length of origin texts to include
    :param add_level_of_truth: If the level of truth should be added to the text
    :param add_localisation: If the localisation should be added to the text
    :return: List of prompts
    """
    simple_prompts = []
    normalization_prompts = []
    documents = bronco_collection.find({"type": annotation_type})
    for document in documents:
        if minimal_length > 0 and len(document["origin"]) < minimal_length:
            continue

        texts = []
        # add level of truth and localisation to the text
        if add_level_of_truth or add_localisation:
            for index, extraction_text in enumerate(document["text"]):
                attributes = []
                for attribute in document["attributes"][index]:
                    if (
                        add_level_of_truth
                        and attribute["attribute_label"] == "LevelOfTruth"
                    ):
                        if attribute["attribute"] == "negative":
                            attributes.append("negativ")
                        if attribute["attribute"] == "speculative":
                            attributes.append("spekulativ")
                        if attribute["attribute"] == "possibleFuture":
                            attributes.append("zukünftig")
                    if (
                        add_localisation
                        and attribute["attribute_label"] == "Localisation"
                    ):
                        if attribute["attribute"] == "L":
                            attributes.append("links")
                        if attribute["attribute"] == "R":
                            attributes.append("rechts")
                        if attribute["attribute"] == "B":
                            attributes.append("beidseitig")

                if len(attributes) < 1:
                    texts.append(extraction_text)
                else:
                    texts.append(f"{extraction_text} [{'|'.join(attributes)}]")
        else:
            texts = document["text"]
        # remove duplicates from text
        texts = list(set(texts))

        # concatenate extraction texts
        text = "|".join(texts)
        if text == "":
            text = "Keine vorhanden"

        simple_prompt_str = extraction.replace("<<CONTEXT>>", document["origin"])
        simple_prompt_str = simple_prompt_str.replace("<<OUTPUT>>", text)
        simple_prompts.append(
            {
                "text": simple_prompt_str.strip(),
                "type": annotation_type,
                "task": "extraction",
                "source": "bronco",
                "extraction_labels": texts if texts != "Keine vorhanden" else "",
            }
        )

        for index, text in enumerate(document["text"]):
            if len(document["normalizations"][index]) < 1:
                continue
            else:
                normalization_entity = document["text"][index]
                norm_prompt_str = normalization_prompt.replace(
                    "<<ENTITY>>", normalization_entity
                )

                normalization = document["normalizations"][index][0]
                norm_prompt_str = norm_prompt_str.replace(
                    "<<CONTEXT>>", document["origin"]
                )
                norm_prompt_str = norm_prompt_str.replace(
                    "<<OUTPUT>>", normalization["normalization"].split(":")[1]
                )
                normalization_prompts.append(
                    {
                        "text": norm_prompt_str.strip(),
                        "type": annotation_type,
                        "task": "normalization",
                        "source": "bronco",
                    }
                )

    return simple_prompts, normalization_prompts


def get_all_bronco_prompts(
    minimal_length: int,
    extraction=True,
    normalization=True,
    diagnosis=True,
    treatment=True,
    medication=True,
    na_prompts=True,
):
    """
    Get all prompts from bronco corpus
    :param minimal_length: The minimal length of origin texts to include
    :param extraction: If extraction prompts should be included
    :param normalization: If normalization prompts should be included
    :param diagnosis: If diagnosis prompts should be included
    :param treatment: If treatment prompts should be included
    :param medication: If medication prompts should be included
    :param na_prompts: If prompts without any annotations should be included
    :return: List of prompts
    """
    prompts = []
    # prompts with annotations
    if medication:
        medication_prompts, medication_norm_prompts = get_bronco_prompts(
            "MEDICATION",
            MEDICATION_PROMPT,
            MEDICATION_NORMALIZATION_PROMPT,
            minimal_length,
            add_level_of_truth=False,
        )
        if extraction:
            prompts.extend(medication_prompts)
        if normalization:
            prompts.extend(medication_norm_prompts)

    if diagnosis:
        diagnosis_prompts, diagnosis_norm_prompts = get_bronco_prompts(
            "DIAGNOSIS",
            DIAGNOSIS_PROMPT,
            DIAGNOSIS_NORMALIZATION_PROMPT,
            minimal_length,
            add_level_of_truth=True,
        )
        if extraction:
            prompts.extend(diagnosis_prompts)
        if normalization:
            prompts.extend(diagnosis_norm_prompts)
    if treatment:
        treatment_prompts, treatment_norm_prompts = get_bronco_prompts(
            "TREATMENT",
            TREATMENT_PROMPT,
            TREATMENT_NORMALIZATION_PROMPT,
            minimal_length,
            add_level_of_truth=True,
        )
        if extraction:
            prompts.extend(treatment_prompts)
        if normalization:
            prompts.extend(treatment_norm_prompts)

    # prompts without annotations
    if na_prompts and extraction:
        empty_medication_prompts, empty_medication_norm_prompts = get_bronco_prompts(
            "NA", MEDICATION_PROMPT, MEDICATION_NORMALIZATION_PROMPT, minimal_length
        )
        empty_diagnosis_prompts, empty_diagnosis_norm_prompts = get_bronco_prompts(
            "NA", DIAGNOSIS_PROMPT, DIAGNOSIS_NORMALIZATION_PROMPT, minimal_length
        )
        empty_treatment_prompts, empty_treatment_norm_prompts = get_bronco_prompts(
            "NA", TREATMENT_PROMPT, TREATMENT_NORMALIZATION_PROMPT, minimal_length
        )
        prompts.extend(empty_medication_prompts)
        prompts.extend(empty_diagnosis_prompts)
        prompts.extend(empty_treatment_prompts)

    logger.debug(
        f"Created {len(prompts)} prompts from the bronco corpus "
        f"[minimal length: {minimal_length}, extraction {extraction}, normalization {normalization}]."
    )

    return prompts
