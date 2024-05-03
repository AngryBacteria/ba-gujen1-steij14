from research.datacorpus.aggregation.prompts import (
    SYSTEM_PROMPT,
    MEDICATION_INSTRUCTION,
    MEDICATION_NORMALIZATION_INSTRUCTION,
    DIAGNOSIS_NORMALIZATION_INSTRUCTION,
    TREATMENT_NORMALIZATION_INSTRUCTION,
    TREATMENT_INSTRUCTION_LEVEL_OF_TRUTH,
    DIAGNOSIS_INSTRUCTION_LEVEL_OF_TRUTH,
    DIAGNOSIS_INSTRUCTION,
    TREATMENT_INSTRUCTION,
    MEDICATION_INSTRUCTION_LEVEL_OF_TRUTH,
    SYSTEM_PROMPT_NORMALIZATION,
)
from research.datacorpus.creation.utils.utils_mongodb import get_collection
from research.logger import logger

bronco_collection = get_collection("corpus", "bronco")
bronco_ner_collection = get_collection("corpus", "bronco_ner")


def get_bronco_instruction(annotation_type: str, add_level_of_truth: bool):
    """Helper function for getting the instruction strings for a given annotation type.
    :param annotation_type: The annotation type to get the instruction string for. Can be
    one of the following: DIAGNOSIS, TREATMENT, MEDICATION
    :param add_level_of_truth: Boolean to if the level of truth shoul dbe added to the instruction string
    """
    if add_level_of_truth:
        if annotation_type == "DIAGNOSIS":
            extraction_instruction = DIAGNOSIS_INSTRUCTION_LEVEL_OF_TRUTH
            normalization_instruction = DIAGNOSIS_NORMALIZATION_INSTRUCTION
        elif annotation_type == "TREATMENT":
            extraction_instruction = TREATMENT_INSTRUCTION_LEVEL_OF_TRUTH
            normalization_instruction = TREATMENT_NORMALIZATION_INSTRUCTION
        elif annotation_type == "MEDICATION":
            extraction_instruction = MEDICATION_INSTRUCTION_LEVEL_OF_TRUTH
            normalization_instruction = MEDICATION_NORMALIZATION_INSTRUCTION
        else:
            raise ValueError("Invalid annotation type")

    else:
        if annotation_type == "DIAGNOSIS":
            extraction_instruction = DIAGNOSIS_INSTRUCTION
            normalization_instruction = DIAGNOSIS_NORMALIZATION_INSTRUCTION
        elif annotation_type == "TREATMENT":
            extraction_instruction = TREATMENT_INSTRUCTION
            normalization_instruction = TREATMENT_NORMALIZATION_INSTRUCTION
        elif annotation_type == "MEDICATION":
            extraction_instruction = MEDICATION_INSTRUCTION
            normalization_instruction = MEDICATION_NORMALIZATION_INSTRUCTION
        else:
            raise ValueError("Invalid annotation type")

    return extraction_instruction, normalization_instruction


# TODO: add prompts with no annotation for the entity, but that have annotations for other entities
def get_bronco_prompts(
    annotation_type: str,
    na_prompts: bool,
    add_level_of_truth: bool,
    minimal_length: int,
):
    """
    Generic function to get prompts from bronco corpus
    :param annotation_type: The type of annotation to get prompts for. Can be
    one of the following: DIAGNOSIS, TREATMENT, MEDICATION
    :param na_prompts: Indicates if the prompts should be created for examples without any annotations
    :param add_level_of_truth: If the level of truth should be added to the text
    :param minimal_length: Minimal length of origin texts to include
    :return: List of prompts
    """
    extraction_prompts = []
    normalization_prompts = []
    extraction_instruction, normalization_instruction = get_bronco_instruction(
        annotation_type, add_level_of_truth
    )
    # if na_prompt is enabled use "na" as annotation type to get the right examples from the db
    if na_prompts:
        annotation_type = "NA"

    documents = bronco_collection.find({"type": annotation_type})
    for document in documents:
        if minimal_length > 0 and len(document["origin"]) < minimal_length:
            continue

        texts = []
        # add level of truth and localisation to the text
        if add_level_of_truth:
            for index, extraction_text in enumerate(document["text"]):
                attributes = []
                is_positive = True
                for attribute in document["attributes"][index]:
                    if (
                        add_level_of_truth
                        and attribute["attribute_label"] == "LevelOfTruth"
                    ):
                        if attribute["attribute"] == "negative":
                            attributes.append("NEGATIV")
                            is_positive = False
                        if attribute["attribute"] == "speculative":
                            attributes.append("SPEKULATIV")
                            is_positive = False
                        if attribute["attribute"] == "possibleFuture":
                            attributes.append("ZUKÜNFTIG")
                            is_positive = False

                if is_positive:
                    attributes.append("POSITIV")
                if len(attributes) < 1:
                    texts.append(extraction_text)
                else:
                    texts.append(f"{extraction_text} [{'|'.join(attributes)}]")
        else:
            texts = document["text"]
        # remove duplicates from text
        texts = list(set(texts))

        # concatenate extraction texts
        extraction_string = "|".join(texts)
        if extraction_string == "":
            extraction_string = "Keine vorhanden"

        extraction_instruction_str = extraction_instruction.replace(
            "<<CONTEXT>>", document["origin"]
        )
        extraction_prompts.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": extraction_instruction_str.strip()},
                    {"role": "assistant", "content": extraction_string.strip()},
                ],
                "type": annotation_type,
                "task": "extraction",
                "source": "bronco",
                "annotation_labels": (
                    extraction_string if extraction_string != "Keine vorhanden" else ""
                ),
            }
        )

        for index, text in enumerate(document["text"]):
            if len(document["normalizations"][index]) < 1:
                continue
            else:
                normalization_entity = document["text"][index]
                norm_instruction_str = normalization_instruction.replace(
                    "<<ENTITY>>", normalization_entity
                )
                norm_instruction_str = norm_instruction_str.replace(
                    "<<CONTEXT>>", document["origin"]
                )
                normalization = document["normalizations"][index][0]
                normalization_prompts.append(
                    {
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT_NORMALIZATION},
                            {"role": "user", "content": norm_instruction_str.strip()},
                            {
                                "role": "assistant",
                                "content": normalization["normalization"].split(":")[1],
                            },
                        ],
                        "type": annotation_type,
                        "task": "normalization",
                        "source": "bronco",
                        "annotation_labels": normalization["normalization"].split(":")[
                            1
                        ],
                    }
                )

    return extraction_prompts, normalization_prompts


def aggregate_bronco_ner():
    """
    Get all NER documents from bronco corpus
    :return: List of NER dictionaries
    """
    ner_docs = []
    documents = bronco_ner_collection.find({})
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
            {
                "words": document["words"],
                "ner_tags": ner_tags,
                "source": "bronco",
            }
        )
    logger.debug(f"Created {len(ner_docs)} ner datapoints from the bronco corpus.")
    return ner_docs


def aggregate_bronco_prompts(
    extraction: bool,
    normalization: bool,
    diagnosis: bool,
    treatment: bool,
    medication: bool,
    na_prompts: bool,
    minimal_length=15,
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
    if diagnosis:
        diagnosis_prompts, diagnosis_norm_prompts = get_bronco_prompts(
            "DIAGNOSIS",
            False,
            True,
            minimal_length,
        )
        if extraction:
            prompts.extend(diagnosis_prompts)
        if normalization:
            prompts.extend(diagnosis_norm_prompts)
    if treatment:
        treatment_prompts, treatment_norm_prompts = get_bronco_prompts(
            "TREATMENT",
            False,
            True,
            minimal_length,
        )
        if extraction:
            prompts.extend(treatment_prompts)
        if normalization:
            prompts.extend(treatment_norm_prompts)
    if medication:
        medication_prompts, medication_norm_prompts = get_bronco_prompts(
            "MEDICATION",
            False,
            False,
            minimal_length,
        )
        if extraction:
            prompts.extend(medication_prompts)
        if normalization:
            prompts.extend(medication_norm_prompts)

    # prompts without annotations
    if na_prompts and extraction:
        if diagnosis:
            empty_diagnosis_prompts, empty_diagnosis_norm_prompts = get_bronco_prompts(
                "DIAGNOSIS",
                True,
                True,
                minimal_length,
            )
            prompts.extend(empty_diagnosis_prompts)
        if treatment:
            empty_treatment_prompts, empty_treatment_norm_prompts = get_bronco_prompts(
                "TREATMENT",
                True,
                True,
                minimal_length,
            )
            prompts.extend(empty_treatment_prompts)
        if medication:
            empty_medication_prompts, empty_medication_norm_prompts = (
                get_bronco_prompts(
                    "MEDICATION",
                    True,
                    False,
                    minimal_length,
                )
            )
            prompts.extend(empty_medication_prompts)

    logger.debug(
        f"Created {len(prompts)} prompts from the bronco corpus "
        f"[minimal length: {minimal_length}, extraction {extraction}, normalization {normalization}]."
    )

    return prompts
