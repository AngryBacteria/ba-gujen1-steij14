from research.datacorpus.aggregation.prompts import (
    MEDICATION_PROMPT,
    TREATMENT_PROMPT,
    DIAGNOSIS_PROMPT,
)
from research.datacorpus.creation.utils.utils_mongodb import get_collection
from research.logger import logger

ggonc_collection = get_collection("corpus", "ggponc_short")
ggonc_collection_ner = get_collection("corpus", "ggponc_short_ner")


# TODO: add prompts with no annotation for the entity, but that have annotations for other entities
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
            texts = list(set(anno["text"]))

            # concatenate
            extraction_string = "|".join(texts)
            if extraction_string == "":
                extraction_string = "Keine vorhanden"

            simple_prompt_str = extraction_prompt.replace("<<CONTEXT>>", anno["origin"])
            simple_prompt_str = simple_prompt_str.replace(
                "<<OUTPUT>>", extraction_string
            )
            prompts.append(
                {
                    "text": simple_prompt_str.strip(),
                    "type": annotation_type,
                    "task": "extraction",
                    "source": "ggponc",
                    "annotation_labels": (
                        extraction_string
                        if extraction_string != "Keine vorhanden"
                        else ""
                    ),
                }
            )

    return prompts


def aggregate_ggponc_ner():
    """
    Get all NER documents from ggponc corpus.
    Filter out all NER tags that are not MEDICATION, TREATMENT, or DIAGNOSIS.
    :return: List of NER annotations as dictionaries
    """
    ner_docs = []
    documents = ggonc_collection_ner.find({})
    for document in documents:
        ner_tags = []
        for tag in documents["ner_tags"]:
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
    if medication:
        medication_prompts = get_ggponc_prompts(
            "MEDICATION", MEDICATION_PROMPT, minimal_length
        )
        prompts.extend(medication_prompts)
    if diagnosis:
        diagnosis_prompts = get_ggponc_prompts(
            "DIAGNOSIS", DIAGNOSIS_PROMPT, minimal_length
        )
        prompts.extend(diagnosis_prompts)
    if treatment:
        treatment_prompts = get_ggponc_prompts(
            "TREATMENT", TREATMENT_PROMPT, minimal_length
        )
        prompts.extend(treatment_prompts)

    # prompts without annotations
    if medication and na_prompts:
        empty_medication_prompts = get_ggponc_prompts(
            "NA", MEDICATION_PROMPT, minimal_length
        )
        prompts.extend(empty_medication_prompts)
    if diagnosis and na_prompts:
        empty_diagnosis_prompts = get_ggponc_prompts(
            "NA", DIAGNOSIS_PROMPT, minimal_length
        )
        prompts.extend(empty_diagnosis_prompts)
    if treatment and na_prompts:
        empty_treatment_prompts = get_ggponc_prompts(
            "NA", TREATMENT_PROMPT, minimal_length
        )
        prompts.extend(empty_treatment_prompts)

    logger.debug(
        f"Created {len(prompts)} prompts from the ggponc corpus [minimal length: {minimal_length}]."
    )

    return prompts
