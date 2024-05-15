import random

import pandas

from datacorpus.aggregation.prompts import (
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
from datacorpus.utils.ner import group_ner_data
from shared.mongodb import get_collection
from shared.logger import logger


# Aggregation for the bronco corpus. Prompts can be created for all three annotation types (DIAGNOSIS, TREATMENT,
# MEDICATION) and can be created with or without the level of truth. Prompts can also be created for examples without
# any annotations. The aggregation also includes the aggregation of NER documents.


def get_bronco_instruction(annotation_type: str, add_level_of_truth: bool):
    """Helper function for getting the instruction strings for a given annotation type.
    :param annotation_type: The type of annotation to get the instruction string for. Can be
    one of the following: DIAGNOSIS, TREATMENT, MEDICATION
    :param add_level_of_truth: Boolean to if the level of truth should be added to the instruction string or not
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


def get_bronco_na_prompts(
    annotation_type: str,
    add_level_of_truth: bool,
    minimal_length: int,
    percentage: float,
):
    """
    Get prompts for bronco corpus where there are no annotations for the given annotation type
    :param annotation_type: The type of annotation to get prompts for. Can be
    one of the following: DIAGNOSIS, TREATMENT, MEDICATION
    :param add_level_of_truth: If the level of truth should be added to the prompt
    :param minimal_length: Minimal length of origin texts to include
    :param percentage: Percentage of documents that should have no annotation
    """

    # Get the collection and transform it into a pandas DataFrame
    bronco_collection = get_collection("corpus", "bronco")
    documents = bronco_collection.find()
    documents = list(documents)
    df = pandas.DataFrame(documents)

    # Group the DataFrame by 'origin' and aggregate all other columns into lists. Also filter out origins that are
    # smaller than the minimal length
    grouped_df = df.groupby("origin").agg(lambda x: x.tolist()).reset_index()
    # filter out origins smaller than minimal length
    grouped_df = grouped_df[
        grouped_df["origin"].apply(lambda x: len(x)) >= minimal_length
    ]

    # Calculate the target number of documents based on the percentage
    number_of_docs = bronco_collection.count_documents({"type": annotation_type})
    target_docs = int(number_of_docs * percentage)
    # Filter out rows where the annotation type is not in the 'type' list
    filtered = []
    for index, row in grouped_df.iterrows():
        if annotation_type not in row["type"]:
            filtered.append(row)
    # Randomly select the target number of documents from the filtered list and return
    output = random.sample(filtered, target_docs)

    # create prompts
    prompts = []
    for data in output:
        extraction_string = "Keine vorhanden"
        extraction_instruction, normalization_instruction = get_bronco_instruction(
            annotation_type, add_level_of_truth
        )
        extraction_instruction_str = extraction_instruction.replace(
            "<<CONTEXT>>", data["origin"]
        )
        prompts.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": extraction_instruction_str.strip()},
                    {"role": "assistant", "content": extraction_string.strip()},
                ],
                "type": annotation_type,
                "task": "extraction",
                "source": "bronco",
                "na_prompt": True,
                "annotation_labels": extraction_string,
            }
        )

    return prompts


def get_bronco_prompts(
    annotation_type: str,
    add_level_of_truth: bool,
    minimal_length: int,
):
    """
    Generic function to create the prompts for bronco corpus
    :param annotation_type: The type of annotation to get prompts for. Can be
    one of the following: DIAGNOSIS, TREATMENT, MEDICATION
    :param add_level_of_truth: If the level of truth should be added to the text
    :param minimal_length: Minimal length of origin texts to include
    :return: List of prompts
    """
    extraction_prompts = []
    normalization_prompts = []
    extraction_instruction, normalization_instruction = get_bronco_instruction(
        annotation_type, add_level_of_truth
    )
    bronco_collection = get_collection("corpus", "bronco")
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
            raise ValueError("No extraction string found")

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
                "na_prompt": False,
                "annotation_labels": extraction_string,
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
                        "na_prompt": False,
                        "normalization_labels": normalization_entity,
                        "annotation_labels": normalization["normalization"].split(":")[
                            1
                        ],
                    }
                )

    return extraction_prompts, normalization_prompts


def aggregate_bronco_ner(block_size: int):
    """
    :param block_size: The size of the blocks to create. -1 means no block grouping
    Get all NER documents from the bronco corpus and convert them to the right format for training
    :return: List of NER dictionaries
    """
    ner_docs = []
    bronco_ner_collection = get_collection("corpus", "bronco_ner")
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

    if block_size > 1:
        ner_docs = group_ner_data(ner_docs, block_size, "bronco")

    logger.debug(
        f"Created {len(ner_docs)} ner datapoints from the bronco corpus (block_size={block_size})."
    )
    return ner_docs


def aggregate_bronco_classification():
    """
    Get all documents from the bronco corpus and convert them to the right format for training
    :return: List of classification dictionaries
    """
    classification_docs = []
    bronco_collection = get_collection("corpus", "bronco")
    documents = bronco_collection.find({})
    for document in documents:
        labels = []
        for index, text in enumerate(document["text"]):
            normalizations = document["normalizations"][index]
            for normalization in normalizations:
                if "ICD10" in normalization["normalization"]:
                    normalization_text = normalization["normalization"].split(":")[1]
                    if "." in normalization_text:
                        normalization_text = normalization_text.split(".")[0]
                        labels.append(normalization_text)
                    else:
                        labels.append(normalization_text)

        classification_docs.append(
            {
                "text": document["origin"],
                "labels": labels,
                "source": "bronco",
            }
        )

    data = pandas.DataFrame(classification_docs)
    classification_docs = (
        data.groupby("text")
        .agg(
            {
                "labels": lambda x: [
                    sublist_label for sublist in x for sublist_label in sublist
                ],
                "source": "first",  # Assumes all documents from the same origin have the same source
            }
        )
        .reset_index()
        .to_dict(orient="records")
    )

    unique_labels = set()
    for doc in classification_docs:
        for label in doc["labels"]:
            unique_labels.add(label)
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}


def aggregate_bronco_prompts(
    extraction: bool,
    normalization: bool,
    diagnosis: bool,
    treatment: bool,
    medication: bool,
    na_prompts: bool,
    minimal_length=30,
):
    """
    Get all prompts from bronco corpus
    :param minimal_length: The minimal length of origin texts to include
    :param extraction: If prompts for the extraction task should be included
    :param normalization: If prompts for the normalization task should be included
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
            minimal_length,
        )
        if extraction:
            prompts.extend(medication_prompts)
        if normalization:
            prompts.extend(medication_norm_prompts)

    # prompts without annotations
    if na_prompts and extraction:
        if diagnosis:
            empty_diagnosis_prompts = get_bronco_na_prompts(
                "DIAGNOSIS", True, minimal_length, 0.22
            )
            prompts.extend(empty_diagnosis_prompts)
        if treatment:
            empty_treatment_prompts = get_bronco_na_prompts(
                "TREATMENT", True, minimal_length, 0.22
            )
            prompts.extend(empty_treatment_prompts)
        if medication:
            empty_medication_prompts = get_bronco_na_prompts(
                "MEDICATION", True, minimal_length, 0.22
            )
            prompts.extend(empty_medication_prompts)

    logger.debug(
        f"Aggregated {len(prompts)} prompts from the bronco corpus "
        f"[minimal length: {minimal_length}, extraction {extraction}, normalization {normalization}]."
    )

    return prompts


huhs = aggregate_bronco_prompts(True, True, True, True, True, True, 15)
huhs = random.sample(huhs, 2)
for huh in huhs:
    print(huh["type"], huh["na_prompt"])
    print(huh["messages"][1]["content"])
    print(huh["messages"][2]["content"])
    print(
        "----------------------------------------------------------------------------"
    )
