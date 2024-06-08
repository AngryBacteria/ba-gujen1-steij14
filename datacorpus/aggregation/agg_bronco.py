import random
from collections import Counter

import pandas
from sklearn.preprocessing import MultiLabelBinarizer

from datacorpus.utils.ner import group_ner_data
from shared.logger import logger
from shared.mongodb import get_collection
from shared.prompt_utils import (
    get_extraction_messages,
    AttributeFormat,
    EntityType,
    get_normalization_messages,
)


# Aggregation for the bronco database collection. Prompts can be created for all three annotation types (DIAGNOSIS,
# TREATMENT and MEDICATION) and can be created with or without the level of truth. Prompts can also be created for
# texts without any annotations. The aggregation also includes the aggregation of NER documents.


def get_bronco_na_prompts(
    entity_type: EntityType,
    minimal_length: int,
    na_percentage: float,
):
    """
    Get prompts for bronco corpus where there are no annotations for the given annotation type
    :param entity_type: The type of annotation to get prompts for. Can be
    one of the following: DIAGNOSIS, TREATMENT, MEDICATION
    :param minimal_length: Minimal length of origin texts to include
    :param na_percentage: Percentage of documents that should have no annotation
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
    number_of_docs = bronco_collection.count_documents({"type": entity_type.value})
    target_docs = int(number_of_docs * na_percentage)
    # Filter out rows where the annotation type is not in the 'type' list
    filtered = []
    for index, row in grouped_df.iterrows():
        if entity_type.value not in row["type"]:
            filtered.append(row)
    # Randomly select the target number of documents from the filtered list and return
    output = random.sample(filtered, target_docs)

    # create prompts
    prompts = []
    for data in output:
        messages = get_extraction_messages(
            data["origin"], AttributeFormat.BRONCO, entity_type
        )
        extraction_string = "Keine vorhanden"
        messages.append({"role": "assistant", "content": extraction_string})

        prompts.append(
            {
                "messages": messages,
                "type": entity_type.value,
                "task": "extraction",
                "source": "bronco",
                "na_prompt": True,
                "context": data["origin"],
                "context_entity": "",
                "output": extraction_string,
            }
        )

    return prompts


def get_bronco_prompts(
    entity_type: EntityType,
    minimal_length: int,
):
    """
    Generic function to create the prompts for bronco corpus
    :param entity_type: The type of annotation to get prompts for. Can be
    one of the following: DIAGNOSIS, TREATMENT, MEDICATION
    :param minimal_length: Minimal length of origin texts to include
    :return: List of prompts
    """
    extraction_prompts = []
    normalization_prompts = []

    bronco_collection = get_collection("corpus", "bronco")
    documents = bronco_collection.find({"type": entity_type.value})
    for document in documents:
        if minimal_length > 0 and len(document["origin"]) < minimal_length:
            continue

        texts = []
        # add attributes to the text
        for index, extraction_text in enumerate(document["text"]):
            attributes = []
            is_positive = True
            for attribute in document["attributes"][index]:
                if attribute["attribute"] == "negative":
                    attributes.append("NEGATIV")
                    is_positive = False
                if attribute["attribute"] == "speculative":
                    attributes.append("SPEKULATIV")
                    is_positive = False
                if attribute["attribute"] == "possibleFuture":
                    attributes.append("ZUKÜNFTIG")
                    is_positive = False

                if attribute["attribute"] == "R":
                    attributes.append("RECHTS")
                if attribute["attribute"] == "L":
                    attributes.append("LINKS")
                if attribute["attribute"] == "B":
                    attributes.append("BEIDSEITIG")

            if is_positive:
                attributes.append("POSITIV")
            if len(attributes) < 1:
                raise ValueError("No attributes found, this should not happen")
            else:
                texts.append(f"{extraction_text} [{'|'.join(attributes)}]")

        # remove duplicates from text
        texts = list(set(texts))
        texts = [text.strip() for text in texts]
        # concatenate extraction texts
        extraction_string = "|".join(texts)
        if extraction_string == "":
            raise ValueError("No extraction string found")

        messages_extraction = get_extraction_messages(
            document["origin"], AttributeFormat.BRONCO, entity_type
        )
        messages_extraction.append({"role": "assistant", "content": extraction_string})
        extraction_prompts.append(
            {
                "messages": messages_extraction,
                "type": entity_type.value,
                "task": "extraction",
                "source": "bronco",
                "na_prompt": False,
                "context": document["origin"],
                "context_entity": "",
                "output": extraction_string,
            }
        )

        for index, text in enumerate(document["text"]):
            if len(document["normalizations"][index]) < 1:
                continue
            else:
                normalization_entity = document["text"][index]
                normalization = document["normalizations"][index][0]
                normalization = normalization["normalization"].split(":")[1].strip()

                messages_normalization = get_normalization_messages(
                    normalization_entity, document["origin"], entity_type
                )
                messages_normalization.append(
                    {"role": "assistant", "content": normalization}
                )
                normalization_prompts.append(
                    {
                        "messages": messages_normalization,
                        "type": entity_type.value,
                        "task": "normalization",
                        "source": "bronco",
                        "na_prompt": False,
                        "context": document["origin"],
                        "context_entity": normalization_entity,
                        "output": normalization,
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


def aggregate_bronco_prompts(
    extraction: bool,
    normalization: bool,
    diagnosis: bool,
    treatment: bool,
    medication: bool,
    na_prompts: bool,
    minimal_length: int,
    na_percentage: float,
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
    :param na_percentage: The percentage of na prompts that should be included
    :return: List of prompts
    """
    prompts = []
    # prompts with annotations
    if diagnosis:
        diagnosis_prompts, diagnosis_norm_prompts = get_bronco_prompts(
            EntityType.DIAGNOSIS,
            minimal_length,
        )
        if extraction:
            prompts.extend(diagnosis_prompts)
        if normalization:
            prompts.extend(diagnosis_norm_prompts)
    if treatment:
        treatment_prompts, treatment_norm_prompts = get_bronco_prompts(
            EntityType.TREATMENT,
            minimal_length,
        )
        if extraction:
            prompts.extend(treatment_prompts)
        if normalization:
            prompts.extend(treatment_norm_prompts)
    if medication:
        medication_prompts, medication_norm_prompts = get_bronco_prompts(
            EntityType.MEDICATION,
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
                EntityType.DIAGNOSIS, minimal_length, na_percentage
            )
            prompts.extend(empty_diagnosis_prompts)
        if treatment:
            empty_treatment_prompts = get_bronco_na_prompts(
                EntityType.TREATMENT, minimal_length, na_percentage
            )
            prompts.extend(empty_treatment_prompts)
        if medication:
            empty_medication_prompts = get_bronco_na_prompts(
                EntityType.MEDICATION, minimal_length, na_percentage
            )
            prompts.extend(empty_medication_prompts)

    logger.debug(
        f"Aggregated {len(prompts)} prompts from the bronco corpus "
        f"[minimal length: {minimal_length}, extraction {extraction}, normalization {normalization}]."
    )

    return prompts


def aggregate_bronco_multi_label_classification(
    task: str,
    normalization_type: str,
    detailed: bool,
    top_x_labels=10,
    include_other=True,
    min_length=15,
):
    """
    Create a bronco dataset for multi-label classification
    :param detailed: How many layers deep to go for icd10/ops codes. True means all, False means 3-4 layers.
    :param task: The task to create the dataset for. Can be one of the following: Normalization, Entity_Type, Attribute
    :param normalization_type: The type of entity to create the dataset for. Such as "ICD10", "OPS", "ATC".
    You can also pass "ALL" to include all types.
    :param top_x_labels: The number of top x labels to create the dataset for.
    :param include_other: Whether to include a category "other" which means it is not in the top_x labels.
    Can be one of the following: ICD10GM, OSP, ATC
    :param min_length: The minimal length of origin texts to include
    :return:
    """
    # load and group
    bronco_collection = get_collection("corpus", "bronco")
    documents = bronco_collection.find()
    documents = list(documents)
    df = pandas.DataFrame(documents)
    grouped_df = df.groupby("origin").agg(lambda x: x.tolist()).reset_index()
    # filter out origins smaller than minimal length
    grouped_df = grouped_df[grouped_df["origin"].apply(lambda x: len(x)) >= min_length]

    # make types unique
    all_labels = []
    if task == "Entity_Type":
        for i, row in grouped_df.iterrows():
            unique_types = list(set(row["type"]))
            all_labels.append(unique_types)
    elif task == "Normalization":
        for i, row in grouped_df.iterrows():
            unique_normalizations = set()
            for pack in row["normalizations"]:
                for unpacked in pack:
                    for normalization in unpacked:
                        if (
                            normalization["normalization"].startswith(
                                normalization_type
                            )
                            or normalization_type == "ALL"
                        ):
                            norm = normalization["normalization"].split(":")[1]
                            if not detailed:
                                norm = norm.split(".")[0]
                            unique_normalizations.add(norm.strip())
            all_labels.append(list(unique_normalizations))
    elif task == "Attribute":
        for i, row in grouped_df.iterrows():
            unique_attributes = set()
            level_of_truth_found = False
            for pack in row["attributes"]:
                for unpacked in pack:
                    for attribute in unpacked:
                        unique_attributes.add(attribute["attribute"].strip())
                        if attribute["attribute"] == "level_of_truth":
                            level_of_truth_found = True
            if include_other and not level_of_truth_found:
                unique_attributes.add("positive")
            all_labels.append(list(unique_attributes))

    else:
        raise ValueError("Not implemented")

    # Count label frequencies
    label_counter = Counter(label for labels in all_labels for label in labels)
    top_labels = {label for label, _ in label_counter.most_common(top_x_labels)}
    logger.debug(f"Label counts: {label_counter}")
    # Filter labels and add "OTHER" category for labels not in top X if include_other is True
    filtered_labels = []
    for labels in all_labels:
        if include_other:
            filtered = [label if label in top_labels else "OTHER" for label in labels]
        else:
            filtered = [label for label in labels if label in top_labels]
        filtered_labels.append(filtered)

    # transform into one-hot encoding
    mlb = MultiLabelBinarizer()
    one_hot_encoded = mlb.fit_transform(filtered_labels).astype(float)
    grouped_df["labels"] = list(one_hot_encoded)

    # remove all other columns
    grouped_df = grouped_df[["origin", "labels"]]

    label2id = {label: idx for idx, label in enumerate(mlb.classes_)}
    id2label = {idx: label for idx, label in enumerate(mlb.classes_)}

    print(label2id)
    print(id2label)
    num_labels = len(label2id)

    logger.debug(
        f"Created {len(grouped_df)} classification datapoints with {num_labels} labels."
    )
    logger.debug(grouped_df.head(10))

    return grouped_df, label2id, id2label, num_labels


# if main method
if __name__ == "__main__":
    data, label2id, id2label, NUM_LABELS = aggregate_bronco_multi_label_classification(
        "Attribute", "ALL", True, 1000000, False
    )
