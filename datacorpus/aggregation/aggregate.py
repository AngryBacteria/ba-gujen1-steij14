import pandas as pd
from datasets import load_dataset

from datacorpus.aggregation.agg_bronco import (
    aggregate_bronco_prompts,
    aggregate_bronco_ner,
)
from datacorpus.aggregation.agg_cardio import (
    aggregate_cardio_pretrain_texts,
    aggregate_cardio_prompts,
    aggregate_cardio_ner,
)
from datacorpus.aggregation.agg_clef import aggregate_clef_pretrain_texts
from datacorpus.aggregation.agg_ggponc import (
    aggregate_ggponc_prompts,
    aggregate_ggponc_ner,
)
from datacorpus.aggregation.agg_jsyncc import aggregate_jsyncc_pretrain_texts
from shared.logger import logger
from shared.model_utils import get_tokenizer_with_template

# Takes all aggregation functions from the individual sources and combines them into the required format for the
# training data. Methods are available to save prompts, ner, and pretrain data into json files.


def get_unique_prompts(prompts: list[dict]) -> list[dict]:
    """
    Get unique prompts from a list of prompt dictionaries.
    The text value is used as the unique identifier.
    :param prompts: List of dictionaries with prompts
    :return: List of unique prompts
    """
    unique_prompts = {}
    for prompt in prompts:
        text_value = prompt["messages"][1]["content"] + prompt["messages"][2]["content"]
        if text_value not in unique_prompts:
            unique_prompts[text_value] = prompt
    return list(unique_prompts.values())


# todo: ADD distribute na and not na prompts
def save_all_prompts(
    bronco: bool,
    ggponc: bool,
    cardio: bool,
    normalization: bool,
    na_prompts: bool,
    minimal_length=15,
):
    # get prompts from the datasets
    prompts = []
    if ggponc:
        ggponc_prompts = aggregate_ggponc_prompts(
            minimal_length=minimal_length,
            diagnosis=False,
            treatment=False,
            medication=True,
            na_prompts=na_prompts,
        )
        prompts.extend(ggponc_prompts)
    if bronco:
        bronco_prompts = aggregate_bronco_prompts(
            minimal_length=minimal_length,
            extraction=True,
            normalization=normalization,
            diagnosis=True,
            treatment=True,
            medication=True,
            na_prompts=na_prompts,
        )
        prompts.extend(bronco_prompts)
    if cardio:
        cardio_prompts = aggregate_cardio_prompts(attributes=True)
        prompts.extend(cardio_prompts)

    # filter out unique prompts
    prompts = get_unique_prompts(prompts)
    # apply chat template and strip whitespace
    tokenizer = get_tokenizer_with_template()
    prompts_df = pd.DataFrame(prompts)
    prompts_df["text"] = prompts_df["messages"].apply(
        lambda x: tokenizer.apply_chat_template(
            x, tokenize=False, add_generation_prompt=False
        )
    )
    prompts_df["text"] = prompts_df["text"].apply(lambda x: x.strip())

    # save to a json file
    prompts_df.to_json("prompts.jsonl", orient="records", lines=True)
    logger.debug(f"Saved {len(prompts)} prompts to prompts.jsonl")

    # load with huggingface datasets and print some examples
    data = load_dataset("json", data_files={"data": "prompts.jsonl"})[
        "data"
    ].train_test_split(test_size=0.1, shuffle=True, seed=42)
    print(data)
    amount = 20
    for i, example in enumerate(data["train"]):
        print(example["text"])
        print("-----------------------------------------")
        if i > amount:
            break


def save_all_ner_annotations(bronco: bool, ggponc: bool, cardio: bool):
    ner_annotations = []
    if bronco:
        bronco_ner_annotations = aggregate_bronco_ner()
        ner_annotations.extend(bronco_ner_annotations)
    if ggponc:
        ggponc_ner_annotations = aggregate_ggponc_ner()
        ner_annotations.extend(ggponc_ner_annotations)
    if cardio:
        cardio_ner_annotations = aggregate_cardio_ner()
        ner_annotations.extend(cardio_ner_annotations)

    ner_annotations_df = pd.DataFrame(ner_annotations)
    ner_annotations_df.to_json("ner.jsonl", orient="records", lines=True)
    logger.debug(f"Saved {len(ner_annotations)} ner annotations to ner.jsonl")

    data = load_dataset("json", data_files={"data": "ner.jsonl"})[
        "data"
    ].train_test_split(test_size=0.1, shuffle=True, seed=42)
    print(data)
    amount = 50
    for i, example in enumerate(data["train"]):
        print(example["ner_tags"])
        if i > amount:
            break


def save_all_pretrain_texts(clef=True, cardio=True, jsyncc=True):
    """
    Save all pretrain texts to pretrain.csv
    :return: None
    """
    texts = []
    if clef:
        clef_texts = aggregate_clef_pretrain_texts()
        texts.extend(clef_texts)

    if cardio:
        cardio_texts = aggregate_cardio_pretrain_texts()
        texts.extend(cardio_texts)

    if jsyncc:
        jsyncc_texts = aggregate_jsyncc_pretrain_texts()
        texts.extend(jsyncc_texts)

    texts = get_unique_prompts(texts)

    pretrain_df = pd.DataFrame(texts)
    pretrain_df.to_json("pretrain.json", orient="records")
    logger.debug(f"Saved {len(texts)} prompts to pretrain.json")

    data = load_dataset("json", data_files={"data": "pretrain.json"})[
        "data"
    ].train_test_split(test_size=0.1, shuffle=True, seed=42)
    print(data)
    amount = 20
    for i, example in enumerate(data["train"]):
        print(example["text"])
        if i > amount:
            break


# save_all_ner_annotations(bronco=True, ggponc=False, cardio=False)

save_all_prompts(
    bronco=True,
    ggponc=False,
    cardio=True,
    normalization=True,
    na_prompts=True,
    minimal_length=15,
)

# save_all_pretrain_texts(clef=True, cardio=True, jsyncc=True)
