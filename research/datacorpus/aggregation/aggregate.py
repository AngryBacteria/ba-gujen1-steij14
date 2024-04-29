import pandas as pd
from datasets import load_dataset

from research.datacorpus.aggregation.agg_bronco import get_all_bronco_prompts
from research.datacorpus.aggregation.agg_cardio import (
    get_cardio_pretrain_texts,
    aggregate_cardio_prompts,
)
from research.datacorpus.aggregation.agg_clef import get_clef_pretrain_texts
from research.datacorpus.aggregation.agg_ggponc import get_all_ggponc_prompts
from research.datacorpus.aggregation.agg_jsyncc import get_jsyncc_pretrain_texts
from research.logger import logger


def get_unique_prompts(prompts: list[dict]) -> list[dict]:
    """
    Get unique prompts from a list of prompt dictionaries.
    The text value is used as the unique identifier.
    :param prompts: List of dictionaries with prompts
    :return: List of unique prompts
    """
    unique_prompts = {}
    for prompt in prompts:
        text_value = prompt["text"]
        if text_value not in unique_prompts:
            unique_prompts[text_value] = prompt
    return list(unique_prompts.values())


def save_all_prompts(
    bronco: bool,
    ggponc: bool,
    cardio: bool,
    normalization: bool,
    na_prompts: bool,
    minimal_length=15,
):
    prompts = []
    if ggponc:
        ggponc_prompts = get_all_ggponc_prompts(
            minimal_length=minimal_length,
            diagnosis=False,
            treatment=False,
            medication=True,
            na_prompts=na_prompts,
        )
        prompts.extend(ggponc_prompts)

    if bronco:
        bronco_prompts = get_all_bronco_prompts(
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
        cardio_prompts = aggregate_cardio_prompts()
        prompts.extend(cardio_prompts)

    prompts = get_unique_prompts(prompts)
    prompts_df = pd.DataFrame(prompts)
    prompts_df.to_json("prompts.json", orient="records")
    logger.debug(f"Saved {len(prompts)} prompts to prompts.json")


def save_all_pretrain_texts(clef=True, cardio=True, jsyncc=True):
    """
    Save all pretrain texts to pretrain.csv
    :return: None
    """
    texts = []
    if clef:
        clef_texts = get_clef_pretrain_texts()
        texts.extend(clef_texts)

    if cardio:
        cardio_texts = get_cardio_pretrain_texts()
        texts.extend(cardio_texts)

    if jsyncc:
        jsyncc_texts = get_jsyncc_pretrain_texts()
        texts.extend(jsyncc_texts)

    texts = get_unique_prompts(texts)

    pretrain_df = pd.DataFrame(texts)
    pretrain_df.to_json("pretrain.json", orient="records")
    logger.debug(f"Saved {len(texts)} prompts to pretrain.json")


def count_training_tokens():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "LeoLM/leo-mistral-hessianai-7b", use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files={"data": "prompts.json"})[
        "data"
    ].train_test_split(test_size=0.1, shuffle=True, seed=42)
    # iterate over dataset
    token_count = 0
    for i, example in enumerate(dataset["train"]):
        token_count = token_count + len(tokenizer.tokenize(example["text"]))

    print("Total number of tokens in training dataset: ", token_count)
    return token_count


# save prompts
save_all_prompts(
    ggponc=False,
    bronco=True,
    cardio=True,
    normalization=True,
    na_prompts=True,
    minimal_length=15,
)

# print selection of prompts
data = load_dataset("json", data_files={"data": "prompts.json"})[
    "data"
].train_test_split(test_size=0.1, shuffle=True, seed=42)
print(data)
for i, example in enumerate(data["test"]):
    if example["task"] == "extraction" and example["source"] == "bronco":
        print(example["text"])
        print("----------------------------------")
        if i > 25:
            break
