# TODO: aggregation of all corpus datasets
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
    bronco=True, ggponc=True, cardio=True, normalization=True, ignore_short=15
):
    prompts = []
    if ggponc:
        ggponc_prompts = get_all_ggponc_prompts(ignore_short)
        prompts.extend(ggponc_prompts)

    if bronco:
        if normalization:
            bronco_normalization_prompts = get_all_bronco_prompts(
                ignore_short,
                extraction=True,
                normalization=True,
            )
            prompts.extend(bronco_normalization_prompts)
        else:
            simple_bronco_prompts = get_all_bronco_prompts(
                ignore_short,
                extraction=True,
                normalization=False,
            )
            prompts.extend(simple_bronco_prompts)

    if cardio:
        prompts.extend(aggregate_cardio_prompts())

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


save_all_prompts()
# count_training_tokens()
prompt_dataset = load_dataset("json", data_files={"data": "prompts.json"})[
    "data"
].train_test_split(test_size=0.1, shuffle=True, seed=42)

nas = 0
not_nas = 0

for i, example in enumerate(prompt_dataset["train"]):
    if example["type"] == "NA":
        nas += 1
    else:
        not_nas += 1

print("Number of NA prompts: ", nas)
print("Number of non-NA prompts: ", not_nas)