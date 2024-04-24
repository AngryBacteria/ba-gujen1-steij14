# TODO: aggregation of all corpus datasets
from datasets import load_dataset

from research.datacorpus.aggregation.agg_bronco import get_all_bronco_prompts
from research.datacorpus.aggregation.agg_cardio import get_cardio_pretrain_texts
from research.datacorpus.aggregation.agg_clef import get_clef_pretrain_texts
from research.datacorpus.aggregation.agg_ggponc import get_all_ggponc_prompts
import pandas as pd

from research.datacorpus.aggregation.agg_jsyncc import get_jsyncc_pretrain_texts
from research.logger import logger


def save_all_prompts(bronco=True, ggponc=True, normalization=True, ignore_short=10):
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

    prompts_df = pd.DataFrame(prompts, columns=["text"])
    prompts_df.to_csv("prompts.csv", index=False)
    logger.debug(f"Saved {len(prompts)} prompts to prompts.csv")


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

    prompts_df = pd.DataFrame(texts, columns=["text"])
    prompts_df.to_csv("pretrain.csv", index=False)
    logger.debug(f"Saved {len(texts)} prompts to prompts.csv")


def count_training_tokens():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "LeoLM/leo-mistral-hessianai-7b", use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("csv", data_files={"data": "prompts.csv"})[
        "data"
    ].train_test_split(test_size=0.1, shuffle=True, seed=42)
    # iterate over dataset
    token_count = 0
    for i, example in enumerate(dataset["train"]):
        token_count = token_count + len(tokenizer.tokenize(example["text"]))

    print("Total number of tokens in training dataset: ", token_count)
    return token_count


# count_training_tokens()
# save_all_pretrain_texts()
# save_all_prompts()
