# TODO: aggregation of all corpus datasets
from datasets import load_dataset

from research.datacorpus.aggregation.agg_bronco import (
    get_all_simple_bronco_prompts,
    get_all_bronco_normalization_prompts,
)
from research.datacorpus.aggregation.agg_ggponc import get_all_simple_ggponc_prompts
import pandas as pd


def save_all_prompts(bronco=True, ggponc=True, normalization=True):
    prompts = []
    if ggponc:
        ggponc_prompts = get_all_simple_ggponc_prompts()
        prompts.extend(ggponc_prompts)
    if bronco:
        simple_bronco_prompts = get_all_simple_bronco_prompts()
        prompts.extend(simple_bronco_prompts)
        if normalization:
            bronco_normalization_prompts = get_all_bronco_normalization_prompts()
            prompts.extend(bronco_normalization_prompts)

    prompts_df = pd.DataFrame(prompts)
    prompts_df.to_csv("prompts.csv", index=False, header=["text"])


dataset = load_dataset("csv", data_files={"data": "prompts.csv"})[
    "data"
].train_test_split(test_size=0.1, shuffle=True, seed=42)
print(dataset)
