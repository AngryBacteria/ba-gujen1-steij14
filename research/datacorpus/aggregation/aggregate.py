# TODO: aggregation of all corpus datasets
from research.datacorpus.aggregation.agg_bronco import (
    get_all_simple_bronco_prompts,
    get_all_bronco_normalization_prompts,
)
from research.datacorpus.aggregation.agg_ggponc import get_all_simple_ggponc_prompts
import pandas as pd


def save_all_prompts(normalization=True):
    prompts = []
    ggponc_prompts = get_all_simple_ggponc_prompts()
    simple_bronco_prompts = get_all_simple_bronco_prompts()
    if normalization:
        bronco_normalization_prompts = get_all_bronco_normalization_prompts()
    else:
        bronco_normalization_prompts = []

    prompts.extend(ggponc_prompts)
    prompts.extend(simple_bronco_prompts)
    prompts.extend(bronco_normalization_prompts)

    prompts_df = pd.DataFrame(prompts)
    prompts_df.to_csv("prompts.csv", index=False)
