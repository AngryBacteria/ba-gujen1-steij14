import pandas as pd
from datasets import load_dataset

from shared.decoder_utils import count_tokens
from shared.prompt_utils import TaskType

# Collection of functions to analyze the prompts that were created with the aggregation script.


if __name__ == "__main__":
    # load the dataset
    data = load_dataset("json", data_files={"data": "prompts.jsonl"})["data"]
    extraction_prompts = []
    normalization_prompts = []
    summary_prompts = []
    catalog_prompts = []
    # organize prompts into different categories
    for row in data:
        if row["task"] == TaskType.EXTRACTION.value:
            extraction_prompts.append(row)
        elif row["task"] == TaskType.NORMALIZATION.value:
            normalization_prompts.append(row)
        elif row["task"] == TaskType.SUMMARIZATION.value:
            summary_prompts.append(row)
        elif row["task"] == TaskType.CATALOG.value:
            catalog_prompts.append(row)

    # create dataframes for each category
    extraction_df = pd.DataFrame(extraction_prompts)
    normalization_df = pd.DataFrame(normalization_prompts)
    summary_df = pd.DataFrame(summary_prompts)
    catalog_df = pd.DataFrame(catalog_prompts)
    total_tokens = 0
    total_prompts = 0

    # display the number of prompts and tokens for each category
    print(f"{30*'-'}Extraction{30*'-'}")
    extraction_df = extraction_df.groupby(["source", "type"])
    for source, group in extraction_df:
        tokens = count_tokens(group["text"])
        total_tokens += tokens
        total_prompts += len(group)
        print(source, len(group), tokens)

    print(f"{30 * '-'}Normalization{30 * '-'}")
    normalization_df = normalization_df.groupby(["source", "type"])
    for source, group in normalization_df:
        tokens = count_tokens(group["text"])
        total_tokens += tokens
        total_prompts += len(group)
        print(source, len(group), tokens)

    print(f"{30 * '-'}Summary{30 * '-'}")
    summary_df = summary_df.groupby(["source"])
    for source, group in summary_df:
        tokens = count_tokens(group["text"])
        total_tokens += tokens
        total_prompts += len(group)
        print(source, len(group), tokens)

    print(f"{30 * '-'}Catalog{30 * '-'}")
    catalog_df = catalog_df.groupby(["source"])
    for source, group in catalog_df:
        tokens = count_tokens(group["text"])
        total_tokens += tokens
        total_prompts += len(group)
        print(source, len(group), tokens)

    print(f"Total tokens: {total_tokens}")
    print(f"Total prompts: {total_prompts}")
