import pandas as pd
from datasets import load_dataset

from shared.model_utils import count_tokens

# load the dataset
data = load_dataset("json", data_files={"data": "prompts.jsonl"})["data"]
extraction_prompts = []
normalization_prompts = []
summary_prompts = []
for row in data:
    if row["task"] == "extraction":
        extraction_prompts.append(row)
    elif row["task"] == "normalization":
        normalization_prompts.append(row)
    elif row["task"] == "summary":
        summary_prompts.append(row)

extraction_df = pd.DataFrame(extraction_prompts)
normalization_df = pd.DataFrame(normalization_prompts)
summary_df = pd.DataFrame(summary_prompts)


print(f"{30*'-'}Extraction{30*'-'}")
extraction_df = extraction_df.groupby(["source", "type"])
for source, group in extraction_df:
    tokens = count_tokens(group["text"], None, "LeoLM/leo-mistral-hessianai-7b")
    print(source, len(group), tokens)

print(f"{30 * '-'}Normalization{30 * '-'}")
normalization_df = normalization_df.groupby(["source", "type"])
for source, group in normalization_df:
    tokens = count_tokens(group["text"], None, "LeoLM/leo-mistral-hessianai-7b")
    print(source, len(group), tokens)
print(f"{30 * '-'}Summary{30 * '-'}")
summary_df = summary_df.groupby(["source"])
for source, group in summary_df:
    tokens = count_tokens(group["text"], None, "LeoLM/leo-mistral-hessianai-7b")
    print(source, len(group), tokens)
