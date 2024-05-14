from datasets import load_dataset, concatenate_datasets

# load the dataset
data = load_dataset("json", data_files={"data": "prompts.jsonl"})[
    "data"
].train_test_split(test_size=0.1, shuffle=True, seed=42)

# combine test and train
train_data = data["train"]
test_data = data["test"]
concatenate_datasets([train_data, test_data])

count = 0
for data in data["train"]:
    if data["type"] == "MEDICATION" and data["task"] == "extraction":
        count += 1

print(count)