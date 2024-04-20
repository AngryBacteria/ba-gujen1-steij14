# TODO: aggregation of all corpus datasets
from datasets import load_dataset

test_dataset = load_dataset("imdb", split="test[:5%]")
train_dataset = load_dataset("imdb", split="train[:40%]")

print(train_dataset)
print(test_dataset)