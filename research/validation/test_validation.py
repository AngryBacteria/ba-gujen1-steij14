import os

import setproctitle
import torch
from datasets import load_dataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
setproctitle.setproctitle("gujen1 - bachelorthesis")

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "LeoLM/leo-mistral-hessianai-7b", use_fast=True
)
model = AutoModelForCausalLM.from_pretrained(
    "F:\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\Training\\Mistral_V02_BRONCO_CARDIO",
    torch_dtype=torch.bfloat16,
    # load_in_8bit=True,
    # load_in_4bit=True,
).to("cuda:0")

_dataset = load_dataset("json", data_files={"data": "prompts.jsonl"})[
    "data"
].train_test_split(test_size=0.1, shuffle=True, seed=42)
data = _dataset["test"]

for i, example in enumerate(data):
    lines = example["text"].splitlines()
    only_prompt = "\n".join(lines[:-1])
    inputs = tokenizer(only_prompt, return_tensors="pt").to("cuda:0")
    outputs = model.generate(**inputs, max_length=512)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("-----------------------------------------")
