import os

import setproctitle
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
setproctitle.setproctitle("gujen1 - bachelorthesis")

from datasets import load_dataset

from shared.model_utils import get_tokenizer_with_template, patch_model
from transformers import AutoModelForCausalLM, pipeline


def test_manual():
    tokenizer = get_tokenizer_with_template(
        tokenizer_name="LeoLM/leo-mistral-hessianai-7b"
    )
    model = AutoModelForCausalLM.from_pretrained(
        "mistral_instruction_low_precision",
        torch_dtype=torch.bfloat16,
        # load_in_8bit=True,
        # load_in_4bit=True,
    )
    model = patch_model(model, tokenizer)
    model.to("cuda:0")

    messages = [
        {
            "role": "system",
            "content": "Du bist ein fortgeschrittener Algorithmus, der darauf spezialisiert ist, aus medizinischen Texten strukturierte Informationen wie Medikamente, Symptome oder Diagnosen und klinische Prozeduren zu extrahieren.",
        },
        {
            "role": "user",
            "content": 'Extrahiere alle Diagnosen und Symptome aus dem folgenden Text. Falls keine im Text vorkommen, schreibe "Keine vorhanden":\n\nIn einem Roentgen Thorax zeigten sich prominente zentrale Lungengefaesszeichnung mit basoapikaler Umverteilung sowie angedeutete Kerley-B-Linien, vereinbar mit einer chronischen pulmonalvenoesen Stauungskomponente bei Hypervolaemie.',
        },
    ]
    only_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(only_prompt, return_tensors="pt").to("cuda:0")
    outputs = model.generate(**inputs, max_length=512)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("-----------------------------------------")


def test_with_file():
    tokenizer = get_tokenizer_with_template(
        tokenizer_name="LeoLM/leo-mistral-hessianai-7b"
    )
    model = AutoModelForCausalLM.from_pretrained(
        "mistral_instruction_low_precision",
        torch_dtype=torch.bfloat16,
        # load_in_8bit=True,
        # load_in_4bit=True,
    )
    model = patch_model(model, tokenizer).to("cuda:0")

    _dataset = load_dataset("json", data_files={"data": "prompts.jsonl"})[
        "data"
    ].train_test_split(test_size=0.1, shuffle=True, seed=42)
    data = _dataset["test"]
    for i, example in enumerate(data):
        only_prompt = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(only_prompt, return_tensors="pt").to("cuda:0")
        outputs = model.generate(**inputs, max_length=512)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        print("-----------------------------------------")
