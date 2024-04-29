import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

login()


def upload_to_huggingface(
    repo_name: str, local_model_folder="mistral_instruction_low_precision"
):
    tokenizer = AutoTokenizer.from_pretrained(
        "LeoLM/leo-mistral-hessianai-7b", use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        local_model_folder, torch_dtype=torch.bfloat16
    )

    model.push_to_hub(f"BachelorThesis/{repo_name}", private=True)
    tokenizer.push_to_hub(f"BachelorThesis/{repo_name}", private=True)


upload_to_huggingface("mistral_v02")
