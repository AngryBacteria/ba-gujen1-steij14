import torch
from transformers import AutoModelForCausalLM
from shared.model_utils import get_tokenizer_with_template


def upload_to_huggingface(
    account_name: str,
    repo_name: str,
    local_model_folder="mistral_instruction_low_precision",
):
    """
    Uploads a model and tokenizer to Huggingface.
    """
    upload_model(account_name, repo_name, local_model_folder)
    upload_tokenizer(account_name, repo_name, local_model_folder)


def upload_model(
    account_name: str,
    repo_name: str,
    local_model_folder="mistral_instruction_low_precision",
):
    """
    Uploads a model to Huggingface.
    """
    model = AutoModelForCausalLM.from_pretrained(
        local_model_folder, torch_dtype=torch.bfloat16
    )
    model.push_to_hub(f"{account_name}/{repo_name}", private=True)


def upload_tokenizer(
    account_name: str,
    repo_name: str,
    local_model_folder="mistral_instruction_low_precision",
):
    """
    Uploads a tokenizer to Huggingface.
    """
    tokenizer = get_tokenizer_with_template(tokenizer_name=local_model_folder)
    tokenizer.push_to_hub(f"{account_name}/{repo_name}", private=True)
