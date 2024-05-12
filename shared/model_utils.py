import re
from enum import Enum

import torch
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
)


# TEMPLATE
class ChatTemplate(Enum):
    """
    Enum class to store different chat templates
    """

    # Alpaca instruction format. Meant for only one instruction and one response. Padding tokens for Mistral
    ALPACA_MISTRAL = {
        "template": "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{{ bos_token + system_message }}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '### Anweisung:\n' + message['content'].strip() + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ '### Antwort:\n' + message['content'].strip() + eos_token + '\n\n' }}{% endif %}{% if loop.last and message['role'] == 'user' and add_generation_prompt %}{{ '### Antwort:\n' }}{% endif %}{% endfor %}",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "</s>",
        "generation_start": "### Antwort:",
    }
    # Alpaca instruction format. Meant for only one instruction and one response. Padding tokens for Llama3
    ALPACA_LLAMA3 = {
        "template": "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{{ bos_token + system_message }}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '### Anweisung:\n' + message['content'].strip() + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ '### Antwort:\n' + message['content'].strip() + eos_token + '\n\n' }}{% endif %}{% if loop.last and message['role'] == 'user' and add_generation_prompt %}{{ '### Antwort:\n' }}{% endif %}{% endfor %}",
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|end_of_text|>",
        "pad_token": "<|end_of_text|>",
        "generation_start": "### Antwort:",
    }


def load_template_from_jinja(file_name="template"):
    """
    Loads a chat template from a jinja file and cleans it
    :param file_name: The name of the jinja file to load
    """
    chat_template = open(f"{file_name}.jinja").read()
    chat_template = chat_template.replace("    ", "").replace("\n", "")
    print(chat_template)
    print(f'"template": "{chat_template}",')
    return chat_template


def test_chat_template(template: ChatTemplate, add_second_conversation=False):
    """
    Test function to apply a chat template to a list of messages and print the output
    :param template: The chat template to use
    :param add_second_conversation: If True, the function will add a second user input to the messages
    """
    test = patch_tokenizer_with_template(template=template)

    messages = [
        {"role": "system", "content": "This is a system prompt."},
        {"role": "user", "content": "This is the first user input."},
        {"role": "assistant", "content": "This is the first assistant response."},
    ]
    if add_second_conversation:
        messages.append(
            {"role": "user", "content": "This is the second user input."},
        )
        output = test.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        output = test.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    print(output)
    return output


# MODEL AND TOKENIZER
class ModelPrecision(Enum):
    """
    Enum class to store different model precisions
    """

    FOUR_BIT = 4
    EIGHT_BIT = 8
    SIXTEEN_BIT = 16


def patch_tokenizer_with_template(
    tokenizer_name="LeoLM/leo-mistral-hessianai-7b",
    template=ChatTemplate.ALPACA_MISTRAL,
):
    """
    Helper function to load a tokenizer with a specific chat template and special tokens
    :param tokenizer_name: The name of the tokenizer to load
    :param template: The chat template to use
    """
    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        padding_side="left",
    )
    # add special tokens
    tokenizer.eos_token = template.value["eos_token"]
    tokenizer.pad_token = template.value["pad_token"]
    tokenizer.bos_token = template.value["bos_token"]
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                template.value["eos_token"],
                template.value["bos_token"],
            ]
        }
    )
    # load the chat template
    tokenizer.chat_template = template.value["template"]
    return tokenizer


def patch_model_with_tokenizer(model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    """
    Helper function to patch a model with a new tokenizer. Adds special tokens and resizes the embedding layer
    :param model: The model to patch
    :param tokenizer: The tokenizer to use
    """
    # resize embedding layer<
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    # Update the model config to use the new eos & bos tokens
    if getattr(model, "config", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
    # Update the generation config to use the new eos & bos token
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model


def load_model_and_tokenizer(
    model_name: str,
    precision: ModelPrecision,
    patch_model=False,
    patch_tokenizer=False,
    template=ChatTemplate.ALPACA_MISTRAL,
):
    """
    Helper function to load a model and tokenizer with a specific precision and chat template.
    Optionally patches the model and tokenizer with the template (recommended if not pre-configured)
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
    )
    if precision == ModelPrecision.FOUR_BIT:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            load_in_4bit=True,
        )
    elif precision == ModelPrecision.EIGHT_BIT:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
        )
    elif precision == ModelPrecision.SIXTEEN_BIT:
        model = AutoModelForCausalLM.from_pretrained(
            "BachelorThesis/Mistral_V03_BRONCO_CARDIO",
            torch_dtype=torch.bfloat16,
        )
        model.to("cuda:0")
    else:
        raise ValueError("Precision has to be 4, 8 or 16")

    if patch_model:
        model = patch_model_with_tokenizer(model, tokenizer)
    if patch_tokenizer:
        tokenizer = patch_tokenizer_with_template(template=template)

    return tokenizer, model


def upload_model(
    account_name: str,
    repo_name: str,
    local_model_folder="mistral_instruction_low_precision",
    patch=True
):
    """
    Uploads a model to Huggingface.
    """
    if patch:
        tokenizer = patch_tokenizer_with_template(tokenizer_name=local_model_folder)
        model = AutoModelForCausalLM.from_pretrained(
            local_model_folder, torch_dtype=torch.bfloat16
        )
        model = patch_model_with_tokenizer(model, tokenizer)
        model.push_to_hub(f"{account_name}/{repo_name}", private=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            local_model_folder
        )
        model.push_to_hub(f"{account_name}/{repo_name}", private=True)


def upload_tokenizer(
    account_name: str,
    repo_name: str,
    local_model_folder="mistral_instruction_low_precision",
    patch=True
):
    """
    Uploads a tokenizer to Huggingface.
    """
    if patch:
        tokenizer = patch_tokenizer_with_template(tokenizer_name=local_model_folder)
        tokenizer.push_to_hub(f"{account_name}/{repo_name}", private=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(local_model_folder)
        tokenizer.push_to_hub(f"{account_name}/{repo_name}", private=True)


# GENERATION
def get_model_output_only(full_output: str, template: ChatTemplate) -> str | None:
    """
    Parses the model output (everything after the instruction) from the whole generated text.
    If the output could not be found return None
    """
    if (
        template == ChatTemplate.ALPACA_MISTRAL
        or template == ChatTemplate.ALPACA_LLAMA3
    ):
        parsed = full_output.split(template.value["generation_start"])
        if len(parsed) > 1:
            return parsed[1].strip()
        else:
            return None


def get_extractions_only(string_input: str):
    """
    Get all extractions from a string input. The string has to be in the typical form of a prompt output:
    extraction1 [attribute1|attribute2] | extraction2 [attribute3|attribute4] | ...
    """
    string_input = string_input.strip().lower()
    string_input = remove_brackets(string_input)
    extractions = string_input.split("|")
    extractions = [extraction.strip() for extraction in extractions]
    extractions = list(set(extractions))

    return extractions


def remove_brackets(string_input: str):
    """
    Remove substrings enclosed in brackets from a string.
    """
    pattern = r"\[[^]]*\]"
    cleaned_string = re.sub(pattern, "", string_input)
    cleaned_string = cleaned_string.strip()
    return cleaned_string


def get_extractions_with_attributes(string_input: str):
    # TODO: return the extractiona and attributes as list, not just a string like here :(
    string_input = string_input.strip().lower()
    extractions = string_input.split("|")
    extractions = [extraction.strip() for extraction in extractions]
    extractions = list(set(extractions))

    return extractions


def test_generation(
    messages=None,
    model_name="LeoLM/leo-mistral-hessianai-7b",
    precision=ModelPrecision.FOUR_BIT,
):
    """Function to test if the inference of the model works on gpu or not"""
    if messages is None:
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
    tokenizer, model = load_model_and_tokenizer(model_name, precision)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    outputs = model.generate(**inputs, max_new_tokens=1000)
    model_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(model_output)


def count_tokens(
    data: list[str],
    tokenizer_instance: PreTrainedTokenizer = None,
    tokenizer_name: str = None,
) -> int:
    """Count the tokens of a dataset with a tokenizer. Either pass a tokenizer or a tokenizer name."""
    if tokenizer_instance is not None:
        tokens = 0
        for text in data:
            tokens += len(tokenizer_instance(text)["input_ids"])
        return tokens

    if tokenizer_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True,
            add_eos_token=False,
            add_bos_token=False,
        )
        tokens = 0
        for text in data:
            tokens += len(tokenizer(text)["input_ids"])
        return tokens

    else:
        raise ValueError("Either pass a tokenizer or a tokenizer name.")
