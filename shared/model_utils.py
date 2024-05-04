from enum import Enum

from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class ChatTemplate(Enum):
    CHATML = {
        "template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|im_end|>",
    }


def load_from_jinja(file_name="template"):
    chat_template = open(f"{file_name}.jinja").read()
    chat_template = chat_template.replace("    ", "").replace("\n", "")
    print(chat_template)
    return chat_template


def test_chat_template(template: ChatTemplate, add_second_conversation=False):
    test = get_tokenizer_with_template(template=template)

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


def get_tokenizer_with_template(
    tokenizer_name="LeoLM/leo-mistral-hessianai-7b", template=ChatTemplate.CHATML
):
    """
    Helper function to load a tokenizer with a specific chat template and special tokens
    """
    # load the tokenizer
    if "mistral" in tokenizer_name:
        padding_side = "left"
    else:
        padding_side = "right"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        padding_side=padding_side,
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


def patch_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    """
    Patches the model and tokenizer to use the same special tokens and generation config
    :param model: Model that should be patched with the config of the tokenizer
    :param tokenizer: The tokenizer that should be patched with the config of the model
    """
    # resize embedding layer
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
