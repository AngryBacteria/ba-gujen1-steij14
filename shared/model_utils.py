from enum import Enum

from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class ChatTemplate(Enum):
    """
    Enum class to store different chat templates
    """

    # ChatML instruction format. Meant for a chat format with alternating user and assistant messages
    CHATML = {
        "template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|im_end|>",
    }
    # Alpaca instruction format. Meant for only one instruction and one response, not a chat format like chatml
    ALPACA = {
        "template": "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{{ bos_token + system_message }}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '### Anweisung:\n' + message['content'].strip() + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ '### Antwort:\n' + message['content'].strip() + eos_token + '\n\n' }}{% endif %}{% if loop.last and message['role'] == 'user' and add_generation_prompt %}{{ '### Antwort:\n' }}{% endif %}{% endfor %}",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "</s>",
    }
    # Mistral instruction format. Meant for only one instruction and one response, not a chat format like chatml
    MISTRAL = {
        "template": "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{{ bos_token }}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + content.strip() + eos_token }}{% endif %}{% endfor %}",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "</s>",
    }


def load_from_jinja(file_name="template"):
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
    tokenizer_name="LeoLM/leo-mistral-hessianai-7b", template=ChatTemplate.ALPACA
):
    """
    Helper function to load a tokenizer with a specific chat template and special tokens
    :param tokenizer_name: The name of the tokenizer to load
    :param template: The chat template to use
    """
    # load the tokenizer
    if "mistral" in tokenizer_name.lower():
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
