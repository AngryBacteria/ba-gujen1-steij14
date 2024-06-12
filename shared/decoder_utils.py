import os

from shared.prompt_utils import get_extraction_messages, AttributeFormat, EntityType

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import time
from shared.gpu_utils import get_cuda_memory_usage
from shared.logger import logger
import re
from enum import Enum

import torch
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedTokenizerFast,
)


# TEMPLATE
class ChatTemplate(Enum):
    """
    Enum class to store different chat templates. Includes the template itself, the bos, eos and pad tokens and the
    generation start token. With the generation start token the answer of the llm can be easily extracted from the rest.
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
    # Alpaca instruction format. Meant for only one instruction and one response. Padding tokens for Gemma
    ALPACA_GEMMA = {
        "template": "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{{ bos_token + system_message }}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '### Anweisung:\n' + message['content'].strip() + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ '### Antwort:\n' + message['content'].strip() + eos_token + '\n\n' }}{% endif %}{% if loop.last and message['role'] == 'user' and add_generation_prompt %}{{ '### Antwort:\n' }}{% endif %}{% endfor %}",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "generation_start": "### Antwort:",
    }


class GenDevice(Enum):
    """Enum to represent device types that can be used to generate text with an LLM"""

    CPU = "cpu"
    MPS = "mps"
    CUDA_0 = "cuda:0"
    CUDA_1 = "cuda:1"
    CUDA_2 = "cuda:2"
    CUDA_3 = "cuda:3"


class ModelPrecision(Enum):
    """
    Enum class that represents the possible model precisions
    """

    FOUR_BIT = 4
    EIGHT_BIT = 8
    SIXTEEN_BIT = 16
    THIRTY_TWO_BIT = 32


# TODO https://huggingface.co/docs/transformers/perf_infer_cpu
CURRENT_DEFAULT_MODEL = r"S:\documents\onedrive_bfh\OneDrive - Berner Fachhochschule\Dokumente\UNI\Bachelorarbeit\Training\Modelle\Gemma-2b_V03_BRONCO_CARDIO_SUMMARY_CATALOG"
CURRENT_DEFAULT_TEMPLATE = ChatTemplate.ALPACA_GEMMA
CURRENT_DEFAULT_PRECISION = ModelPrecision.SIXTEEN_BIT


def load_template_from_jinja(file_name="template"):
    """
    Loads a chat template from a jinja file and cleans it.
    More templates can be found here https://github.com/chujiezheng/chat_templates
    :param file_name: The name of the jinja file to load
    """
    chat_template = open(f"{file_name}.jinja").read()
    chat_template = chat_template.replace("    ", "").replace("\n", "")
    print(chat_template)
    print(f'"template": "{chat_template}",')
    return chat_template


# MODEL AND TOKENIZER
def load_tokenizer_with_template(
    tokenizer_name=CURRENT_DEFAULT_MODEL,
    template=CURRENT_DEFAULT_TEMPLATE,
):
    """
    Helper function to load a tokenizer with a specific chat template and special tokens.
    Patches both model and tokenizer to use those new tokens.
    :param tokenizer_name: The name of the tokenizer to load
    :param template: The chat template to use
    """
    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
    )
    # add special tokens
    tokenizer.eos_token = template.value["eos_token"]
    if tokenizer.pad_token is None:
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
    logger.debug(f"Loaded tokenizer {tokenizer_name} with template {template.name}")
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

    logger.debug(
        f"Patched model {model.name_or_path} with tokenizer {tokenizer.name_or_path}"
    )
    return model


def get_best_device() -> GenDevice:
    """
    Function to get the best device to use for generation.
    :return: The best device to use for generation.
    """
    if torch.cuda.is_available():
        return GenDevice.CUDA_0
    elif torch.backends.mps.is_available():
        return GenDevice.MPS
    else:
        return GenDevice.CPU


def load_model_and_tokenizer(
    model_name=CURRENT_DEFAULT_MODEL,
    precision=CURRENT_DEFAULT_PRECISION,
    patch_model=False,
    patch_tokenizer=False,
    use_flash_attn=False,
    device=get_best_device(),
    template=CURRENT_DEFAULT_TEMPLATE,
    empty_cache=False,
):
    """
    Helper function to load a model and tokenizer with a specific precision and chat template.
    Optionally patches the model and tokenizer with the template (recommended if not pre-configured)
    :param empty_cache: If the cuda cache should be cleared before loading the model
    :param model_name: The path of the model to load. Can be a huggingface repository name or a local directory.
    For example: "BachelorThesis/LLama3_V02_BRONCO_CARDIO_SUMMARY_CATALOG"
    :param precision: The precision to load the model in. Can be 4, 8, 16 or 32 bit. Recommended is 16 bit
    for the best quality.
    :param patch_model: If the model should be patched with the tokenizer vocab. Not required if the model is one
    of ours
    :param patch_tokenizer: If the tokenizer should be patched with the template. Not required if the tokenizer is
    one of ours.
    :param use_flash_attn: If the more efficient flash attention should be used. Always recommended. But it needs to
    be installed and a CUDA device.
    :param device: The device to use. If you have a CUDA device, it is recommended to use it. Else mps for Apple
    Silicon devices and cpu for all other devices (will be terribly slow).
    :param template: The chat/instruction template to use.
    :return: Model and Tokenizer
    """
    if empty_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
    )

    # if cuda should be used or not
    should_use_bf16 = (
        device is not GenDevice.CPU
        and device is not GenDevice.MPS
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )

    # get correct bnb compute dtype
    if should_use_bf16:
        bnb_compute_dtype = torch.bfloat16
    else:
        bnb_compute_dtype = torch.float16

    # set correct config
    attn_implementation = "flash_attention_2" if use_flash_attn else "sdpa"
    if precision == ModelPrecision.FOUR_BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=bnb_compute_dtype,
        )
        model_config = {"quantization_config": bnb_config}
    elif precision == ModelPrecision.EIGHT_BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model_config = {"quantization_config": bnb_config}
    elif precision == ModelPrecision.SIXTEEN_BIT:
        if should_use_bf16:
            model_config = {"torch_dtype": torch.bfloat16}
        elif device != GenDevice.CPU:
            model_config = {"torch_dtype": torch.float16}
        else:
            logger.warning(
                "Using 16 bit precision on CPU is not recommended. Switching to 32 bit."
            )
            model_config = {"torch_dtype": torch.float32}
    elif precision == ModelPrecision.THIRTY_TWO_BIT:
        model_config = {"torch_dtype": torch.float32}
    else:
        raise ValueError("Precision has to be 4, 8, 16 or 32")
    model_config.update(
        {
            "attn_implementation": attn_implementation,
        }
    )

    # "to" function is not supported with 4/8 bit models
    if precision == ModelPrecision.FOUR_BIT or precision == ModelPrecision.EIGHT_BIT:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_config).to(
            device.value
        )

    if patch_tokenizer:
        tokenizer = load_tokenizer_with_template(
            tokenizer_name=model_name, template=template
        )
    if patch_model:
        model = patch_model_with_tokenizer(model, tokenizer)

    logger.debug(
        f"Loaded model/tokenizer {model_name} with precision {precision.name} and tokenization template {template.name}"
    )

    return tokenizer, model


def upload_model(
    account_name: str,
    repo_name: str,
    local_model_folder: str,
):
    """
    Uploads a model to Huggingface.
    :param account_name: The name of the account to upload to. Can also be an organization.
    :param repo_name: The name of the repository to upload to or create.
    :param local_model_folder: The name of the local model folder to upload
    """
    model = AutoModelForCausalLM.from_pretrained(local_model_folder)
    model.push_to_hub(f"{account_name}/{repo_name}", private=True)


def upload_tokenizer(
    account_name: str,
    repo_name: str,
    local_model_folder: str,
):
    """
    Uploads a tokenizer to Huggingface.
    :param account_name: The name of the account to upload to. Can also be an organization.
    :param repo_name: The name of the repository to upload to or create.
    :param local_model_folder: The name of the local tokenizer folder to upload
    :return:
    """
    tokenizer = AutoTokenizer.from_pretrained(local_model_folder)
    tokenizer.push_to_hub(f"{account_name}/{repo_name}", private=True)


# GENERATION
def generate_output(
    messages: str | list[dict],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    device=GenDevice.CUDA_0,
    max_new_tokens=2000,
) -> tuple[str, str]:
    start_time = time.perf_counter_ns()
    if isinstance(messages, str):
        inputs = tokenizer(messages, return_tensors="pt").to(device.value)
    else:
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer(inputs, return_tensors="pt").to(device.value)

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    model_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    model_output_raw = tokenizer.decode(outputs[0], skip_special_tokens=False)
    end_time = time.perf_counter_ns()
    # execution time in milliseconds
    execution_time = (end_time - start_time) / 10**6

    # if cuda available get gpu usage
    mem_info = ""
    if torch.cuda.is_available() and "cuda" in device.value:
        gpu = int(device.value[-1])
        allocated, capacity = get_cuda_memory_usage(gpu)
        mem_info = f"while using {allocated:.1f}GB of {capacity:.2f}GB GPU memory"

    logger.debug(
        f"Generated {len(outputs[0])} tokens with {model.name_or_path} in {execution_time}ms on {device.value} {mem_info}"
    )
    return model_output, model_output_raw


def get_model_output_only(
    full_output: str, template=CURRENT_DEFAULT_TEMPLATE, lower=False
) -> str | None:
    """
    Parses the model output (everything after the instruction) from the whole generated text.
    If the output could not be found return None
    :param lower: If the output should be converted to lowercase
    :param full_output: The full output that the model generated.
    :param template: The template that was used during generation.
    :return: Only the output of the model.
    """
    parsed = full_output.split(template.value["generation_start"])
    if len(parsed) > 1:
        if lower:
            return parsed[1].strip().lower()
        else:
            return parsed[1].strip()
    else:
        logger.warning(
            f"No model output found. The template was {template.name} in the model response: {full_output}"
        )
        return None


def get_extractions_without_attributes(string_input: str):
    """
    Get all extractions without attributes from a string input.
    The string has to be in the typical form of a prompt output:
    extraction1 [attribute1|attribute2] | extraction2 [attribute3|attribute4] | ...
    :param string_input: The string input to get the extractions from.
    """
    string_input = string_input.strip().lower()
    string_input = remove_brackets(string_input)
    extractions = string_input.split("|")
    extractions = [extraction.strip() for extraction in extractions]
    extractions = list(set(extractions))

    if len(extractions) == 0 and string_input != "Keine vorhanden":
        logger.warning(f"No extractions found in the string: {string_input}")

    return extractions


def remove_brackets(string_input: str):
    """
    Remove substrings enclosed in brackets from a string.
    :param string_input: The string to remove the brackets from.
    """
    pattern = r"\[[^]]*\]"
    cleaned_string = re.sub(pattern, "", string_input)
    cleaned_string = cleaned_string.strip()
    return cleaned_string


def get_extractions_with_attributes(string_input: str):
    """
    Get all extractions with attributes from a string input.
    The string has to be in the typical form of a prompt output:
    extraction1 [attribute1|attribute2] | extraction2 [attribute3|attribute4] | ...
    :param string_input: The string input to get the extractions from.
    """
    string_input = string_input.strip().lower()
    extractions = string_input.split("|")
    extractions = [extraction.strip() for extraction in extractions]
    extractions = list(set(extractions))

    if len(extractions) == 0 and string_input != "Keine vorhanden":
        logger.warning(f"No extractions found in the string: {string_input}")

    return extractions


def get_extractions_with_attributes_grouped(string_input: str):
    """
    Get all extractions with attributes from a string input.
    The string has to be in the typical form of a prompt output:
    extraction1 [attribute1|attribute2] | extraction2 [attribute3|attribute4] | ...
    :param string_input: The string input to get the extractions from.
    :return: Dict with extraction as key and attributes as value.
    """
    import re

    # Regular expression to match "extraction[attribute1|attribute2]"
    pattern = re.compile(r"([^\[\]]+)(?:\[([^\[\]]*)])?")

    # Split the string by "|" outside of brackets
    parts = pattern.findall(string_input)
    extraction_dict = {}
    for part in parts:
        extraction = part[0].replace("|", "").strip()
        attributes = part[1].strip()
        if attributes:
            # Split the attributes by "|" and strip spaces
            attribute_list = [attr.strip() for attr in attributes.split("|")]
        else:
            attribute_list = []

        # Add the extraction and its attributes to the dictionary
        attribute_list = [attr.replace("|", "").strip() for attr in attribute_list]

        # group if same extraction is found twice
        if extraction_dict.get(extraction):
            extraction_dict[extraction].extend(attribute_list)
        else:
            extraction_dict[extraction] = attribute_list

    if len(extraction_dict) == 0 and string_input != "Keine vorhanden":
        logger.warning(f"No extractions found in the string: {string_input}")

    return extraction_dict


def get_attributes_only(string_input: str):
    """
    Get all attributes from a string input. Should be in the form of:
    extraction1 [attribute1|attribute2] | extraction2 [attribute3|attribute4] | ...
    """
    pattern = re.compile(r"\[([^]]+)]")
    matches = pattern.findall(string_input)
    matches = [match.strip() for match in matches]

    attributes = []
    for match in matches:
        if "|" in match:
            attributes.extend(match.split("|"))
        else:
            attributes.append(match)
    attributes = [attribute.strip() for attribute in attributes]

    if len(attributes) == 0 and string_input != "Keine vorhanden":
        logger.warning(f"No attributes found in the string: {string_input}")

    return attributes


def test_generation(
    model_name=CURRENT_DEFAULT_MODEL,
    precision=CURRENT_DEFAULT_PRECISION,
    device=get_best_device(),
    template=CURRENT_DEFAULT_TEMPLATE,
):
    """Function to test if the inference of the model works on gpu or not"""
    messages1 = get_extraction_messages(
        "Der Patient hat Kopfschmerzen.", AttributeFormat.BRONCO, EntityType.DIAGNOSIS
    )
    messages2 = get_extraction_messages(
        "Die Patientin Elizabeth Brönnimann hatte sich heute in der Notfallaufnahme gemeldet weil sie Brustschmerzen hatte.",
        AttributeFormat.BRONCO,
        EntityType.DIAGNOSIS,
    )
    messages_concat = [messages1, messages2]

    tokenizer, model = load_model_and_tokenizer(
        model_name, precision, device=device, template=template
    )
    for messages in messages_concat:
        output, _ = generate_output(messages, model, tokenizer, device=device)
        print(output)
        print(30 * "-")


def count_tokens(
    data: list[str],
    tokenizer_instance: PreTrainedTokenizer = None,
    tokenizer_name=CURRENT_DEFAULT_MODEL,
    template=CURRENT_DEFAULT_TEMPLATE,
) -> int:
    """
    Count the tokens of a dataset with a tokenizer. Either pass a tokenizer or a tokenizer name.
    :param template: The template to use for the tokenizer.
    :param data: List of strings that will get counted.
    :param tokenizer_instance: The tokenizer to use. If nothing is passed, the tokenizer_name will be used.
    :param tokenizer_name: The name of the tokenizer to use. If a tokenizer is passed, this will be ignored.
    :return: Number of tokens in the dataset.
    """
    if tokenizer_instance is not None:
        tokens = 0
        for text in data:
            tokens += len(tokenizer_instance(text)["input_ids"])
        return tokens

    elif tokenizer_name is not None:
        tokenizer = load_tokenizer_with_template(
            tokenizer_name=tokenizer_name, template=template
        )
        tokens = 0
        for text in data:
            tokens += len(tokenizer(text)["input_ids"])
        return tokens

    else:
        raise ValueError("Either pass a tokenizer or a tokenizer name.")


if __name__ == "__main__":
    test_generation()
