import tiktoken


def count_tokens(text: str, model="gpt-4") -> int:
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(text))


def count_tokens_list(data: list[str]) -> int:
    total_tokens = 0
    for document in data:
        total_tokens += count_tokens(document)
    return total_tokens
