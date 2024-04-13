import tiktoken


def count_tokens(text: str):
    encoder = tiktoken.encoding_for_model("gpt-4")
    return len(encoder.encode(text))


def count_tokens_db(data: list[str]):
    total_tokens = 0
    for document in data:
        total_tokens += count_tokens(document)
    return total_tokens
