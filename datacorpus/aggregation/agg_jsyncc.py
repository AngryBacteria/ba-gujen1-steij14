from shared.mongodb import get_collection

# Agrgegation of jsyncc synthethic data into unstructured text. Can be used for masked- or causal language modelling.


def aggregate_jsyncc_pretrain_texts():
    """
    Get all texts from jsyncc
    :return: List of texts from jsyncc
    """
    jsyncc = get_collection("corpus", "jsyncc")
    jsyncc_texts = [
        {
            "text": doc["heading"] + "\n" + doc["text"],
            "task": "pretrain",
            "source": "jsyncc",
        }
        for doc in jsyncc.find({}, {"text": 1, "heading": 1})
    ]
    return jsyncc_texts
