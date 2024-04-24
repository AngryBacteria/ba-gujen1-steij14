from research.datacorpus.creation.utils.utils_mongodb import get_collection

jsyncc = get_collection("corpus", "jsyncc")


def get_jsyncc_pretrain_texts():
    """
    Get all texts from jsyncc
    :return: List of texts from jsyncc
    """
    jsyncc_texts = [
        {
            "text": doc["heading"] + "\n" + doc["text"],
            "task": "pretrain",
            "source": "jsyncc",
        }
        for doc in jsyncc.find({}, {"text": 1, "heading": 1})
    ]
    return jsyncc_texts
