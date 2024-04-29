from research.datacorpus.creation.utils.utils_mongodb import get_collection

clef = get_collection("corpus", "clef2019")


def aggregate_clef_pretrain_texts():
    """
    Get all pretrain texts from CLEF 2019
    :return: List of pretrain texts from CLEF 2019
    """
    clef_texts = [
        {"text": doc["text"], "task": "pretrain", "source": "clef2019"}
        for doc in clef.find({}, {"text": 1})
    ]
    return clef_texts
