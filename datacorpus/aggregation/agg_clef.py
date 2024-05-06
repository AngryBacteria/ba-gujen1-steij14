from datacorpus.utils.mongodb import get_collection

# Agrgegation of CLEF 2019 data into unstructured text. Can be used for masked- or causal language modelling.

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
