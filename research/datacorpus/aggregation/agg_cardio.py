# TODO: aggregation of cardio
from research.datacorpus.creation.utils.utils_mongodb import get_collection

cardio = get_collection("corpus", "cardio")
cardio_heldout = get_collection("corpus", "cardio_heldout")


def get_cardio_pretrain_texts():
    """
    Get all cardio pretrain texts
    :return: List of pretrain texts from cardio and cardio_heldout
    """
    cardio_texts = [doc["full_text"] for doc in cardio.find({}, {"full_text": 1})]
    cardio_heldout_texts = [
        doc["full_text"] for doc in cardio_heldout.find({}, {"full_text": 1})
    ]
    cardio_texts = cardio_texts + cardio_heldout_texts
    return cardio_texts
