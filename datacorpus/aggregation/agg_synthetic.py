from shared.logger import logger
from shared.mongodb import get_collection
from shared.prompt_utils import (
    get_summarization_messages,
    TaskType,
)


# Aggregation of the synthetic data in the mongodb database. So far only the task summarization is supported in the
# as no other synthetic data is available yet.


def aggregate_synthetic_prompts():
    """
    Aggregate prompts for the synthetic dataset. Right now only summarization prompts are synthesized.
    """
    synthetic_collection = get_collection("corpus", "synthetic")
    documents = synthetic_collection.find({})

    prompts = []
    for doc in documents:
        messages = get_summarization_messages(doc["origin"])
        messages.append({"role": "assistant", "content": doc["summary"]})
        prompts.append(
            {
                "messages": messages,
                "type": "",
                "task": TaskType.SUMMARIZATION.value,
                "source": doc["source"],
                "na_prompt": False,
                "context": doc["origin"],
                "context_entity": "",
                "output": doc["summary"],
            }
        )

    logger.debug(f"Aggregated {len(prompts)} summary prompts from the synthetic corpus")
    return prompts
