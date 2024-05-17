from datacorpus.aggregation.prompts import (
    SYSTEM_PROMPT_SUMMARIZATION,
    SUMMARY_INSTRUCTION,
)
from shared.logger import logger
from shared.mongodb import get_collection


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
        summary_instruction = SUMMARY_INSTRUCTION.replace("<<CONTEXT>>", doc["origin"])
        prompts.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_SUMMARIZATION},
                    {"role": "user", "content": summary_instruction.strip()},
                    {"role": "assistant", "content": doc["summary"]},
                ],
                "type": "",
                "task": "summary",
                "source": doc["source"],
                "na_prompt": False,
                "context": doc["origin"],
                "context_entity": "",
                "output": doc["summary"],
            }
        )

    logger.debug(f"Aggregated {len(prompts)} summary prompts from the synthetic corpus")
    return prompts
