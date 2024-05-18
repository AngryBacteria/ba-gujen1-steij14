import re

from shared.logger import logger
from shared.mongodb import get_collection


def aggregate_catalog_prompts(detailed: bool):
    """
    :param detailed Controls whether the aggregation is detailed or not meaning how many level deep into the code
    systems prompts should be aggregated. For example detailed=False for OPS means only to aggregate codes up to 3
    digits, detailed=True means to aggregate all codes.
    """
    atc_collection = get_collection("catalog", "atc")
    icd10gm_collection = get_collection("catalog", "icd10gm")
    ops_collection = get_collection("catalog", "ops")

    # Retrieve all documents and add source attribute
    atc_docs = [{**doc, "source": "atc"} for doc in atc_collection.find({})]
    ic10gm_pattern = re.compile(r"^(?!.*\.).*$", re.IGNORECASE)
    icd10gm_docs = [
        {**doc, "source": "icd10gm"}
        for doc in icd10gm_collection.find({"code": {"$regex": ic10gm_pattern}})
    ]
    ops_pattern = re.compile(r"^(?!.*\.).*$", re.IGNORECASE)
    ops_docs = [
        {**doc, "source": "ops"}
        for doc in ops_collection.find({"code": {"$regex": ops_pattern}})
    ]

    # add docs together
    docs = list(atc_docs) + list(icd10gm_docs) + list(ops_docs)

    prompts = []

    for doc in docs:
        prompts.append(
            {
                # TODO: ADD prompts
                "messages": [
                    {"role": "system", "content": "SYSTEM_PROMPT"},
                    {"role": "user", "content": "extraction_instruction_str"},
                    {"role": "assistant", "content": "extraction_string"},
                ],
                "type": "",
                "task": "code",
                "source": doc["source"],
                "na_prompt": False,
                "context": doc["title"],
                "context_entity": doc["title"],
                "output": doc["code"],
            }
        )

    logger.debug(f"Aggregated {len(prompts)} prompts for code task from catalogs.")
    return prompts


aggregate_catalog_prompts(True)
