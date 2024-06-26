import re

from shared.logger import logger
from shared.mongodb import get_collection
from shared.prompt_utils import (
    get_catalog_messages,
    CatalogType,
    TaskType,
)


def aggregate_catalog_prompts():
    """
    Aggregate prompts for the catalog task from the atc, icd10gm and ops collections.
    :return: List of prompts for the catalog task
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
        if doc["source"] == "atc":
            messages = get_catalog_messages(doc["title"], CatalogType.ATC)
            output = f"Der ATC Code für {doc['title']} ist {doc['code']}"
            prompt_type = "MEDICATION"
        elif doc["source"] == "icd10gm":
            messages = get_catalog_messages(doc["title"], CatalogType.ICD)
            output = f"Der ICD10-GM Code für {doc['title']} ist {doc['code']}"
            prompt_type = "DIAGNOSIS"
        elif doc["source"] == "ops":
            messages = get_catalog_messages(doc["title"], CatalogType.OPS)
            output = f"Der OPS Code für {doc['title']} ist {doc['code']}"
            prompt_type = "TREATMENT"
        else:
            raise ValueError(f"Unknown source: {doc['source']}")

        messages.append({"role": "assistant", "content": output})
        prompts.append(
            {
                "messages": messages,
                "type": prompt_type,
                "task": TaskType.CATALOG.value,
                "source": doc["source"],
                "na_prompt": False,
                "context": doc["title"],
                "context_entity": "",
                "output": output,
            }
        )

    logger.debug(f"Aggregated {len(prompts)} prompts for code task from catalogs.")
    return prompts
