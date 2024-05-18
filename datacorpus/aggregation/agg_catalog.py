import re

from datacorpus.aggregation.prompts import (
    SYSTEM_PROMPT_CODE,
    ATC_INSTRUCTION,
    ICD10GM_INSTRUCTION,
    OPS_INSTRUCTION,
)
from shared.logger import logger
from shared.mongodb import get_collection


def aggregate_catalog_prompts():
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
            code_instruction_str = ATC_INSTRUCTION.replace(
                "<<ENTITY>>", doc["title"]
            ).strip()
        elif doc["source"] == "icd10gm":
            code_instruction_str = ICD10GM_INSTRUCTION.replace(
                "<<ENTITY>>", doc["title"]
            ).strip()
        elif doc["source"] == "ops":
            code_instruction_str = OPS_INSTRUCTION.replace(
                "<<ENTITY>>", doc["title"]
            ).strip()
        else:
            raise ValueError(f"Unknown source: {doc['source']}")

        prompts.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_CODE},
                    {"role": "user", "content": code_instruction_str},
                    {"role": "assistant", "content": doc["code"]},
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


aggregate_catalog_prompts()
