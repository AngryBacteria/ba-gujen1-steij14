# TODO: aggregation of ggponc2 corpus
from research.datacorpus.aggregation.prompts import (
    MEDICATION_PROMPT,
    TREATMENT_PROMPT,
    DIAGNOSIS_PROMPT,
)
from research.datacorpus.utils.utils_mongodb import get_collection

ggonc_collection = get_collection("corpus", "ggponc_short")


def get_ggponc_prompts(
    document_type: str, simple_prompt: str, ignore_short: int
) -> list[str]:
    simple_prompts = []
    documents = ggonc_collection.find({"annotations.type": document_type})
    for document in documents:
        for anno in document["annotations"]:
            if anno["type"] != document_type:
                continue
            if ignore_short > 0 and len(anno["origin"]) < ignore_short:
                continue

            texts = "|".join(anno["text"])
            simple_prompt_str = simple_prompt.replace("<<CONTEXT>>", anno["origin"])
            simple_prompt_str = simple_prompt_str.replace("<<OUTPUT>>", texts)
            simple_prompts.append(simple_prompt_str)

    return simple_prompts


def get_all_simple_ggponc_prompts(ignore_short: int) -> list[str]:
    output = []
    medication_prompts = get_ggponc_prompts(
        "MEDICATION", MEDICATION_PROMPT, ignore_short
    )
    diagnosis_prompts = get_ggponc_prompts("DIAGNOSIS", DIAGNOSIS_PROMPT, ignore_short)
    treatment_prompts = get_ggponc_prompts("TREATMENT", TREATMENT_PROMPT, ignore_short)

    output.extend(medication_prompts)
    output.extend(diagnosis_prompts)
    output.extend(treatment_prompts)
    return output
