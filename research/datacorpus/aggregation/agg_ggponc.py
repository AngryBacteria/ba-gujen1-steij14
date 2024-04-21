# TODO: aggregation of ggponc2 corpus
from research.datacorpus.aggregation.prompts import (
    MEDICATION_PROMPT,
    TREATMENT_PROMPT,
    DIAGNOSIS_PROMPT,
)
from research.datacorpus.utils.utils_mongodb import get_collection

bronco_collection = get_collection("corpus", "ggponc_short")


def get_ggponc_prompts(document_type, simple_prompt):
    simple_prompts = []
    documents = bronco_collection.find({"annotations.type": document_type})
    for document in documents:
        for anno in document["annotations"]:
            if anno["type"] != document_type:
                continue

            texts = "|".join(anno["text"])
            simple_prompt_str = simple_prompt.replace("<<CONTEXT>>", anno["origin"])
            simple_prompt_str = simple_prompt_str.replace("<<OUTPUT>>", texts)
            simple_prompts.append(simple_prompt_str)

    return simple_prompts


def get_all_simple_ggponc_prompts():
    output = []
    medication_prompts = get_ggponc_prompts("MEDICATION", MEDICATION_PROMPT)
    diagnosis_prompts = get_ggponc_prompts("DIAGNOSIS", DIAGNOSIS_PROMPT)
    treatment_prompts = get_ggponc_prompts("TREATMENT", TREATMENT_PROMPT)

    output.extend(medication_prompts)
    output.extend(diagnosis_prompts)
    output.extend(treatment_prompts)
    return output
