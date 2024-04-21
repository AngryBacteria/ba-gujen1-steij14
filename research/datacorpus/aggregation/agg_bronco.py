from research.datacorpus.aggregation.aggregate import (
    MEDICATION_PROMPT,
    DIAGNOSIS_PROMPT,
    TREATMENT_PROMPT,
    MEDICATION_NORMALIZATION_PROMPT,
    DIAGNOSIS_NORMALIZATION_PROMPT,
    TREATMENT_NORMALIZATION_PROMPT,
)
from research.datacorpus.utils.utils_mongodb import get_collection

bronco_collection = get_collection("corpus", "bronco")


def get_bronco_prompts(document_type, simple_prompt, normalization_prompt):
    simple_prompts = []
    normalization_prompts = []
    documents = bronco_collection.find({"type": document_type})
    for document in documents:
        texts = "|".join(document["text"])
        simple_prompt_str = simple_prompt.replace("<<CONTEXT>>", document["origin"])
        simple_prompt_str = simple_prompt_str.replace("<<OUTPUT>>", texts)
        simple_prompts.append(simple_prompt_str)

        for index, text in enumerate(document["text"]):
            if document["normalizations"][index][0] is None:
                continue
            else:
                normalization = document["normalizations"][index][0]
                norm_prompt_str = normalization_prompt.replace("<<CONTEXT>>", text)
                norm_prompt_str = norm_prompt_str.replace(
                    "<<OUTPUT>>", normalization["normalization"].split(":")[1]
                )
                normalization_prompts.append(norm_prompt_str)

    return simple_prompts, normalization_prompts


def get_all_simple_bronco_prompts():
    output = []
    medication_prompts, medication_norm_prompts = get_bronco_prompts(
        "MEDICATION", MEDICATION_PROMPT, MEDICATION_NORMALIZATION_PROMPT
    )
    diagnosis_prompts, diagnosis_norm_prompts = get_bronco_prompts(
        "DIAGNOSIS", DIAGNOSIS_PROMPT, DIAGNOSIS_NORMALIZATION_PROMPT
    )
    treatment_prompts, treatment_norm_prompts = get_bronco_prompts(
        "TREATMENT", TREATMENT_PROMPT, TREATMENT_NORMALIZATION_PROMPT
    )

    output.extend(medication_prompts)
    output.extend(diagnosis_prompts)
    output.extend(treatment_prompts)
    return output


def get_all_bronco_normalization_prompts():
    medication_prompts, medication_norm_prompts = get_bronco_prompts(
        "MEDICATION", MEDICATION_PROMPT, MEDICATION_NORMALIZATION_PROMPT
    )
    diagnosis_prompts, diagnosis_norm_prompts = get_bronco_prompts(
        "DIAGNOSIS", DIAGNOSIS_PROMPT, DIAGNOSIS_NORMALIZATION_PROMPT
    )
    treatment_prompts, treatment_norm_prompts = get_bronco_prompts(
        "TREATMENT", TREATMENT_PROMPT, TREATMENT_NORMALIZATION_PROMPT
    )

    output = []
    output.extend(medication_norm_prompts)
    output.extend(diagnosis_norm_prompts)
    output.extend(treatment_norm_prompts)
    return output
