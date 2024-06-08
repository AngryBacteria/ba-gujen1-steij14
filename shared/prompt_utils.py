# Collection of system- and instruction prompts for the tasks summarization, extraction and normalization.
from enum import Enum

# SYSTEM PROMPTS
SYSTEM_PROMPT_EXTRACTION = "Du bist ein fortgeschrittener Algorithmus, der darauf spezialisiert ist, aus medizinischen Texten strukturierte Informationen wie Medikamente, Symptome oder Diagnosen und klinische Prozeduren zu extrahieren."
SYSTEM_PROMPT_NORMALIZATION = "Du bist ein fortgeschrittener Algorithmus, der darauf spezialisiert ist, medizinischen Entitäten wie Medikamenten, Symptomen oder Diagnosen und Prozeduren entsprechende Codierungen zuzuordnen."
SYSTEM_PROMPT_SUMMARIZATION = "Du bist ein fortgeschrittener Algorithmus, der darauf spezialisiert ist, aus umfangreichen Texten die Hauptpunkte zu extrahieren und sie in einer prägnanten, zusammengefassten Form darzustellen. Deine Aufgabe ist es, den Kerninhalt effektiv zu erfassen und dabei wichtige Details, Schlussfolgerungen und Argumentationsstränge beizubehalten."
SYSTEM_PROMPT_CATALOG = "Du bist ein fortgeschrittener Algorithmus, der darauf spezialisiert ist, verschiedenen medizinischen Entitäten systematisch spezifische Codierungen zuzuordnen."

# SUMMARY PROMPTS
SUMMARY_INSTRUCTION = """Bitte fasse den folgenden klinischen Text präzise zusammen. Stelle sicher, dass alle wichtigen klinischen Informationen erhalten bleiben:

<<CONTEXT>>"""

# CATALOG PROMPTS
ATC_INSTRUCTION = """Was ist der ATC Code für das Medikament „<<ENTITY>>“?"""
ICD10GM_INSTRUCTION = (
    """Was ist der ICD10-GM Code für die Diagnose oder das Symptom „<<ENTITY>>“?"""
)
OPS_INSTRUCTION = """Was ist der OPS Code für die Prozedur „<<ENTITY>>“?"""

# MEDICATION PROMPTS
MEDICATION_INSTRUCTION_GENERIC = """Extrahiere alle Medikamente aus dem folgenden Text. Falls keine im Text vorkommen, schreibe "Keine vorhanden":

<<CONTEXT>>"""
MEDICATION_INSTRUCTION_BRONCO = """Extrahiere alle Medikamente aus dem folgenden Text. Für jedes extrahierte Medikament, füge in eckigen Klammern an, ob das Medikament in Bezug auf den Patienten positiv [POSITIV], negativ [NEGATIV], spekulativ [SPEKULATIV] oder zukünftig [ZUKÜNFTIG] und links [LINKS], rechts [RECHTS] oder beidseitig [BEIDSEITIG] ist. Falls keine Medikamente im Text vorkommen, schreibe "Keine vorhanden":

<<CONTEXT>>"""
MEDICATION_NORMALIZATION_INSTRUCTION = """Weise dem im Text genannten Medikament „<<ENTITY>>“ den entsprechenden ATC-Code zu:

<<CONTEXT>>"""
MEDICATION_INSTRUCTION_CARDIO = """Extrahiere alle Medikamente aus dem folgenden Text. Für jedes extrahierte Medikament, füge falls vorhanden in eckigen Klammern die FREQUENZ und DOSIERUNG des Medikaments hinzu. Falls beides nicht vorhanden ist, füge leere eckige Klammern ein. Falls keine Medikamente im Text vorhanden sind, schreibe "Keine vorhanden":

<<CONTEXT>>"""

# DIAGNOSIS PROMPTS
DIAGNOSIS_INSTRUCTION_GENERIC = """Extrahiere alle Diagnosen und Symptome aus dem folgenden Text. Falls keine im Text vorkommen, schreibe "Keine vorhanden":

<<CONTEXT>>"""
DIAGNOSIS_INSTRUCTION_BRONCO = """Extrahiere alle Diagnosen und Symptome aus dem folgenden Text. Für jede Diagnose oder Symptom, füge in eckigen Klammern an, ob die Diagnose oder das Symptom in Bezug auf den Patienten positiv [POSITIV], negativ [NEGATIV], spekulativ [SPEKULATIV] oder zukünftig [ZUKÜNFTIG] und links [LINKS], rechts [RECHTS] oder beidseitig [BEIDSEITIG] ist. Falls keine Diagnosen oder Symptome im Text vorkommen, schreibe "Keine vorhanden":

<<CONTEXT>>"""
DIAGNOSIS_NORMALIZATION_INSTRUCTION = """Weise der im Text genannten Diagnose oder Symptom „<<ENTITY>>“ den entsprechenden ICD10-GM Code zu:

<<CONTEXT>>"""

# TREATMENT PROMPTS
TREATMENT_INSTRUCTION_GENERIC = """Extrahiere alle klinischen Prozeduren aus dem folgenden Text. Falls keine im Text vorkommen, schreibe "Keine vorhanden":

<<CONTEXT>>"""
TREATMENT_INSTRUCTION_BRONCO = """Extrahiere alle klinischen Prozeduren aus dem folgenden Text. Für jede klinische Prozedur, füge in eckigen Klammern an, ob die klinische Prozedur in Bezug auf den Patienten positiv [POSITIV], negativ [NEGATIV], spekulativ [SPEKULATIV] oder zukünftig [ZUKÜNFTIG] und links [LINKS], rechts [RECHTS] oder beidseitig [BEIDSEITIG] ist. Falls keine klinischen Prozeduren im Text vorkommen, schreibe "Keine vorhanden":

<<CONTEXT>>"""
TREATMENT_NORMALIZATION_INSTRUCTION = """Weise der im Text genannten klinischen Prozedur „<<ENTITY>>“ den entsprechenden OPS Code zu:

<<CONTEXT>>"""


class AttributeFormat(Enum):
    """
    Enum class for the different attribute formats of the extraction task. The attribute format determines the
    additional information that should be extracted for the entity.
    """

    BRONCO = "bronco"  # Bronco specific attributes --> Level of truth and location
    CARDIO = "cardio"  # Cardio specific attributes --> Frequency and dosage
    GENERIC = "generic"  # Generic attributes --> No attributes


class EntityType(Enum):
    """
    Enum class for the different entity types of the extraction and normalization tasks. The entity type determines
     the type of entity that should be extracted or normalized.
    """

    MEDICATION = "MEDICATION"
    DIAGNOSIS = "DIAGNOSIS"
    TREATMENT = "TREATMENT"


def get_extraction_messages(
    source: str, attribute_type: AttributeFormat, entity_type: EntityType
):
    messages = [{"role": "system", "content": SYSTEM_PROMPT_EXTRACTION}]
    if entity_type == EntityType.MEDICATION:
        if attribute_type == AttributeFormat.BRONCO:
            messages.append(
                {
                    "role": "user",
                    "content": MEDICATION_INSTRUCTION_BRONCO.replace(
                        "<<CONTEXT>>", source
                    ),
                }
            )
        elif attribute_type == AttributeFormat.CARDIO:
            messages.append(
                {
                    "role": "user",
                    "content": MEDICATION_INSTRUCTION_CARDIO.replace(
                        "<<CONTEXT>>", source
                    ),
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": MEDICATION_INSTRUCTION_GENERIC.replace(
                        "<<CONTEXT>>", source
                    ),
                }
            )
    elif entity_type == EntityType.DIAGNOSIS:
        if attribute_type == AttributeFormat.BRONCO:
            messages.append(
                {
                    "role": "user",
                    "content": DIAGNOSIS_INSTRUCTION_BRONCO.replace(
                        "<<CONTEXT>>", source
                    ),
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": DIAGNOSIS_INSTRUCTION_GENERIC.replace(
                        "<<CONTEXT>>", source
                    ),
                }
            )
    elif entity_type == EntityType.TREATMENT:
        if attribute_type == AttributeFormat.BRONCO:
            messages.append(
                {
                    "role": "user",
                    "content": TREATMENT_INSTRUCTION_BRONCO.replace(
                        "<<CONTEXT>>", source
                    ),
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": TREATMENT_INSTRUCTION_GENERIC.replace(
                        "<<CONTEXT>>", source
                    ),
                }
            )
    else:
        raise ValueError("Invalid entity type.")

    for message in messages:
        message["content"] = message["content"].strip()
    return messages


def get_normalization_messages(entity: str, source: str, entity_type: EntityType):
    """
    Generates the system and instruction prompts for the normalization task based on the entity and catalog type.
    :param entity: The entity to be normalized.
    :param source: The context in which the entity occurs. Can be empty but should be provided if possible.
    :param entity_type: The entity type to normalize.
    :return: List of system and instruction messages. These can then be passed to the tokenizer to create the correct
    prompt.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT_NORMALIZATION}]
    if entity_type == EntityType.DIAGNOSIS:
        messages.append(
            {
                "role": "user",
                "content": DIAGNOSIS_NORMALIZATION_INSTRUCTION.replace(
                    "<<ENTITY>>", entity
                ).replace("<<CONTEXT>>", source),
            }
        )
    elif entity_type == EntityType.MEDICATION:
        messages.append(
            {
                "role": "user",
                "content": MEDICATION_NORMALIZATION_INSTRUCTION.replace(
                    "<<ENTITY>>", entity
                ).replace("<<CONTEXT>>", source),
            }
        )
    elif entity_type == EntityType.TREATMENT:
        messages.append(
            {
                "role": "user",
                "content": TREATMENT_NORMALIZATION_INSTRUCTION.replace(
                    "<<ENTITY>>", entity
                ).replace("<<CONTEXT>>", source),
            }
        )
    else:
        raise ValueError("Invalid catalog type.")

    for message in messages:
        message["content"] = message["content"].strip()
    return messages


def get_summarization_messages(source: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT_SUMMARIZATION},
        {
            "role": "user",
            "content": SUMMARY_INSTRUCTION.replace("<<CONTEXT>>", source).strip(),
        },
    ]


class CatalogType(Enum):
    """
    Enum class for the different catalog types of the catalog task. The catalog type determines the type of catalog that
    should be used for the normalization task.
    """

    ATC = "ATC"
    ICD = "ICD"
    OPS = "OPS"


def get_catalog_messages(entity: str, catalog_type: CatalogType):
    messages = [{"role": "system", "content": SYSTEM_PROMPT_CATALOG}]
    if catalog_type == CatalogType.ATC:
        messages.append(
            {
                "role": "user",
                "content": ATC_INSTRUCTION.replace("<<ENTITY>>", entity),
            }
        )
    elif catalog_type == CatalogType.ICD:
        messages.append(
            {
                "role": "user",
                "content": ICD10GM_INSTRUCTION.replace("<<ENTITY>>", entity),
            }
        )
    elif catalog_type == CatalogType.OPS:
        messages.append(
            {
                "role": "user",
                "content": OPS_INSTRUCTION.replace("<<ENTITY>>", entity),
            }
        )
    else:
        raise ValueError("Invalid catalog type.")

    for message in messages:
        message["content"] = message["content"].strip()
    return messages


if __name__ == "__main__":
    print(
        get_normalization_messages(
            "Diabetes", "Der Patient leidet an Diabetes.", EntityType.DIAGNOSIS
        )
    )
    print(
        get_normalization_messages(
            "Paracetamol", "Der Patient nimmt Paracetamol.", EntityType.MEDICATION
        )
    )
    print(
        get_normalization_messages(
            "Blutdruckmessung", "Der Blutdruck wird gemessen.", EntityType.TREATMENT
        )
    )
    print("------------------------------")
    print(
        get_extraction_messages(
            "Der Patient nimmt Paracetamol.",
            AttributeFormat.BRONCO,
            EntityType.MEDICATION,
        )
    )
    print(
        get_extraction_messages(
            "Der Patient nimmt Paracetamol.",
            AttributeFormat.CARDIO,
            EntityType.MEDICATION,
        )
    )
    print("------------------------------")
    print(
        get_extraction_messages(
            "Der Patient hat Diabetes.", AttributeFormat.BRONCO, EntityType.DIAGNOSIS
        )
    )
    print("------------------------------")
    print(
        get_extraction_messages(
            "Der Patient wird operiert.", AttributeFormat.BRONCO, EntityType.TREATMENT
        )
    )
    print("------------------------------")
