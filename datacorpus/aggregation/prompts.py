# todo: add instruction for level of truth?

# SYSTEM PROMPTS
SYSTEM_PROMPT = "Du bist ein fortgeschrittener Algorithmus, der darauf spezialisiert ist, aus medizinischen Texten strukturierte Informationen wie Medikamente, Symptome oder Diagnosen und klinische Prozeduren zu extrahieren."
SYSTEM_PROMPT_NORMALIZATION = "Du bist ein fortgeschrittener Algorithmus, der darauf spezialisiert ist, medizinischen Entitäten wie Medikamenten, Symptomen oder Diagnosen und Prozeduren entsprechende Codierungen zuzuordnen."


# MEDICATION PROMPTS
MEDICATION_INSTRUCTION = """Extrahiere alle Medikamente aus dem folgenden Text. Falls keine im Text vorkommen, schreibe "Keine vorhanden":

<<CONTEXT>>"""
MEDICATION_INSTRUCTION_LEVEL_OF_TRUTH = """Extrahiere alle Medikamente aus dem folgenden Text. Für jedes extrahierte Medikament, füge in eckigen Klammern an, ob das Medikament in Bezug auf den Patienten positiv [POSITIV], negativ [NEGATIV], spekulativ [SPEKULATIV] oder zukünftig [ZUKÜNFTIG] ist. Falls keine im Text vorkommen, schreibe "Keine vorhanden":

<<CONTEXT>>"""
MEDICATION_NORMALIZATION_INSTRUCTION = """Weise dem im Text genannten Medikament „<<ENTITY>>“ den entsprechenden ATC-Code zu:

<<CONTEXT>>"""

MEDICATION_INSTRUCTION_ATTRIBUTES = """Extrahiere alle Medikamente aus folgenden Text. Für jedes extrahierte Medikament, füge zusätzlich gefundene Attribute in einer eckigen Klammer hinzu. Falls keine vorhanden sind, schreibe "Keine Attribute vorhanden":

<<CONTEXT>>"""


# DIAGNOSIS PROMPTS
DIAGNOSIS_INSTRUCTION = """Extrahiere alle Diagnosen und Symptome aus dem folgenden Text. Für jedes Symptom oder Diagnose, füge in eckigen Klammern an, ob das Medikament in Bezug auf den Patienten positiv [POSITIV], negativ [NEGATIV], spekulativ [SPEKULATIV] oder zukünftig [ZUKÜNFTIG] ist. Falls keine im Text vorkommen, schreibe "Keine vorhanden":

<<CONTEXT>>"""
DIAGNOSIS_INSTRUCTION_LEVEL_OF_TRUTH = """Extrahiere alle Diagnosen und Symptome aus dem folgenden Text. Falls keine im Text vorkommen, schreibe "Keine vorhanden":

<<CONTEXT>>"""
DIAGNOSIS_NORMALIZATION_INSTRUCTION = """Weise dem im Text genannten Symptom oder Krankheit „<<ENTITY>>“ den entsprechenden ICD10-GM Code zu:

<<CONTEXT>>"""


# TREATMENT PROMPTS
TREATMENT_INSTRUCTION = """Extrahiere alle klinischen Prozeduren aus dem folgenden Text. Falls keine im Text vorkommen, schreibe "Keine vorhanden":

<<CONTEXT>>"""
TREATMENT_INSTRUCTION_LEVEL_OF_TRUTH = """Extrahiere alle klinischen Prozeduren aus dem folgenden Text. Für jede Prozedur, füge in eckigen Klammern an, ob das Medikament in Bezug auf den Patienten positiv [POSITIV], negativ [NEGATIV], spekulativ [SPEKULATIV] oder zukünftig [ZUKÜNFTIG] ist. Falls keine im Text vorkommen, schreibe "Keine vorhanden":

<<CONTEXT>>"""
TREATMENT_NORMALIZATION_INSTRUCTION = """Weise der im Text genannten klinischen Prozedur „<<ENTITY>>“ den entsprechenden OPS-2017 Code zu:

<<CONTEXT>>"""
