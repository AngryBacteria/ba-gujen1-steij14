# SYSTEM PROMPTS
SYSTEM_PROMPT = "Du bist ein fortgeschrittener Algorithmus, der darauf spezialisiert ist, aus medizinischen Texten strukturierte Informationen wie Medikamente, Symptome oder Diagnosen und klinische Prozeduren zu extrahieren."
SYSTEM_PROMPT_NORMALIZATION = "Du bist ein fortgeschrittener Algorithmus, der darauf spezialisiert ist, medizinischen Entitäten wie Medikamenten, Symptomen oder Diagnosen und Prozeduren entsprechende Codierungen zuzuordnen."
SYSTEM_PROMPT_SUMMARIZATION = "Du bist ein fortschrittlicher Algorithmus, der darauf spezialisiert ist, aus umfangreichen Texten die Hauptpunkte zu extrahieren und sie in einer prägnanten, zusammengefassten Form darzustellen. Deine Aufgabe ist es, den Kerninhalt effektiv zu erfassen und dabei wichtige Details, Schlussfolgerungen und Argumentationsstränge beizubehalten."

# MEDICATION PROMPTS
MEDICATION_INSTRUCTION = """Extrahiere alle Medikamente aus dem folgenden Text. Falls keine im Text vorkommen, schreibe "Keine vorhanden":

<<CONTEXT>>"""
MEDICATION_INSTRUCTION_LEVEL_OF_TRUTH = """Extrahiere alle Medikamente aus dem folgenden Text. Für jedes extrahierte Medikament, füge in eckigen Klammern an, ob das Medikament in Bezug auf den Patienten positiv [POSITIV], negativ [NEGATIV], spekulativ [SPEKULATIV] oder zukünftig [ZUKÜNFTIG] ist. Falls keine Medikamente im Text vorkommen, schreibe "Keine vorhanden":

<<CONTEXT>>"""
MEDICATION_NORMALIZATION_INSTRUCTION = """Weise dem im Text genannten Medikament „<<ENTITY>>“ den entsprechenden ATC-Code zu:

<<CONTEXT>>"""
MEDICATION_INSTRUCTION_ATTRIBUTES = """Extrahiere alle Medikamente aus folgenden Text. Für jedes extrahierte Medikament, füge falls vorhanden in eckigen Klammern die FREQUENZ und DOSIERUNG des Medikaments hinzu. Falls beides nicht vorhanden ist, füge leere eckige Klammern ein. Falls keine Medikamente im Text vorhanden sind, schreibe "Keine vorhanden":

<<CONTEXT>>"""


# DIAGNOSIS PROMPTS
DIAGNOSIS_INSTRUCTION = """Extrahiere alle Diagnosen und Symptome aus dem folgenden Text. Falls keine im Text vorkommen, schreibe "Keine vorhanden":

<<CONTEXT>>"""
DIAGNOSIS_INSTRUCTION_LEVEL_OF_TRUTH = """Extrahiere alle Diagnosen und Symptome aus dem folgenden Text. Für jede Diagnose oder Symptom, füge in eckigen Klammern an, ob die Diagnose oder das Symptom in Bezug auf den Patienten positiv [POSITIV], negativ [NEGATIV], spekulativ [SPEKULATIV] oder zukünftig [ZUKÜNFTIG] ist. Falls keine Diagnosen oder Symptome im Text vorkommen, schreibe "Keine vorhanden":

<<CONTEXT>>"""
DIAGNOSIS_NORMALIZATION_INSTRUCTION = """Weise der im Text genannten Diagnose oder Symptom „<<ENTITY>>“ den entsprechenden ICD10-GM Code zu:

<<CONTEXT>>"""


# TREATMENT PROMPTS
TREATMENT_INSTRUCTION = """Extrahiere alle klinischen Prozeduren aus dem folgenden Text. Falls keine im Text vorkommen, schreibe "Keine vorhanden":

<<CONTEXT>>"""
TREATMENT_INSTRUCTION_LEVEL_OF_TRUTH = """Extrahiere alle klinischen Prozeduren aus dem folgenden Text. Für jede klinische Prozedur, füge in eckigen Klammern an, ob die klinische Prozedur in Bezug auf den Patienten positiv [POSITIV], negativ [NEGATIV], spekulativ [SPEKULATIV] oder zukünftig [ZUKÜNFTIG] ist. Falls keine klinischen Prozeduren im Text vorkommen, schreibe "Keine vorhanden":

<<CONTEXT>>"""
TREATMENT_NORMALIZATION_INSTRUCTION = """Weise der im Text genannten klinischen Prozedur „<<ENTITY>>“ den entsprechenden OPS-2017 Code zu:

<<CONTEXT>>"""

# SUMMARY PROMPTS
SUMMARY_INSTRUCTION = """Bitte fasse den folgenden Artikel aus der Zeitschrift zusammen. Das Ziel ist es, einen kompakten Überblick über den Inhalt des Artikels zu geben:

<<CONTEXT>>"""
