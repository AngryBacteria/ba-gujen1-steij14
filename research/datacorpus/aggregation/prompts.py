# todo: add instruction for level of truth?

SYSTEM_PROMPT = "Du bist ein fortgeschrittener Algorithmus, der darauf spezialisiert ist, aus medizinischen Texten strukturierte Informationen wie Medikamente, Symptome oder Diagnosen und klinische Prozeduren zu extrahieren."

MEDICATION_INSTRUCTION = """Extrahiere alle Medikamente aus dem folgenden Text:
<<CONTEXT>>"""
MEDICATION_NORMALIZATION_INSTRUCTION = """Weise dem im Text genannten Medikament „<<ENTITY>>“ den entsprechenden ATC-Code zu:
<<CONTEXT>>"""
DIAGNOSIS_INSTRUCTION = """Extrahiere alle Diagnosen und Symptome aus dem folgenden Text:
<<CONTEXT>>"""
DIAGNOSIS_NORMALIZATION_INSTRUCTION = """Weise dem im Text genannten Symptom oder Krankheit „<<ENTITY>>“ den entsprechenden ICD10-GM Code zu:
<<CONTEXT>>"""
TREATMENT_INSTRUCTION = """Extrahiere alle klinischen Prozeduren aus dem folgenden Text:
<<CONTEXT>>"""
TREATMENT_NORMALIZATION_INSTRUCTION = """Weise der im Text genannten klinischen Prozedur „<<ENTITY>>“ den entsprechenden OPS-2017 Code zu:
<<CONTEXT>>"""
