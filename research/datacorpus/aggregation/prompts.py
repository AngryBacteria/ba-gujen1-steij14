MEDICATION_PROMPT = """<|im_start|>system
Du bist ein fortgeschrittener Algorithmus, spezialisiert darauf aus medizinischen Texten strukturierte Informationen wie Medikamente, Symptome oder Diagnosen und medizinische Prozeduren zu extrahieren.<|im_end|>
<|im_start|>user
Bitte extrahiere alle Medikamente und Wirkstoffe aus dem folgenden Text:
<<CONTEXT>><|im_end|>
<|im_start|><<OUTPUT>>"""
MEDICATION_NORMALIZATION_PROMPT = """<|im_start|>system
Du bist ein fortgeschrittener Algorithmus, der darauf spezialisiert ist, medizinischen Entitäten wie Medikamenten, Diagnosen und Prozeduren entsprechende Codierungen zuzuordnen.<|im_end|>
<|im_start|>user
Weise den im Text genannten Medikamenten „<<ENTITIES>>“ ihre entsprechenden ATC-Codes zu:
<<CONTEXT>><|im_end|>
<|im_start|><<OUTPUT>>"""

DIAGNOSIS_PROMPT = """<|im_start|>system
Du bist ein fortgeschrittener Algorithmus, spezialisiert darauf aus medizinischen Texten strukturierte Informationen wie Medikamente, Symptome oder Diagnosen und medizinische Prozeduren zu extrahieren.<|im_end|>
<|im_start|>user
Bitte extrahiere alle Diagnosen und Symptome aus dem folgenden Text:
<<CONTEXT>><|im_end|>
<|im_start|><<OUTPUT>>"""
DIAGNOSIS_NORMALIZATION_PROMPT = """<|im_start|>system
Du bist ein fortgeschrittener Algorithmus, der darauf spezialisiert ist, medizinischen Entitäten wie Medikamenten, Diagnosen und Prozeduren entsprechende Codierungen zuzuordnen.<|im_end|>
<|im_start|>user
Weise den im Text genannten Symptome oder Krankheiten „<<ENTITIES>>“ ihre entsprechenden ICD10-GM Codes zu:
<<CONTEXT>><|im_end|>
<|im_start|><<OUTPUT>>"""

TREATMENT_PROMPT = """<|im_start|>system
Du bist ein fortgeschrittener Algorithmus, spezialisiert darauf aus medizinischen Texten strukturierte Informationen wie Medikamente, Symptome oder Diagnosen und klinische Prozeduren zu extrahieren.<|im_end|>
<|im_start|>user
Bitte extrahiere alle klinischen Prozeduren aus dem folgenden Text:
<<CONTEXT>><|im_end|>
<|im_start|><<OUTPUT>>"""
TREATMENT_NORMALIZATION_PROMPT = """<|im_start|>system
Du bist ein fortgeschrittener Algorithmus, der darauf spezialisiert ist, medizinischen Entitäten wie Medikamenten, Diagnosen und Prozeduren entsprechende Codierungen zuzuordnen.<|im_end|>
<|im_start|>user
Weise den im Text genannten klinischen Prozeduren „<<ENTITIES>>“ ihre entsprechenden OPS-2017 Codes zu:
<<CONTEXT>><|im_end|>
<|im_start|><<OUTPUT>>"""
