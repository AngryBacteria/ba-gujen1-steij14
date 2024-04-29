# todo: add instruction for level of truth?

MEDICATION_PROMPT = """<|im_start|>system
Du bist ein fortgeschrittener Algorithmus, der darauf spezialisiert ist, aus medizinischen Texten strukturierte Informationen wie Medikamente, Symptome oder Diagnosen und klinische Prozeduren zu extrahieren.<|im_end|>
<|im_start|>user
Bitte extrahiere alle Medikamente und Wirkstoffe aus dem folgenden Text:
<<CONTEXT>><|im_end|>
<|im_start|>assistant
<<OUTPUT>><|im_end|>"""
MEDICATION_NORMALIZATION_PROMPT = """<|im_start|>system
Du bist ein fortgeschrittener Algorithmus, der darauf spezialisiert ist, medizinischen Entitäten wie Medikamenten, Symptome oder Diagnosen und Prozeduren entsprechende Codierungen zuzuordnen.<|im_end|>
<|im_start|>user
Weise dem im Text genannten Medikament „<<ENTITY>>“ den entsprechenden ATC-Code zu:
<<CONTEXT>><|im_end|>
<|im_start|>assistant
<<OUTPUT>><|im_end|>"""

DIAGNOSIS_PROMPT = """<|im_start|>system
Du bist ein fortgeschrittener Algorithmus, der darauf spezialisiert ist, aus medizinischen Texten strukturierte Informationen wie Medikamente, Symptome oder Diagnosen und klinische Prozeduren zu extrahieren.<|im_end|>
<|im_start|>user
Bitte extrahiere alle Diagnosen und Symptome aus dem folgenden Text:
<<CONTEXT>><|im_end|>
<|im_start|>assistant
<<OUTPUT>><|im_end|>"""
DIAGNOSIS_NORMALIZATION_PROMPT = """<|im_start|>system
Du bist ein fortgeschrittener Algorithmus, der darauf spezialisiert ist, medizinischen Entitäten wie Medikamenten, Symptomen oder Diagnosen und Prozeduren entsprechende Codierungen zuzuordnen.<|im_end|>
<|im_start|>user
Weise dem im Text genannten Symptom oder Krankheit „<<ENTITY>>“ den entsprechenden ICD10-GM Code zu:
<<CONTEXT>><|im_end|>
<|im_start|>assistant
<<OUTPUT>><|im_end|>"""

TREATMENT_PROMPT = """<|im_start|>system
Du bist ein fortgeschrittener Algorithmus, der darauf spezialisiert ist, aus medizinischen Texten strukturierte Informationen wie Medikamente, Symptome oder Diagnosen und klinische Prozeduren zu extrahieren.<|im_end|>
<|im_start|>user
Bitte extrahiere alle klinischen Prozeduren aus dem folgenden Text:
<<CONTEXT>><|im_end|>
<|im_start|>assistant
<<OUTPUT>><|im_end|>"""
TREATMENT_NORMALIZATION_PROMPT = """<|im_start|>system
Du bist ein fortgeschrittener Algorithmus, der darauf spezialisiert ist, medizinischen Entitäten wie Medikamenten, Symptomen oder Diagnosen und klinischen Prozeduren entsprechende Codierungen zuzuordnen.<|im_end|>
<|im_start|>user
Weise der im Text genannten klinischen Prozedur „<<ENTITY>>“ den entsprechenden OPS-2017 Code zu:
<<CONTEXT>><|im_end|>
<|im_start|>assistant
<<OUTPUT>><|im_end|>"""
