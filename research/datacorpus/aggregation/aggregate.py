# TODO: aggregation of all corpus datasets

MEDICATION_PROMPT = """Extrahiere aus dem nachfolgenden Satz alle Medikamente und Wirkstoffe:
# Satz:
<<CONTEXT>>

# Medikamente und Wirkstoffe:
<<OUTPUT>>
"""
MEDICATION_NORMALIZATION_PROMPT = """Weise diesem Medikament den ATC-Code zu:
# Medikament:
<<CONTEXT>>

# ATC-Code:
<<OUTPUT>>
"""


DIAGNOSIS_PROMPT = """Extrahiere aus dem nachfolgenden Satz alle Diagnosen und Symptome:
# Satz:
<<CONTEXT>>

# Diagnosen und Symptome:
<<OUTPUT>>
"""
DIAGNOSIS_NORMALIZATION_PROMPT = """Weise dieser Krankheit oder diesem Symptom den ICD10-GM Code zu:
# Diagnose oder Symptom:
<<CONTEXT>>

# ICD10-GM Code:
<<OUTPUT>>
"""


TREATMENT_PROMPT = """Extrahiere aus dem nachfolgenden Satz alle medizinischen Behandlungen:
# Satz:
<<CONTEXT>>

# Behandlungen:
<<OUTPUT>>
"""
TREATMENT_NORMALIZATION_PROMPT = """Weise dieser Behandlung den OPS-2017 Code zu:
# Behandlung:
<<CONTEXT>>

# ATC-Code:
<<OUTPUT>>
"""
