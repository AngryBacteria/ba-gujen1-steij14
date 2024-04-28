import os

import setproctitle
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
setproctitle.setproctitle("gujen1 - bachelorthesis")

from transformers import AutoModelForCausalLM, AutoTokenizer

prompt1 = """<|im_start|>system
Du bist ein fortgeschrittener Algorithmus, spezialisiert darauf aus medizinischen Texten strukturierte Informationen wie Medikamente, Symptome oder Diagnosen und medizinische Prozeduren zu extrahieren.<|im_end|>
<|im_start|>user
Bitte extrahiere alle Diagnosen und Symptome aus dem folgenden Text:
Wegen einer symptomatischen Anaemie erfolgte die komplikationslose Transfusion von 1 Erythrozytenkonzentraten.<|im_end|>
<|im_start|>assistant
Anaemie<|im_end|>"""

prompt2 = """<|im_start|>system
Du bist ein fortgeschrittener Algorithmus, spezialisiert darauf aus medizinischen Texten strukturierte Informationen wie Medikamente, Symptome oder Diagnosen und medizinische Prozeduren zu extrahieren.<|im_end|>
<|im_start|>user
Bitte extrahiere alle Diagnosen und Symptome aus dem folgenden Text:
Die stationaere Aufnahme von Herrn/Frau PATIENT erfolgte zur elektiv radikalen Lymphadenektomie inguinal und iliakal bei Lymphknotenmetastasen eines malignen Melanoms.<|im_end|>
<|im_start|>assistant
malignen Melanoms|Lymphknotenmetastasen"""

tokenizer = AutoTokenizer.from_pretrained(
    "LeoLM/leo-mistral-hessianai-7b-chat", use_fast=True, add_bos_token=True
)
model = AutoModelForCausalLM.from_pretrained("mistral_instruction_low_precision", torch_dtype=torch.bfloat16)

model.to("cuda:0")

inputs = tokenizer(prompt2, return_tensors="pt").to("cuda:0")
outputs = model.generate(**inputs, max_length=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))