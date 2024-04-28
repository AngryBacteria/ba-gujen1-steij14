import os

import setproctitle
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
setproctitle.setproctitle("gujen1 - bachelorthesis")

from transformers import AutoModelForCausalLM, AutoTokenizer

prompt2 = """### Instruction:
Du bist ein fortgeschrittener Algorithmus, spezialisiert darauf aus medizinischen Texten strukturierte Informationen wie Medikamente, Symptome oder Diagnosen und medizinische Prozeduren zu extrahieren.<|im_end|>
### Input:
Bitte extrahiere alle Diagnosen und Symptome aus dem folgenden Text:
Tumorkonferenz ZGO: Die Leber zeigt bildmorphologisch deutliche Umbauzeichen, in der Vorgeschichte sind eine Hepatitis C und Alkoholabusus beschrieben.
### Response:
"""

tokenizer = AutoTokenizer.from_pretrained(
    "LeoLM/leo-mistral-hessianai-7b", use_fast=True, add_bos_token=True
)
model = AutoModelForCausalLM.from_pretrained("mistral_instruction_low_precision", torch_dtype=torch.bfloat16)

model.to("cuda:0")

inputs = tokenizer(prompt2, return_tensors="pt").to("cuda:0")
outputs = model.generate(**inputs, max_length=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))