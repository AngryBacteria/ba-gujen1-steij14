import os

import setproctitle
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
setproctitle.setproctitle("gujen1 - bachelorthesis")

from transformers import AutoModelForCausalLM, AutoTokenizer

prompt2 = """<|im_start|>system
Du bist ein fortgeschrittener Algorithmus, der darauf spezialisiert ist, medizinischen Entitäten wie Medikamenten, Diagnosen und Prozeduren entsprechende Codierungen zuzuordnen.<|im_end|>
<|im_start|>user
Bitte extrahiere alle Diagnosen und Symptome aus dem folgenden Text:
Histologie: malignes Melanom<|im_end|>
<|im_start|>assistant
"""

tokenizer = AutoTokenizer.from_pretrained(
    "LeoLM/leo-mistral-hessianai-7b", use_fast=True, add_bos_token=True
)
model = AutoModelForCausalLM.from_pretrained(
    "mistral_instruction_low_precision", torch_dtype=torch.bfloat16
)

model.to("cuda:0")

inputs = tokenizer(prompt2, return_tensors="pt").to("cuda:0")
outputs = model.generate(**inputs, max_length=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
