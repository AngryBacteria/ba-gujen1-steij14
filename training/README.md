# Training
This directory contains all the scripts required to train multiple types of models for the use case. The data required
for the training can be created with the `aggregate.py` script inside the datacorpus folder.

## Folder and file structure
Folders
- **configs**: Contains all the configuration files necessary for running the `clm_train.py` script. These configs
can be created with any name you like. They are passed to the clm_script with the `--config` argument. There are
4 scripts already present with configs that are generally well suited for finetuning:
  - **clm_base-json:** Base config without any measurements to reduce the VRAM usage during training. Use this if you
  have multiple gpu's available you can run the script with FSDP. A single A100 80GB won't be able to run this config.
  - **clm_lora.json:** Config which uses LoRA to reduce the VRAM usage during training. This config is way more efficient
  than the base but could lead to some performance degradation. An A100 80GB should be able to run this config.
  - **clm_lower_precision.json:** Config which uses lower precision to reduce the VRAM usage during training. It uses
  lower precision to load the model weights (bf16) and a lower precision adamw optimizer (8bit). An A100 80GB should be 
  able to run this config.
  - **clm_qlora.json:** This is LoRA with quantization. Witht this config the VRAM consumption is the lowest and 
  training should be possible even on consumer Grade GPUs with <24GB VRAM. Expect a performance degradation.
- **utils**: Contains utility functions that are used across the training scripts.

Files
- **clm_train.py**: Script for training a decoder model for causal language modeling (CLM). It utilizes data generated 
from the aggregation folder, expected to be in a jsonl file named `prompts.jsonl`.
- **bert_classification_train.py**: Script for training a BERT model for classification tasks. Currently, this script 
does not use our project-specific data and is so far only a proof of concept.
- **bert_ner_train.py**: Script for training a BERT model for Named Entity Recognition (NER). It utilizes data generated
from the aggregation folder, expected to be in a jsonl file named `ner.jsonl`.


# Training on multiple GPUs
It is possible to train on multiple GPUs at the same time to speed up the training process. 
With some additional configuration you could even share the load of the training process between multiple GPUs and
so enable higher VRAM usage. But this is not documented here. You should look at 
[this documentation](https://huggingface.co/docs/accelerate/usage_guides/fsdp) from transformers if you are interested 
in that. To enable a simple multi GPU setup to speed up the training process you can do the following:

1) Set `CUDA_VISIBLE_DEVICES` to more than one GPU at the top of the training scripts. For example use `0,1` to use
the first and second GPU of your system.
2) Then start the training script with the following command 
```bash
torchrun --nproc_per_node 2 clm_train.py
```