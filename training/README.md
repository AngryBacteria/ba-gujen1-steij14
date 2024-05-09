# Folder struture
- **configs**: Contains all the configuration files necessary for running the `clm_train.py` script.
- **utils**: Contains utility functions that are used across the training scripts.


- **bert_classification_train.py**: Script for training a BERT model for classification tasks. Currently, this script does not use our project-specific data.
- **bert_ner_train.py**: Script for training a BERT model for Named Entity Recognition (NER). It is not yet configured to use our project-specific data.
- **clm_train.py**: Script for training a Mistral model for causal language modeling (CLM). It utilizes data generated from the aggregation module, expected to be in a CSV file named `prompts.json`.

# Fix python linux setup
1) export PYTHONPATH=/home/duser/ba-gujen1-steij14:$PYTHONPATH

# Training on multiple GPUs
1) Set CUDA_VISIBLE_DEVICES to more than one GPU
2) torchrun --nproc_per_node 2 train.py