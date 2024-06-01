# Evaluation
This directory contains the scripts required to validate the different models that can be created with the training
scripts. The validation scripts are used to evaluate the performance of the models on the validation dataset. Right
now the following models can be validated:
- **eval_decoder.py:** This script evaluates the performance of a causal language model (CLM) on the validation dataset.
It validates the tasks extraction and normalization for medical entities. The key metrics are the Precision, Recall and
the F1-Score. It does the validation with the `prompts.jsonl` file which can be created with the scripts from
the aggregation directory.
- **eval_lookup.py:** TODO: not yet implemented. Comparison for quality with a simple database lookup
- **Validation for BERT Models:** The BERT Models are automatically validated with the same metrics during the training.
Separate scripts are hence not required because the logging during the training suffices.