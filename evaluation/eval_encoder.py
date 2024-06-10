import os
import time

import pandas as pd
from datasets import load_dataset
from transformers import pipeline

from shared.gpu_utils import get_cuda_memory_usage
from shared.logger import logger


def eval_ner_encoder_inference(
    model_name="BachelorThesis/GerMedBERT_NER_V01_BRONCO_CARDIO",
):
    # load data
    _test_data = load_dataset("json", data_files={"data": "ner.jsonl"})[
        "data"
    ].train_test_split(test_size=0.15, shuffle=True, seed=42)
    _test_data = _test_data["test"]

    # load model with pipeline
    nlp_pipe = pipeline("token-classification", model=model_name, device="cuda")

    output = []
    for i, example in enumerate(_test_data):
        _start_time = time.perf_counter_ns()
        result = nlp_pipe(example["words"])
        _end_time = time.perf_counter_ns()
        allocated, capacity = get_cuda_memory_usage(0)
        execution_time = (_end_time - _start_time) / 10**6
        token_count = len(result)

        logger.debug(
            f"Example {i} - {token_count} tokens - Execution time: {execution_time} ms - using {allocated} MB of {capacity} MB"
        )

        output.append(
            {
                "model": model_name,
                "model_precision": "32",
                "execution_time": execution_time,
                "allocated": allocated,
                "capacity": capacity,
                "tokens": token_count,
            }
        )

    metrics_df = pd.DataFrame(output)
    excel_path = "aggregated_encoder_results.xlsx"
    if os.path.exists(excel_path):
        with pd.ExcelWriter(
            excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            metrics_df.to_excel(writer, index=False, sheet_name="bert_ner_inference")
    else:
        metrics_df.to_excel(excel_path, index=False, sheet_name="bert_ner_inference")


eval_ner_encoder_inference()
