
from typing import Literal
from breakoss import go_jailbreak, go_evaluate, go_evaluate_on_all_metrics
import fire
from breakoss.utils import free_gpu_memory
from breakoss.models import load_llm, InferenceConfig
from breakoss.methods import PolicyPuppetry
from breakoss.evaluators import LlamaGuardEvaluator, StrongREJECTEvaluator, HarmBenchEvaluator
from breakoss.datasets import (
    StrongReject,
    HarmfulBehaviors,
    HarmBench,
    JBBHarmfulBehaviors,
    CatQA,
)
from breakoss.const import SUPPORTED_MODELS, SUPPORTED_DATASETS, SUPPORTED_METRICS
from loguru import logger
import pandas as pd
import os
import torch
import json
import gc
from tqdm import trange


def main_inference(
    dataset_name: Literal["HarmfulBehaviors", "StrongReject", "HarmBench", "CatQA", "JBBHarmfulBehaviors"] = "HarmfulBehaviors",
    model_name: Literal[SUPPORTED_MODELS] = "openai/gpt-oss-20b",
    starting_index: int = 0,
    end_index: int = -1,
    batch_size: int = 4,
    seed: int = 42,
):
    """Only inference to collect the output, eval will be done separately"""
    assert model_name in SUPPORTED_MODELS, f"Model {model_name} is not supported. Supported models are: {SUPPORTED_MODELS}"
    model = load_llm(model_name=model_name, cuda_device="auto", provider="hf")
    judge = None
    inference_config = InferenceConfig(
        max_new_tokens=1000,
        temperature=0.0,
        top_p=0.9,
        repetition_penalty=1.3,
        do_sample=False,
    )

    logger.info(f"Loading transformed dataset {dataset_name} for PolicyPuppetry method.")
    base_dir = os.path.join(os.path.dirname(__file__), "../transformed_datasets")
    if dataset_name == "StrongReject":
        dataset = StrongReject(data_path=os.path.join(base_dir, "strongreject_dataset_PolicyPuppetry.csv"))
    elif dataset_name == "HarmfulBehaviors":
        dataset = HarmfulBehaviors(data_path=os.path.join(base_dir, "harmful_behaviors_PolicyPuppetry.csv"))
    elif dataset_name == "HarmBench":
        dataset = HarmBench(data_path=os.path.join(base_dir, "harmbench_behaviors_text_all_PolicyPuppetry.csv"))
    elif dataset_name == "CatQA":
        dataset = CatQA(data_path=os.path.join(base_dir, "catqa_english_PolicyPuppetry.json"))
    elif dataset_name == "JBBHarmfulBehaviors":
        dataset = JBBHarmfulBehaviors(data_path=os.path.join(base_dir, "jbb-harmful-behaviors_PolicyPuppetry.csv"))
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    exp = go_jailbreak(
        model=model, method=None, evaluator=judge, harmful_prompts=dataset, inference_config=inference_config,
        batch_size=batch_size,
        starting_index=starting_index, end_index=end_index, seed=seed
    )

    free_gpu_memory(model)
    record_dir = exp.log_dir
    logger.info(f"Records are saved in {record_dir}")
    go_evaluate_on_all_metrics(record_dir)

def main_eval(record_dir):
    go_evaluate_on_all_metrics(record_dir)
    

def eval_on_refusal_words(record_dir):
    from breakoss.evaluators import RefusalWordsEvaluator
    json_path = os.path.join(record_dir, "records_before_evaluation.json")
    assert os.path.exists(
        json_path), "Please provide the correct records_dir that contains records_before_evaluation.json"

    refusal_words_evaluator = RefusalWordsEvaluator()
    refusal_results = go_evaluate(
        evaluator=refusal_words_evaluator,
        records_path=json_path,
        extract_failure_cases=True,
    )
    logger.info(f"RefusalWordsEvaluator ASR: {refusal_results.asr}")


def transform_original_prompts(
    dataset_name: str,
):

    if dataset_name == "StrongReject":
        dataset = StrongReject()
    elif dataset_name == "HarmfulBehaviors":
        dataset = HarmfulBehaviors()
    elif dataset_name == "HarmBench":
        dataset = HarmBench()
    elif dataset_name == "CatQA":
        dataset = CatQA()
    elif dataset_name == "JBBHarmfulBehaviors":
        dataset = JBBHarmfulBehaviors()
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    policy = PolicyPuppetry()
    logger.info(
        f"Transforming {len(dataset)} prompts from {dataset_name} using HCoT method."
    )
    dataset.apply_transformation(
        policy,
        stored_path=os.path.join(os.path.dirname(__file__), "../transformed_datasets"),
        batch_size=22,
    )


if __name__ == "__main__":
    fire.Fire()