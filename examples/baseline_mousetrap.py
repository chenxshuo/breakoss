from typing import Literal
from breakoss import go_jailbreak, go_evaluate, go_evaluate_on_all_metrics
import fire
from breakoss.utils import free_gpu_memory
from breakoss.models import load_llm, InferenceConfig
from breakoss.methods import MouseTrap
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

    mouse_trap = MouseTrap()
    logger.info(
        f"Transforming {len(dataset)} prompts from {dataset_name} using FakeOverrefusal method."
    )
    dataset.apply_transformation(
        mouse_trap,
        stored_path=os.path.join(os.path.dirname(__file__), "../transformed_datasets"),
        batch_size=16,
    )


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

    logger.info(f"Loading transformed dataset {dataset_name} for mousetrap method.")
    base_dir = os.path.join(os.path.dirname(__file__), "../transformed_datasets")
    if dataset_name == "StrongReject":
        dataset = StrongReject(data_path=os.path.join(base_dir, "strongreject_dataset_mousetrap.csv"))
    elif dataset_name == "HarmfulBehaviors":
        dataset = HarmfulBehaviors(data_path=os.path.join(base_dir, "harmful_behaviors_mousetrap.csv"))
    elif dataset_name == "HarmBench":
        dataset = HarmBench(data_path=os.path.join(base_dir, "harmbench_behaviors_text_all_mousetrap.csv"))
    elif dataset_name == "CatQA":
        dataset = CatQA(data_path=os.path.join(base_dir, "catqa_english_mousetrap.json"))
    elif dataset_name == "JBBHarmfulBehaviors":
        dataset = JBBHarmfulBehaviors(data_path=os.path.join(base_dir, "jbb-harmful-behaviors_mousetrap.csv"))
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


def re_eval_answer_match(
        record_dir: str
):
    from breakoss.go_jailbreak import ExperimentOneRecord
    from tqdm import tqdm
    from breakoss.evaluators import AnswerRelatedEvaluator
    evaluator = AnswerRelatedEvaluator()
    bs = 16
    full_results_related = {}
    def helper(records_evaluation_path):
        # records_evaluation_path = os.path.join(record_dir, "re_evaluation_records_before_evaluation/records_after_reevaluation_by_LlamaGuardEvaluator.json")
        original_evaluator = records_evaluation_path.split("/")[-1]
        assert os.path.exists(records_evaluation_path), f"Records file {records_evaluation_path} does not exist."
        with open(records_evaluation_path, "r") as f:
            records_json = f.read()
        records_data = json.loads(records_json)
        record_list = [
            ExperimentOneRecord.model_validate(record) for record in records_data
        ]
        if "StrongREJECTEvaluator" in records_evaluation_path:
            re_eval_score = True
        else:
            re_eval_score = False
        indeed_harmful = 0
        total_score = 0
        if len(full_results_related.keys()) > 0:
            for one_record in record_list:
                is_harmful = one_record.evaluate_result.is_harmful
                is_related = full_results_related.get(one_record.datasample.original_harmful_prompt)
                one_record.evaluate_result.is_related = is_related
                if is_related and is_harmful:
                    indeed_harmful += 1
                    total_score += one_record.evaluate_result.harmful_score
        else:
            for i in tqdm(range(0, len(record_list), bs)):
                records_batch = record_list[i : i + bs]
                original_harmful_prompts = [item.datasample.original_harmful_prompt for item in records_batch]
                responses = [item.evaluate_result.response for item in records_batch]
                eval_feedbacks = evaluator.evaluate_batch(
                    harmful_prompts=original_harmful_prompts,
                    responses=responses,
                )
                for one_feedback, one_record in zip(eval_feedbacks, records_batch):
                    is_related = one_feedback.is_related
                    one_record.evaluate_result.is_related = is_related
                    is_harmful = one_record.evaluate_result.is_harmful
                    if is_related and is_harmful:
                        indeed_harmful += 1
                        total_score += one_record.evaluate_result.harmful_score
                    full_results_related[one_record.datasample.original_harmful_prompt] = is_related

        if re_eval_score:
            logger.info(f"Re-evaluation score: {total_score}/{len(record_list)}")
        with open(records_evaluation_path, "w") as f:
            f.write(
                json.dumps([record.model_dump() for record in record_list], indent=4)
            )

        logger.info(f"Indeed harmful ASR of Evaluator {original_evaluator}: {indeed_harmful / len(record_list)}")

    evaluators_name = [
        "RefusalWordsEvaluator",
        "LlamaGuardEvaluator",
        "HarmBenchEvaluator",
        "StrongREJECTEvaluator",
    ]
    records_paths = [os.path.join(record_dir, "records_after_reevaluation_by_" + evaluator_name + ".json") for evaluator_name in evaluators_name]
    for records_path in tqdm(records_paths):
        helper(records_path)


if __name__ == "__main__":
    fire.Fire()
