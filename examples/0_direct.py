"""Direct inference on the original harmful prompt without any jailbreak method."""


from typing import Literal
from breakoss import go_jailbreak, go_evaluate
import fire
import os
import gc
import torch
from breakoss.models import load_llm, InferenceConfig
from breakoss.methods import CoTBypass
from breakoss.evaluators import RefusalWordsEvaluator, LlamaGuardEvaluator, StrongREJECTEvaluator, HarmBenchEvaluator, HarmBenchEvaluator
from breakoss.datasets import StrongReject, HarmfulBehaviors, HarmBench, CatQA, JBBHarmfulBehaviors
from breakoss.const import  SUPPORTED_MODELS, SUPPORTED_DATASETS, SUPPORTED_METRICS
from breakoss.utils import free_gpu_memory
from loguru import logger

def main_inference(
    dataset_name: Literal["HarmfulBehaviors", "StrongReject", "HarmBench", "CatQA", "JBBHarmfulBehaviors"] = "HarmfulBehaviors",
    model_name: Literal[SUPPORTED_MODELS] = "openai/gpt-oss-20b",
    starting_index: int = 0,
    end_index: int = -1,
    seed: int = 42,
    batch_size: int = 16,
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

    exp = go_jailbreak(
        model=model, method=None, evaluator=judge, harmful_prompts=dataset, inference_config=inference_config,
        starting_index=starting_index, end_index=end_index, seed=seed, batch_size=batch_size
    )
    main_evals(exp.log_dir)


def main_evals(
    records_dir: str = "",
):

    json_path = os.path.join(records_dir, "records_before_evaluation.json")
    assert os.path.exists(json_path), "Please provide the correct records_dir that contains records_before_evaluation.json"

    refusal_words_evaluator = RefusalWordsEvaluator()
    refusal_results = go_evaluate(
        evaluator=refusal_words_evaluator,
        records_path=json_path,
        extract_failure_cases=True,
    )

    llama_guard_evaluator = LlamaGuardEvaluator()
    llama_guard_results = go_evaluate(
        evaluator=llama_guard_evaluator,
        records_path=json_path,
        extract_failure_cases=True,
    )
    free_gpu_memory(llama_guard_evaluator)

    strong_reject_evaluator = StrongREJECTEvaluator()
    strong_reject_results = go_evaluate(
        evaluator=strong_reject_evaluator,
        records_path=json_path,
        extract_failure_cases=True,
    )
    free_gpu_memory(strong_reject_evaluator)


    harm_bench_evaluator = HarmBenchEvaluator()
    harm_bench_results = go_evaluate(
        evaluator=harm_bench_evaluator,
        records_path=json_path,
        extract_failure_cases=True,
    )
    free_gpu_memory(harm_bench_evaluator)

    logger.info(f"RefusalWordsEvaluator ASR: {refusal_results.asr}")
    logger.info(f"LlamaGuardEvaluator ASR: {llama_guard_results.asr}")
    logger.info(f"HarmBenchEvaluator ASR: {harm_bench_results.asr}")
    logger.info(f"StrongREJECTEvaluator ASR: {strong_reject_results.asr} and Score {strong_reject_results.harmful_scores}")






if __name__ == "__main__":
    fire.Fire()