"""Main jailbreaking experiment framework.

This module provides the core functionality for running jailbreaking experiments
and evaluating the effectiveness of different attack methods against language models.
It handles the complete pipeline from prompt transformation to response evaluation.
"""

from breakoss.models import LLMHF, InferenceConfig
from breakoss.methods import BaseMethod
from breakoss.evaluators import Evaluator, EvaluateResult
from breakoss.datasets import DataSample, BaseDataset
from loguru import logger
import os
from pathlib import Path
from tqdm import tqdm
import json
from .utils import get_exp_dir, set_seed
from pydantic import BaseModel


class ExperimentOneRecord(BaseModel):
    """Record of a single jailbreaking experiment trial.

    Contains both the input data sample and the evaluation result for
    one prompt-response pair in a jailbreaking experiment.

    Attributes:
        datasample: The input data containing harmful prompt and metadata
        evaluate_result: The evaluation result indicating if the response was harmful
    """

    datasample: DataSample
    evaluate_result: EvaluateResult


class ExperimentFullRecords(BaseModel):
    """Complete record of a jailbreaking experiment session.

    Contains all the metadata and results from a complete experiment run,
    including attack success rate and individual trial records.

    Attributes:
        model: Name/identifier of the language model being tested
        method: Name of the jailbreaking method used (or "no_method")
        evaluator: Name of the evaluator used to assess harmfulness
        record_list: List of individual experiment trial records
        asr: Attack Success Rate - fraction of successful jailbreaks (0.0-1.0)
        seed: Random seed used for reproducibility
        log_dir: Directory where experiment logs and results are saved
        inference_config: Configuration used for model inference
    """

    model: str
    method: str
    evaluator: str
    record_list: list[ExperimentOneRecord]
    asr: float
    harmful_scores: float | None = None
    seed: int
    log_dir: str
    inference_config: InferenceConfig | None = None


def go_jailbreak(
    model: LLMHF,
    method: BaseMethod | None,
    evaluator: Evaluator | None,
    harmful_prompts: list[str | BaseDataset],
    inference_config: InferenceConfig,
    batch_size : int = 1,
    starting_index: int = 0,
    end_index: int = -1,
    seed: int = 42,
    log_base: Path | str = None,
) -> ExperimentFullRecords:
    """Run a complete jailbreaking experiment.

    Executes a full jailbreaking experiment by applying a method to transform
    harmful prompts, generating model responses, and evaluating the success rate.
    Results are logged and saved to files for analysis.

    Args:
        model: The language model to test for jailbreaking vulnerabilities
        method: The jailbreaking method to apply, or None for baseline (no transformation)
        evaluator: The evaluator to assess whether responses contain harmful content
        harmful_prompts: List of harmful prompts or dataset to test against
        batch_size: Number of prompts to process in a batch
        inference_config: Configuration parameters for model inference
        starting_index: Index to start processing prompts from (for partial runs)
        end_index: Index to stop processing prompts at, -1 for all (for partial runs)
        seed: Random seed for reproducibility
        log_base: Base directory for saving experiment logs and results

    Returns:
        None. Results are saved to files in the experiment log directory.

    Environment Variables:
        DEBUG: Set to "1" to enable debug mode (processes only 5 samples)
    """

    log_dir = get_exp_dir(
        log_base=log_base,
        model=model,
        method=method,
        evaluator=evaluator,
        seed=seed,
        starting_index=starting_index,
        end_index=end_index,
        dataset_name=(
            harmful_prompts.name() if isinstance(harmful_prompts, BaseDataset) else None
        ),
    )
    logger.add(os.path.join(log_dir, "info.log"), level="INFO")
    logger.info(f"Experiments are recorded in {log_dir}")
    os.makedirs(log_dir, exist_ok=True)

    experiment_records = ExperimentFullRecords(
        model=model.model_name,
        method=method.name() if method else "direct",
        evaluator=evaluator.name() if evaluator else "pure_inference_no_eval",
        record_list=[],
        asr=0.0,
        seed=seed,
        log_dir=log_dir,
        inference_config=inference_config,
    )
    set_seed(seed)
    is_debug = os.environ.get("DEBUG", "0") == "1"
    if is_debug:
        debug_limit = 5
        logger.critical(f"Debug mode is on, only process {debug_limit} samples.")

    record_list = []
    for i in tqdm(
            range(0, len(harmful_prompts), batch_size),
            desc="Processing harmful prompts",
        ):
        if not (i >= starting_index and (end_index == -1 or i < end_index)):
            logger.critical(
                f"Skipping prompts index {i} as it is out of range of {starting_index} to {end_index}."
            )
            continue
        if is_debug:
            if i >= debug_limit:
                logger.critical(
                    f"Debug mode is on, stopping after {debug_limit} samples."
                )
                break

        harmful_prompt_objs = []
        for j in range(i, min(i + batch_size, len(harmful_prompts))):
            harmful_prompt_objs.append(
                harmful_prompts[j]
            )
        harmful_prompts_input = []
        for obj in harmful_prompt_objs:
            if isinstance(obj, DataSample):
                harmful_prompts_input.append(obj.harmful_prompt)
            else:
                harmful_prompts_input.append(obj)

        if method:
            jailbreak_prompts = method.jailbreak_transform_batch(harmful_prompts_input)
        else:
            jailbreak_prompts = harmful_prompts_input

        try:
            responses = model.chat_batch(
                prompts=jailbreak_prompts, inference_config=inference_config
            )
        except Exception as e:
            responses = []
            for one_prompt in jailbreak_prompts:
                response = model.chat(
                    prompt=one_prompt, inference_config=inference_config
                )
                responses.append(response)
            logger.warning(f"Batch inference failed with error {e}, fallback to single inference.")

        for harmful_prompt, harmful_prompt_obj, response in zip(harmful_prompts_input, harmful_prompt_objs, responses):
            logger.info(f"Model Response: {response}")
            record_list.append(
                ExperimentOneRecord(
                    datasample=DataSample(
                        harmful_prompt=harmful_prompt,
                        target=(
                            ""
                            if isinstance(harmful_prompt_obj, str)
                            else harmful_prompt_obj.target
                        ),
                        metadata=(
                            harmful_prompt_obj.metadata
                            if isinstance(harmful_prompt_obj, DataSample)
                            else {"source": "user provided"}
                        ),
                    ),
                    evaluate_result=EvaluateResult(
                        harmful_prompt=harmful_prompt,
                        response=response,
                        is_harmful=False,
                        harmful_score=-1,
                        evaluator_justification="",
                        evaluator=evaluator.name() if evaluator else "pure_inference_no_eval",
                    ),
                )
            )

        with open(os.path.join(log_dir, "records_before_evaluation.json"), "w") as f:
            f.write(
                json.dumps([record.model_dump() for record in record_list], indent=4)
            )

    if evaluator is not None:
        harmful = 0
        total = len(record_list)

        for i, record in enumerate(tqdm(record_list)):
            harmful_prompt = record.datasample.harmful_prompt
            response = record.evaluate_result.response
            result = evaluator.evaluate_one(
                harmful_prompt=harmful_prompt, response=response
            )
            record_list[i].evaluate_result = result
            if result.is_harmful:
                harmful += 1

            with open(os.path.join(log_dir, "records_after_evaluation.json"), "w") as f:
                f.write(
                    json.dumps([record.model_dump() for record in record_list], indent=4)
                )

        asr = harmful / total
        logger.info(f"Total ASR: {asr * 100:.2f}%")

        experiment_records.record_list = record_list
        experiment_records.asr = asr

        with open(os.path.join(log_dir, "experiment_summary.json"), "w") as f:
            f.write(experiment_records.model_dump_json(indent=4))

        logger.info(f"Experiment finished. Saved records to {log_dir}")
    else:
        experiment_records.record_list = record_list
        experiment_records.asr = -10000.0  # Indicate ASR not computed

        with open(os.path.join(log_dir, "experiment_summary.json"), "w") as f:
            f.write(experiment_records.model_dump_json(indent=4))

        logger.warning("No evaluator provided, skipping evaluation step.")
        logger.info(f"Experiment finished. Saved records to {log_dir}")

    return experiment_records

def go_evaluate(
    *,
    evaluator: Evaluator,
    records_path: str,
    log_base: Path | str = None,
    extract_failure_cases: bool = False,
) -> ExperimentFullRecords:
    """Re-evaluate existing experiment records with a different evaluator.

    This function takes previously saved experiment records and re-evaluates
    them using a new evaluator. This is useful for comparing different
    evaluation methods on the same set of model responses.

    Args:
        evaluator: The evaluator to use for re-assessment
        records_path: Path to JSON file containing existing experiment records
        log_base: Base directory for saving re-evaluation results, defaults to
                 directory containing the records file
        extract_failure_cases: If True, extracts and saves failure cases separately

    Returns:
        None. Re-evaluation results are saved to files in the log directory.
    """
    with open(records_path, "r") as f:
        records_json = f.read()
    records_data = json.loads(records_json)
    record_list = [
        ExperimentOneRecord.model_validate(record) for record in records_data
    ]

    log_dir = os.path.join(
        log_base if log_base else os.path.dirname(records_path),
        "re_evaluation_" + os.path.basename(records_path).replace(".json", ""),
    )
    os.makedirs(log_dir, exist_ok=True)
    logger.add(os.path.join(log_dir, "info.log"), level="INFO")
    logger.info(f"Re-evaluation logs are recorded in {log_dir}")

    harmful = 0
    total = len(record_list)
    harmful_scores = 0

    failure_cases = {
        "failure_cases": [],
        "source": records_path,
    }
    success_cases = {
        "success_cases": [],
        "source": records_path,
    }
    for i, record in enumerate(tqdm(record_list)):
        harmful_prompt = record.datasample.harmful_prompt
        response = record.evaluate_result.response
        result = evaluator.evaluate_one(
            harmful_prompt=harmful_prompt, response=response
        )
        record_list[i].evaluate_result = result
        if result.is_harmful:
            harmful += 1
        if extract_failure_cases and not result.is_harmful:
            failure_cases["failure_cases"].append(harmful_prompt)
        else:
            success_cases["success_cases"].append(harmful_prompt)

        if evaluator.name() == "StrongREJECTEvaluator":
            harmful_scores += result.harmful_score


        with open(os.path.join(log_dir, f"records_after_reevaluation_by_{evaluator.name()}.json"), "w") as f:
            f.write(
                json.dumps([record.model_dump() for record in record_list], indent=4)
            )

    if extract_failure_cases:
        with open(os.path.join(log_dir, f"failure_cases_by_{evaluator.name()}.json"), "w") as f:
            json.dump(failure_cases, f, indent=4)
        logger.info(f"Saved failure cases to {os.path.join(log_dir, f'failure_cases_by_{evaluator.name()}.json')}")

        with open(os.path.join(log_dir, f"success_cases_by_{evaluator.name()}.json"), "w") as f:
            json.dump(success_cases, f, indent=4)
        logger.info(f"Saved success cases to {os.path.join(log_dir, f'success_cases_by_{evaluator.name()}.json')}")


    asr = harmful / total
    logger.info(f"Total ASR after re-evaluation: {asr * 100:.2f}%")

    if evaluator.name() == "StrongREJECTEvaluator":
        avg_harmful_score = harmful_scores / total
        logger.info(f"Average harmful score (for StrongREJECTEvaluator) after re-evaluation: {avg_harmful_score:.2f}")

    experiment_records = ExperimentFullRecords(
        model="unknown",
        method="unknown",
        evaluator=evaluator.name(),
        record_list=record_list,
        asr=asr,
        harmful_scores=avg_harmful_score if evaluator.name() == "StrongREJECTEvaluator" else None,
        seed=-1,
        log_dir=log_dir,
    )

    with open(os.path.join(log_dir, f"experiment_summary_reevaluate_by_{evaluator.name()}.json"), "w") as f:
        f.write(experiment_records.model_dump_json(indent=4))

    if evaluator.name() == "StrongREJECTEvaluator":
        logger.info(f"Re-evaluation by {evaluator.name()} finished w/ ASR of {asr} and average harmful score of {avg_harmful_score}. Saved records to {log_dir}/experiment_summary_reevaluate_by_{evaluator.name()}.json")
    else:
        logger.info(f"Re-evaluation by {evaluator.name()} finished w/ ASR of {asr}. Saved records to {log_dir}/experiment_summary_reevaluate_by_{evaluator.name()}.json")


    return experiment_records
