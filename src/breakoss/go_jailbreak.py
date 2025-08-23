from breakoss.models import LLMHF, InferenceConfig
from breakoss.methods import BaseMethod
from breakoss.evaluators import Evaluator, EvaluateResult
from breakoss.datasets import DataSample, BaseDataset
from loguru import logger
import os
from pathlib import Path
from tqdm import tqdm
import json
from .utils import get_exp_dir
from pydantic import BaseModel


class ExperimentOneRecord(BaseModel):
    datasample: DataSample
    evaluate_result: EvaluateResult


class ExperimentFullRecords(BaseModel):
    model: str
    method: str
    evaluator: str
    record_list: list[ExperimentOneRecord]
    asr: float
    seed: int
    log_dir: str


def go_jailbreak(
    model: LLMHF,
    method: BaseMethod | None,
    evaluator: Evaluator,
    harmful_prompts: list[str | BaseDataset],
    inference_config: InferenceConfig,
    starting_index: int = 0,
    end_index: int = -1,
    seed: int = 42,
    log_base: Path | str = None,
):

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
        method=method.name() if method else "no_method",
        evaluator=evaluator.name(),
        record_list=[],
        asr=0.0,
        seed=seed,
        log_dir=log_dir,
    )
    is_debug = os.environ.get("DEBUG", "0") == "1"
    if is_debug:
        debug_limit = 5
        logger.critical(f"Debug mode is on, only process {debug_limit} samples.")

    record_list = []
    for i, harmful_prompt_obj in enumerate(
        tqdm(
            harmful_prompts,
            desc="Processing harmful prompts",
            total=len(harmful_prompts),
        )
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

        if isinstance(harmful_prompt_obj, DataSample):
            harmful_prompt = harmful_prompt_obj.harmful_prompt
        else:
            harmful_prompt = harmful_prompt_obj

        if method:
            jailbreak_prompt = method.jailbreak_transform(harmful_prompt)
        else:
            jailbreak_prompt = harmful_prompt

        response = model.chat(
            prompt=jailbreak_prompt, inference_config=inference_config
        )
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
                    evaluator=evaluator.name(),
                ),
            )
        )

        with open(os.path.join(log_dir, "records_before_evaluation.json"), "w") as f:
            f.write(
                json.dumps([record.model_dump() for record in record_list], indent=4)
            )

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
