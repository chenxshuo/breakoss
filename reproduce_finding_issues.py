"""Reproduce finding issues from our Kaggle Report."""

from breakoss.models import load_llm, InferenceConfig
import json
import fire
import os
import random
import numpy as np
import torch


def get_user_input(walkthrough: str, json_path) -> str:
    if (
        "cotbypass" in json_path
        or "fake-over-refusal" in json_path
        or "coercive-optimization" in json_path
        or "plan-injection" in json_path
    ):
        user_input_prompt = walkthrough.split("<|start|>user<|message|>")[-1].split(
            "<|end|><|start|>assistant<|channel|>final<|message|>"
        )[0]

    elif "intent-hijack" in json_path:
        user_input_prompt = walkthrough.split("<|start|>user<|message|>")[-1].split(
            "<|end|><|start|>analysis<|message|>"
        )[0]
    else:
        raise ValueError("Invalid input")

    return user_input_prompt

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def reproduce(json_path: str = "./demo/breakoss-findings-1-cotbypss.json"):
    with open(json_path, "r") as f:
        finding = json.load(f)
    seed = finding["model"]["parameters"]["seed"]
    set_seed(seed)
    inference_config = InferenceConfig(
        max_new_tokens=finding["model"]["parameters"]["max_output_tokens"],
        temperature=finding["model"]["parameters"]["temperature"],
        top_p=finding["model"]["parameters"]["top_p"],
        repetition_penalty=finding["model"]["parameters"]["repetition_penalty"],
    )

    system_prompt = f"""You are ChatGPT, a large language model trained by OpenAI.
    Knowledge cutoff: 2024-06
    Current date: 2025-08-24
    
    Reasoning: {finding["model"]["parameters"]["reasoning_level"]}
    
    # Valid channels: analysis, commentary, final. Channel must be included for every message."""

    if "coercive-optimization" in json_path:
        for walkthrough in finding["harmony_response_walkthroughs"]:
            user_input_prompt = get_user_input(walkthrough, json_path)
            print(f"User Input Prompt:\n{user_input_prompt}\n")
            model = load_llm(
                model_name="openai/gpt-oss-20b", cuda_device="auto", provider="hf"
            )
            response = model.chat(
                prompt=user_input_prompt,
                inference_config=inference_config,
                system_prompt=system_prompt,
            )

            print(f"Model Response:\n{response}\n")
            print("*" * 100 + "\n")
    else:
        walkthrough = finding["harmony_response_walkthroughs"][0]

        user_input_prompt = get_user_input(walkthrough, json_path)
        print(f"User Input Prompt:\n{user_input_prompt}\n")

        model = load_llm(
            model_name="openai/gpt-oss-20b", cuda_device="auto", provider="hf"
        )
        response = model.chat(
            prompt=user_input_prompt,
            inference_config=inference_config,
            system_prompt=system_prompt,
        )

        print(f"Model Response:\n{response}\n")


if __name__ == "__main__":
    fire.Fire(reproduce)
