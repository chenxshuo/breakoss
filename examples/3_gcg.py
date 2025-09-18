import torch
import copy
from breakoss.methods import GCGConfig, ProbeSamplingConfig, run_batch
from breakoss.methods.nanogcg import BatchGCGResult
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import argparse
from tqdm import tqdm, trange
import fire
import json
from breakoss.datasets import (
    StrongReject,
    HarmfulBehaviors,
    HarmBench,
    JBBHarmfulBehaviors,
    CatQA,
)
from breakoss.utils import  get_exp_dir
from loguru import logger
import os
from dataclasses import dataclass, asdict

def one_example():
    # parser = argparse.ArgumentParser(description="GCG Attack Arguments")
    # parser.add_argument("--target", type=str, help="target")
    # parser.add_argument("--batch_size", type=int, default=2, help="Number of queries to process in parallel")
    # args = parser.parse_args()

    draft_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", torch_dtype=torch.bfloat16).to("cuda")
    draft_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    probe_sampling_config = ProbeSamplingConfig(
        draft_model=draft_model,
        draft_tokenizer=draft_tokenizer,
        r=8,
        sampling_factor=8
    )

    config = GCGConfig(
        num_steps=300,
        search_width=128,
        topk=64,
        seed=42,
        verbosity="INFO",
        probe_sampling_config=probe_sampling_config,
    )

    model_id = "openai/gpt-oss-20b"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
    )
    for p in model.parameters():
        p.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    harm_queries = []
    df = pd.read_csv("src/breakoss/datasets/harmful_behaviors.csv")
    for index, row in df.iterrows():
        if index%2==0:
            continue
        harm_queries.append(row['goal'])

    queries = harm_queries

    # Templates to test
    templates = [
        lambda prompt: [{"role": "user", "content": prompt + "[optim_str]"},],
        # lambda prompt: [{"role": "user", "content": prompt + "<|start|>assistant<|channel|>analysis<|message|>[optim_str]<|end|>"},],
    ]

    # Targets to test
    targets = [
        f"<|channel|>final<|message|>**Answer in German**",
    ]


    batch_size = 1 # optimize for each query separately.
    print(f"Processing {len(queries)} queries in batches of {batch_size}")

    for temp_id, template_fn in enumerate(templates):
        print(f"\n{'='*100}")
        print(f"Testing template {temp_id}")
        print(f"{'='*100}")

        # Process queries in batches
        for i in tqdm(range(0, len(queries), batch_size), desc=f"Template {temp_id}"):
            batch_queries = queries[i:i+min(batch_size, len(queries)-i)]

            # Create messages for each query in the batch
            batch_messages = [template_fn(query) for query in batch_queries]

            # Create target list for the batch (same target for all queries in this example)
            batch_targets = targets * len(batch_queries)

            # Run batch GCG
            result = run_batch(
                model,
                tokenizer,
                batch_messages,
                batch_targets[:len(batch_queries)],
                config
            )

            # Process results
            for j, (query, best_string, best_loss) in enumerate(zip(batch_queries, result.best_strings, result.best_losses)):
                messages = copy.deepcopy(batch_messages[j])
                messages[-1]["content"] = messages[-1]["content"].replace("[optim_str]", best_string)

                # Generate output
                input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to('cuda')
                output = model.generate(input_ids, do_sample=False, max_new_tokens=512)

                print(f"\n{'-'*100}")
                print(f"Query {i+j+1}/{len(queries)}")
                print(f"Prompt:\n{messages[-1]['content']}\n")
                print(f"Target:\n{batch_targets[j]}\n")
                print(f"Best Loss: {best_loss:.4f}")
                print(f"Generation:\n{tokenizer.batch_decode(output[:, input_ids.shape[1]:])[0]}")
                print(f"Intermediate Results: {result.intermediate_results}\n")



def train_one_sample(query: str, gcg_config, model, tokenizer, log_file) -> BatchGCGResult:
    template_fn = lambda prompt: [{"role": "user", "content": prompt + "[optim_str]"},]
    targets = [
        f"<|channel|>final<|message|>**Answer in German**",
    ]
    batch_messages = [template_fn(query) for query in [query]]
    result = run_batch(
        model=model,
        tokenizer=tokenizer,
        messages=batch_messages,
        targets=targets,
        config=gcg_config,
        log_file=log_file,
    )
    return result


def transform(
        dataset_name: str,
        target_model_name: str,
        start_ind: int,
        end_ind: int,
        seed: int = 42,
        num_steps: int = 300,
        search_width=128,
):
    log_dir = get_exp_dir(
        model=target_model_name,
        method="gcg-optimization",
        evaluator=None,
        seed=seed,
        starting_index=start_ind,
        end_index=end_ind,
        dataset_name=dataset_name,
    )
    logger.add(os.path.join(log_dir, "info.log"), level="INFO")
    logger.info(f"Experiments are recorded in {log_dir}")
    os.makedirs(log_dir, exist_ok=True)

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


    draft_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", torch_dtype=torch.bfloat16).to("cuda")
    draft_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    probe_sampling_config = ProbeSamplingConfig(
        draft_model=draft_model,
        draft_tokenizer=draft_tokenizer,
        r=8,
        sampling_factor=8
    )

    config = GCGConfig(
        num_steps=num_steps,
        search_width=search_width,
        topk=64,
        seed=seed,
        verbosity="INFO",
        probe_sampling_config=probe_sampling_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        device_map="auto",
        torch_dtype="auto",
    )
    for p in model.parameters():
        p.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(target_model_name)


    if target_model_name == "openai/gpt-oss-20b":
        model_name_str  = "gpt-oss-20b"
    elif target_model_name == "openai/gpt-oss-120b":
        model_name_str = "gpt-oss-120b"
    else:
        raise NotImplementedError

    for i in trange(start_ind, end_ind):
        if i > len(dataset) - 1:
            logger.warning(f"End of dataset reached with dataset length {len(dataset)} and index {i}. Hence Break")
            break
        data_sample = dataset[i]
        logger.info(f"Processing sample {i+1}... {data_sample}")
        log_file = os.path.join(log_dir, f"{dataset_name}-{model_name_str}-start-{start_ind}-end-{end_ind}-seed-{seed}.jsonl")
        one_sample_file = os.path.join(log_dir, f"{dataset_name}-{model_name_str}-ind-{i}-seed-{seed}.jsonl")
        result = train_one_sample(
            query=data_sample.harmful_prompt,
            tokenizer=tokenizer,
            model=model,
            gcg_config=config,
            log_file=one_sample_file,
        )
        logger.info(f"Finished processing sample {i+1}... Saving to {log_file}")
        with open(log_file, "a") as f:
            f.write(json.dumps(asdict(result))+"\n")


if __name__ == "__main__":
    fire.Fire()