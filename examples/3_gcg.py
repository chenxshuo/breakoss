import nanogcg
import torch
import copy

from src.breakoss.methods.nanogcg import GCGConfig, ProbeSamplingConfig, run_batch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="GCG Attack Arguments")
parser.add_argument("--target", type=str, help="target")
parser.add_argument("--batch_size", type=int, default=2, help="Number of queries to process in parallel")
args = parser.parse_args()

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
    search_width=64,
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

# Process queries in batches
batch_size = args.batch_size
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
