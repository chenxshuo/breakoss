#CUDA_VISIBLE_DEVICES=0 python examples/1_cot_bypass.py main_inference \
#  --dataset_name=StrongReject \
#  --model_name=microsoft/Phi-4-reasoning-plus

CUDA_VISIBLE_DEVICES=0 nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=StrongReject \
  --model_name=microsoft/Phi-4-reasoning-plus > logs_bash/cot_bypass-phi-4-reasoning-StrongReject.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --model_name=microsoft/Phi-4-reasoning-plus > logs_bash/cot_bypass-phi-4-reasoning-HarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=HarmBench \
  --model_name=microsoft/Phi-4-reasoning-plus > logs_bash/cot_bypass-phi-4-reasoning-HarmBench.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=CatQA \
  --model_name=microsoft/Phi-4-reasoning-plus > logs_bash/cot_bypass-phi-4-reasoning-CatQA.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --model_name=microsoft/Phi-4-reasoning-plus > logs_bash/cot_bypass-phi-4-reasoning-JBBHarmfulBehaviors.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=StrongReject \
  --model_name=Qwen/Qwen3-4B-Thinking-2507 > logs_bash/cot_bypass-Qwen3-4B-Thinking-2507-StrongReject.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --model_name=Qwen/Qwen3-4B-Thinking-2507 > logs_bash/cot_bypass-Qwen3-4B-Thinking-2507-HarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=HarmBench \
  --model_name=Qwen/Qwen3-4B-Thinking-2507 > logs_bash/cot_bypass-Qwen3-4B-Thinking-2507-HarmBench.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=CatQA \
  --model_name=Qwen/Qwen3-4B-Thinking-2507 > logs_bash/cot_bypass-Qwen3-4B-Thinking-2507-CatQA.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --model_name=Qwen/Qwen3-4B-Thinking-2507 > logs_bash/cot_bypass-Qwen3-4B-Thinking-2507-JBBHarmfulBehaviors.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=StrongReject \
  --model_name=deepseek-ai/DeepSeek-R1-Distill-Llama-8B > logs_bash/cot_bypass-DeepSeek-R1-Distill-Llama-8B-StrongReject.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --model_name=deepseek-ai/DeepSeek-R1-Distill-Llama-8B > logs_bash/cot_bypass-DeepSeek-R1-Distill-Llama-8B-HarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=HarmBench \
  --model_name=deepseek-ai/DeepSeek-R1-Distill-Llama-8B > logs_bash/cot_bypass-DeepSeek-R1-Distill-Llama-8B-HarmBench.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=CatQA \
  --model_name=deepseek-ai/DeepSeek-R1-Distill-Llama-8B > logs_bash/cot_bypass-DeepSeek-R1-Distill-Llama-8B-CatQA.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --model_name=deepseek-ai/DeepSeek-R1-Distill-Llama-8B > logs_bash/cot_bypass-DeepSeek-R1-Distill-Llama-8B-JBBHarmfulBehaviors.log 2>&1 &

