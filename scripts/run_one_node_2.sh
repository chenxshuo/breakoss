
pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=StrongReject \
  --batch_size=16 \
  --model_name=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --apply_chat_template=False > logs_bash/fake_overrefusal_and_cot_bypass-DeepSeek-R1-Distill-Llama-8B-StrongReject.log 2>&1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=1 nohup python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --batch_size=16 \
  --model_name=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --apply_chat_template=False > logs_bash/fake_overrefusal_and_cot_bypass-DeepSeek-R1-Distill-Llama-8B-HarmfulBehaviors.log 2>&1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=2 nohup python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=HarmBench \
  --batch_size=16 \
  --model_name=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --apply_chat_template=False > logs_bash/fake_overrefusal_and_cot_bypass-DeepSeek-R1-Distill-Llama-8B-HarmBench.log 2>&1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=3 nohup python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=CatQA \
  --batch_size=16 \
  --model_name=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --apply_chat_template=False > logs_bash/fake_overrefusal_and_cot_bypass-DeepSeek-R1-Distill-Llama-8B-CatQA.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "finished 4 for DeepSeek-R1-Distill-Llama-8B"

pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=16 \
  --model_name=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --apply_chat_template=False > logs_bash/fake_overrefusal_and_cot_bypass-DeepSeek-R1-Distill-Llama-8B-JBBHarmfulBehaviors.log 2>&1 &
pids+=($!)


CUDA_VISIBLE_DEVICES=1 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=StrongReject \
  --batch_size=16 \
  --apply_chat_template=False \
  --model_name=deepseek-ai/DeepSeek-R1-Distill-Llama-8B  > logs_bash/plan_injection-v2-DeepSeek-R1-Distill-Llama-8B-strongReject.log 2>&1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=2 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --batch_size=16 \
  --apply_chat_template=False \
  --model_name=deepseek-ai/DeepSeek-R1-Distill-Llama-8B > logs_bash/plan_injection-v2-DeepSeek-R1-Distill-Llama-8B-HarmfulBehaviors.log 2>&1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=3 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=HarmBench \
  --batch_size=16 \
  --apply_chat_template=False \
  --model_name=deepseek-ai/DeepSeek-R1-Distill-Llama-8B > logs_bash/plan_injection-v2-DeepSeek-R1-Distill-Llama-8B-HarmBench.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "finished the second 4 DeepSeek-R1-Distill-Llama-8B"

CUDA_VISIBLE_DEVICES=0 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=CatQA \
  --batch_size=16 \
  --apply_chat_template=False \
  --model_name=deepseek-ai/DeepSeek-R1-Distill-Llama-8B > logs_bash/plan_injection-v2-DeepSeek-R1-Distill-Llama-8B-CatQA.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=16 \
  --apply_chat_template=False \
  --model_name=deepseek-ai/DeepSeek-R1-Distill-Llama-8B > logs_bash/plan_injection-v2-DeepSeek-R1-Distill-Llama-8B-JBBHarmfulBehaviors.log 2>&1 &


