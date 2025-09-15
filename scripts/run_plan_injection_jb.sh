
CUDA_VISIBLE_DEVICES=0 python examples/5_plan_injection.py main_inference \
  --model_name=Qwen/Qwen3-4B-Thinking-2507 \
  --dataset_name=StrongReject \
  --batch_size=10


CUDA_VISIBLE_DEVICES=0 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=StrongReject \
  --batch_size=32 \
  --model_name=Qwen/Qwen3-4B-Thinking-2507  > logs_bash/plan_injection-v2-qwen3-4b-thinking-2507-strongReject.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --batch_size=32 \
  --model_name=Qwen/Qwen3-4B-Thinking-2507  > logs_bash/plan_injection-v2-qwen3-4b-thinking-2507-HarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=HarmBench \
  --batch_size=32 \
  --model_name=Qwen/Qwen3-4B-Thinking-2507 > logs_bash/plan_injection-v2-qwen3-4b-thinking-2507-HarmBench.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=CatQA \
  --batch_size=32 \
  --model_name=Qwen/Qwen3-4B-Thinking-2507 > logs_bash/plan_injection-v2-qwen3-4b-thinking-2507-CatQA.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=32 \
  --model_name=Qwen/Qwen3-4B-Thinking-2507 > logs_bash/plan_injection-v2-qwen3-4b-thinking-2507-JBBHarmfulBehaviors.log 2>&1 &





CUDA_VISIBLE_DEVICES=0 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=StrongReject \
  --batch_size=16 \
  --model_name=Qwen/Qwen3-30B-A3B-Thinking-2507  > logs_bash/plan_injection-v2-qwen3-30b-thinking-2507-strongReject.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --batch_size=16 \
  --model_name=Qwen/Qwen3-30B-A3B-Thinking-2507  > logs_bash/plan_injection-v2-qwen3-30b-thinking-2507-HarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=HarmBench \
  --batch_size=16 \
  --model_name=Qwen/Qwen3-30B-A3B-Thinking-2507 > logs_bash/plan_injection-v2-qwen3-30b-thinking-2507-HarmBench.log 2>&1 &

pids=()
CUDA_VISIBLE_DEVICES=3 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=16 \
  --model_name=Qwen/Qwen3-30B-A3B-Thinking-2507 > logs_bash/plan_injection-v2-qwen3-30b-thinking-2507-JBBHarmfulBehaviors.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done

CUDA_VISIBLE_DEVICES=3 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=CatQA \
  --batch_size=16 \
  --model_name=Qwen/Qwen3-30B-A3B-Thinking-2507 > logs_bash/plan_injection-v2-qwen3-30b-thinking-2507-CatQA.log 2>&1 &



CUDA_VISIBLE_DEVICES=1 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=StrongReject \
  --batch_size=16 \
  --model_name=microsoft/Phi-4-reasoning-plus  > logs_bash/plan_injection-v2-Phi-4-reasoning-plus-2507-strongReject.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --batch_size=16 \
  --model_name=microsoft/Phi-4-reasoning-plus > logs_bash/plan_injection-v2-Phi-4-reasoning-plus-HarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=HarmBench \
  --batch_size=16 \
  --model_name=microsoft/Phi-4-reasoning-plus > logs_bash/plan_injection-v2-Phi-4-reasoning-plus-HarmBench.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=CatQA \
  --batch_size=16 \
  --model_name=microsoft/Phi-4-reasoning-plus > logs_bash/plan_injection-v2-Phi-4-reasoning-plus-CatQA.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=16 \
  --model_name=microsoft/Phi-4-reasoning-plus > logs_bash/plan_injection-v2-Phi-4-reasoning-plus-JBBHarmfulBehaviors.log 2>&1 &










CUDA_VISIBLE_DEVICES=0 python examples/5_plan_injection.py main_inference \
  --model_name=Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --dataset_name=StrongReject \
  --batch_size=10





#CUDA_VISIBLE_DEVICES=0 nohup python examples/5_plan_injection.py main_inference \
#  --dataset_name=StrongReject \
#  --batch_size=10 \
#  --model_name=openai/gpt-oss-20b > logs_bash/plan_injection-v2-gpt-oss-20b-strongReject.log 2>&1 &
#
#CUDA_VISIBLE_DEVICES=1 nohup python examples/5_plan_injection.py main_inference \
#  --dataset_name=HarmfulBehaviors \
#  --batch_size=10 \
#  --model_name=openai/gpt-oss-20b > logs_bash/plan_injection-v2-gpt-oss-20b-HarmfulBehaviors.log 2>&1 &
#
#CUDA_VISIBLE_DEVICES=2 nohup python examples/5_plan_injection.py main_inference \
#  --dataset_name=HarmBench \
#  --batch_size=10 \
#  --model_name=openai/gpt-oss-20b > logs_bash/plan_injection-v2-gpt-oss-20b-HarmBench.log 2>&1 &

#pids=()
#CUDA_VISIBLE_DEVICES=3 nohup python examples/5_plan_injection.py main_inference \
#  --dataset_name=CatQA \
#  --batch_size=10 \
#  --model_name=openai/gpt-oss-20b > logs_bash/plan_injection-v2-gpt-oss-20b-CatQA.log 2>&1 &
#pids+=($!)
#for pid in "${pids[@]}"; do
#    wait $pid
#done
#
#CUDA_VISIBLE_DEVICES=3 nohup python examples/5_plan_injection.py main_inference \
#  --dataset_name=JBBHarmfulBehaviors \
#  --batch_size=10 \
#  --model_name=openai/gpt-oss-20b > logs_bash/plan_injection-v2-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &
#
#
#
#exit 0

# 120b
# fake-overrefusal + CoT Bypass]
pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=StrongReject \
  --batch_size=4 \
  --model_name=openai/gpt-oss-120b > logs_bash/plan_injection-gpt-oss-120b-strongReject.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on StrongReject"


pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --batch_size=4 \
  --model_name=openai/gpt-oss-120b > logs_bash/plan_injection-gpt-oss-120b-HarmfulBehaviors.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on HarmfulBehaviors"


pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=HarmBench \
  --batch_size=4 \
  --model_name=openai/gpt-oss-120b > logs_bash/plan_injection-gpt-oss-120b-HarmBench.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on HarmBench"

pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=CatQA \
  --batch_size=4 \
  --model_name=openai/gpt-oss-120b > logs_bash/plan_injection-gpt-oss-120b-CatQA.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on CatQA"

pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=4 \
  --model_name=openai/gpt-oss-120b > logs_bash/plan_injection-gpt-oss-120b-JBBHarmfulBehaviors.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on JBBHarmfulBehaviors"




