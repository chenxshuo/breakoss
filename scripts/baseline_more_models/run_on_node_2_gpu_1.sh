
# policy puppetry on qwen
pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/baseline_policy_puppetry.py main_inference \
  --dataset_name=StrongReject \
  --batch_size=16 \
  --model_name=Qwen/Qwen3-4B-Thinking-2507 \
   > logs_bash/baseline_policy_puppetry-Qwen3-4B-Thinking-2507-StrongReject.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/baseline_policy_puppetry.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --batch_size=16 \
  --model_name=Qwen/Qwen3-4B-Thinking-2507 \
   > logs_bash/baseline_policy_puppetry-Qwen3-4B-Thinking-2507-HarmfulBehaviors.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/baseline_policy_puppetry.py main_inference \
  --dataset_name=HarmBench \
  --batch_size=16 \
  --model_name=Qwen/Qwen3-4B-Thinking-2507 \
   > logs_bash/baseline_policy_puppetry-Qwen3-4B-Thinking-2507-HarmBench.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/baseline_policy_puppetry.py main_inference \
  --dataset_name=CatQA \
  --batch_size=16 \
  --model_name=Qwen/Qwen3-4B-Thinking-2507 \
   > logs_bash/baseline_policy_puppetry-Qwen3-4B-Thinking-2507-CatQA.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/baseline_policy_puppetry.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=16 \
  --model_name=Qwen/Qwen3-4B-Thinking-2507 \
   > logs_bash/baseline_policy_puppetry-Qwen3-4B-Thinking-2507-JBBHarmfulBehaviors.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done



