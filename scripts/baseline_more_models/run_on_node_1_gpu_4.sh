####### policy puppetry
pids=()
CUDA_VISIBLE_DEVICES=3 nohup python examples/baseline_policy_puppetry.py main_inference \
  --dataset_name=StrongReject \
  --batch_size=16 \
  --model_name=microsoft/Phi-4-reasoning-plus \
   > logs_bash/baseline_policy_puppetry-phi-4-reasnoning-plus-StrongReject.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
pids=()
CUDA_VISIBLE_DEVICES=3 nohup python examples/baseline_policy_puppetry.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --batch_size=16 \
  --model_name=microsoft/Phi-4-reasoning-plus \
   > logs_bash/baseline_policy_puppetry-phi-4-reasnoning-plus-HarmfulBehaviors.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
pids=()
CUDA_VISIBLE_DEVICES=3 nohup python examples/baseline_policy_puppetry.py main_inference \
  --dataset_name=HarmBench \
  --batch_size=16 \
  --model_name=microsoft/Phi-4-reasoning-plus \
   > logs_bash/baseline_policy_puppetry-phi-4-reasnoning-plus-HarmBench.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
pids=()
CUDA_VISIBLE_DEVICES=3 nohup python examples/baseline_policy_puppetry.py main_inference \
  --dataset_name=CatQA \
  --batch_size=16 \
  --model_name=microsoft/Phi-4-reasoning-plus \
   > logs_bash/baseline_policy_puppetry-phi-4-reasnoning-plus-CatQA.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
pids=()
CUDA_VISIBLE_DEVICES=3 nohup python examples/baseline_policy_puppetry.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=16 \
  --model_name=microsoft/Phi-4-reasoning-plus \
   > logs_bash/baseline_policy_puppetry-phi-4-reasnoning-plus-JBBHarmfulBehaviors.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
