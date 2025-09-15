SEED=16
pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/baseline_policy_puppetry.py main_inference \
  --dataset_name=StrongReject \
  --batch_size=4 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-120b > logs_bash/seed-$SEED-120b-baseline_policy_puppetry-gpt-oss-120b-strongReject.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on StrongReject"


pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/baseline_policy_puppetry.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --batch_size=4 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-120b > logs_bash/baseline_policy_puppetry-gpt-oss-120b-HarmfulBehaviors.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on HarmfulBehaviors"


pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/baseline_policy_puppetry.py main_inference \
  --dataset_name=HarmBench \
  --batch_size=4 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-120b > logs_bash/baseline_policy_puppetry-gpt-oss-120b-HarmBench.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on HarmBench"

pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/baseline_policy_puppetry.py main_inference \
  --dataset_name=CatQA \
  --batch_size=4 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-120b > logs_bash/baseline_policy_puppetry-gpt-oss-120b-CatQA.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on CatQA"
