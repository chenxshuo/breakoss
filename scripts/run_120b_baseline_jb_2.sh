pids=()
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python examples/baseline_policy_puppetry.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=4 \
  --model_name=openai/gpt-oss-120b > logs_bash/baseline_policy_puppetry-gpt-oss-120b-JBBHarmfulBehaviors.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on JBBHarmfulBehaviors"

############## policy puppetry ends ############


############## HCoT Starts ############


pids=()
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python examples/baseline_hcot.py main_inference \
  --dataset_name=StrongReject \
  --batch_size=4 \
  --model_name=openai/gpt-oss-120b > logs_bash/baseline_hcot-gpt-oss-120b-strongReject.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on StrongReject"


pids=()
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python examples/baseline_hcot.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --batch_size=4 \
  --model_name=openai/gpt-oss-120b > logs_bash/baseline_hcot-gpt-oss-120b-HarmfulBehaviors.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on HarmfulBehaviors"


pids=()
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python examples/baseline_hcot.py main_inference \
  --dataset_name=HarmBench \
  --batch_size=4 \
  --model_name=openai/gpt-oss-120b > logs_bash/baseline_hcot-gpt-oss-120b-HarmBench.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on HarmBench"





