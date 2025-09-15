SEED=16

pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/baseline_policy_puppetry.py main_inference \
  --dataset_name=StrongReject \
  --batch_size=4 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-120b > logs_bash/seed/seed-$SEED-120b-baseline_policy_puppetry-gpt-oss-120b-strongReject.log 2>&1 &
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
  --model_name=openai/gpt-oss-120b > logs_bash/seed/seed-$SEED-120b-baseline_policy_puppetry-gpt-oss-120b-HarmfulBehaviors.log 2>&1 &
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
  --model_name=openai/gpt-oss-120b > logs_bash/seed/seed-$SEED-120b-baseline_policy_puppetry-gpt-oss-120b-HarmBench.log 2>&1 &
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
  --model_name=openai/gpt-oss-120b > logs_bash/seed/seed-$SEED-120b-baseline_policy_puppetry-gpt-oss-120b-CatQA.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on CatQA"

pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/baseline_policy_puppetry.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=4 \
  --model_name=openai/gpt-oss-120b > logs_bash/seed/seed-$SEED-120b-baseline_policy_puppetry-gpt-oss-120b-JBBHarmfulBehaviors.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on JBBHarmfulBehaviors"

############## policy puppetry ends ############


############## HCoT Starts ############


pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/baseline_hcot.py main_inference \
  --dataset_name=StrongReject \
  --batch_size=4 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-120b > logs_bash/seed/seed-$SEED-120b-baseline_hcot-gpt-oss-120b-strongReject.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on StrongReject"


pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/baseline_hcot.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --batch_size=4 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-120b > logs_bash/seed/seed-$SEED-120b-baseline_hcot-gpt-oss-120b-HarmfulBehaviors.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on HarmfulBehaviors"


pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/baseline_hcot.py main_inference \
  --dataset_name=HarmBench \
  --batch_size=4 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-120b > logs_bash/seed/seed-$SEED-120b-baseline_hcot-gpt-oss-120b-HarmBench.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on HarmBench"



pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/baseline_hcot.py main_inference \
  --dataset_name=CatQA \
  --batch_size=4 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-120b > logs_bash/seed/seed-$SEED-120b-baseline_hcot-gpt-oss-120b-CatQA.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on CatQA"

pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/baseline_hcot.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=4 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-120b > logs_bash/seed/seed-$SEED-120b-baseline_hcot-gpt-oss-120b-JBBHarmfulBehaviors.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on JBBHarmfulBehaviors"

######### HCoT Ends #############

######## AutoRAN starts ##########
pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/baseline_autoran.py main_inference \
  --dataset_name=StrongReject \
  --batch_size=4 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-120b > logs_bash/seed/seed-$SEED-120b-baseline_autoran-gpt-oss-120b-strongReject.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on StrongReject"


pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/baseline_autoran.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --batch_size=4 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-120b > logs_bash/seed/seed-$SEED-120b-baseline_autoran-gpt-oss-120b-HarmfulBehaviors.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on HarmfulBehaviors"



pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/baseline_autoran.py main_inference \
  --dataset_name=HarmBench \
  --batch_size=4 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-120b > logs_bash/seed/seed-$SEED-120b-baseline_autoran-gpt-oss-120b-HarmBench.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on HarmBench"

pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/baseline_autoran.py main_inference \
  --dataset_name=CatQA \
  --batch_size=4 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-120b > logs_bash/seed/seed-$SEED-120b-baseline_autoran-gpt-oss-120b-CatQA.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on CatQA"

pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/baseline_autoran.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=4 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-120b > logs_bash/seed/seed-$SEED-120b-baseline_autoran-gpt-oss-120b-JBBHarmfulBehaviors.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on JBBHarmfulBehaviors"




