
SEED=28
pids=()
CUDA_VISIBLE_DEVICES=3 nohup python examples/baseline_autoran.py main_inference \
  --dataset_name=StrongReject \
  --batch_size=10 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-20b > logs_bash/seed/seed-$SEED-baseline_autoran-gpt-oss-20b-strongReject.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "finished autoran strongreject"
pids=()
CUDA_VISIBLE_DEVICES=3 nohup python examples/baseline_autoran.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --batch_size=10 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-20b > logs_bash/seed/seed-$SEED-baseline_autoran-gpt-oss-20b-HarmfulBehaviors.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "finished autoran harmfulbehaviors"
pids=()
CUDA_VISIBLE_DEVICES=3 nohup python examples/baseline_autoran.py main_inference \
  --dataset_name=HarmBench \
  --batch_size=10 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-20b > logs_bash/seed/seed-$SEED-baseline_autoran-gpt-oss-20b-HarmBench.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "finished autoran harmbench"
pids=()
CUDA_VISIBLE_DEVICES=3 nohup python examples/baseline_autoran.py main_inference \
  --dataset_name=CatQA \
  --batch_size=10 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-20b > logs_bash/seed/seed-$SEED-baseline_autoran-gpt-oss-20b-CatQA.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "finished autoran catqa"
pids=()

CUDA_VISIBLE_DEVICES=3 nohup python examples/baseline_autoran.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-20b > logs_bash/seed/seed-$SEED-baseline_autoran-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "finished autoran jbb"
