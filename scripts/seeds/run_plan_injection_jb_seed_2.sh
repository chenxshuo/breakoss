
SEED=28
pids=()
CUDA_VISIBLE_DEVICES=1 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=StrongReject \
  --batch_size=10 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-20b > logs_bash/seed/seed-$SEED-plan_injection-v2-gpt-oss-20b-strongReject.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "finished policy strongreject"
pids=()
CUDA_VISIBLE_DEVICES=1 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --batch_size=10 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-20b > logs_bash/seed/seed-$SEED-plan_injection-v2-gpt-oss-20b-HarmfulBehaviors.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "finished policy harmfulbehaviors"
pids=()
CUDA_VISIBLE_DEVICES=1 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=HarmBench \
  --batch_size=10 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-20b > logs_bash/seed/seed-$SEED-plan_injection-v2-gpt-oss-20b-HarmBench.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "finished policy harmbench"
pids=()
CUDA_VISIBLE_DEVICES=1 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=CatQA \
  --batch_size=10 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-20b > logs_bash/seed/seed-$SEED-plan_injection-v2-gpt-oss-20b-CatQA.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished policy catqa"
pids=()
CUDA_VISIBLE_DEVICES=1 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-20b > logs_bash/seed/seed-$SEED-plan_injection-v2-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "finished policy jbb"
exit 0

## 120b
## fake-overrefusal + CoT Bypass]
#pids=()
#CUDA_VISIBLE_DEVICES=1,1,2,3 nohup python examples/5_plan_injection.py main_inference \
#  --dataset_name=StrongReject \
#  --batch_size=4 \
#  --model_name=openai/gpt-oss-120b > logs_bash/plan_injection-gpt-oss-120b-strongReject.log 2>&1 &
#pids+=($!)
#for pid in "${pids[@]}"; do
#    wait $pid
#done
#echo "finished on StrongReject"
#
#
#pids=()
#CUDA_VISIBLE_DEVICES=1,1,2,3 nohup python examples/5_plan_injection.py main_inference \
#  --dataset_name=HarmfulBehaviors \
#  --batch_size=4 \
#  --model_name=openai/gpt-oss-120b > logs_bash/plan_injection-gpt-oss-120b-HarmfulBehaviors.log 2>&1 &
#pids+=($!)
#for pid in "${pids[@]}"; do
#    wait $pid
#done
#echo "finished on HarmfulBehaviors"
#
#
#pids=()
#CUDA_VISIBLE_DEVICES=1,1,2,3 nohup python examples/5_plan_injection.py main_inference \
#  --dataset_name=HarmBench \
#  --batch_size=4 \
#  --model_name=openai/gpt-oss-120b > logs_bash/plan_injection-gpt-oss-120b-HarmBench.log 2>&1 &
#pids+=($!)
#for pid in "${pids[@]}"; do
#    wait $pid
#done
#echo "finished on HarmBench"
#
#pids=()
#CUDA_VISIBLE_DEVICES=1,1,2,3 nohup python examples/5_plan_injection.py main_inference \
#  --dataset_name=CatQA \
#  --batch_size=4 \
#  --model_name=openai/gpt-oss-120b > logs_bash/plan_injection-gpt-oss-120b-CatQA.log 2>&1 &
#pids+=($!)
#for pid in "${pids[@]}"; do
#    wait $pid
#done
#echo "finished on CatQA"
#
#pids=()
#CUDA_VISIBLE_DEVICES=1,1,2,3 nohup python examples/5_plan_injection.py main_inference \
#  --dataset_name=JBBHarmfulBehaviors \
#  --batch_size=4 \
#  --model_name=openai/gpt-oss-120b > logs_bash/plan_injection-gpt-oss-120b-JBBHarmfulBehaviors.log 2>&1 &
#pids+=($!)
#for pid in "${pids[@]}"; do
#    wait $pid
#done
#echo "finished on JBBHarmfulBehaviors"
#
#
#

