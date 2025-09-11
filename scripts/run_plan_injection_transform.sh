

#CUDA_VISIBLE_DEVICES=0 python examples/5_plan_injection.py transform_original_prompts \
#  --dataset_name=StrongReject \
#  --target_model_name=openai/gpt-oss-20b

#pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/5_plan_injection.py transform_original_prompts \
  --dataset_name=StrongReject \
  --target_model_name=openai/gpt-oss-20b > logs_bash/transform-plan-injection-gpt-oss-20b-StrongReject.log 2>&1 &

#pids+=($!)
CUDA_VISIBLE_DEVICES=1 nohup python examples/5_plan_injection.py transform_original_prompts \
  --dataset_name=HarmfulBehaviors \
  --target_model_name=openai/gpt-oss-20b   > logs_bash/transform-plan-injection-gpt-oss-20b-HarmfulBehaviors.log 2>&1 &
#pids+=($!)
#for ((i=${#pids[@]}; i>0; i--)) ; do
#    wait -n
#done
#echo "Finished Transform Intent Hijack StrongReject, HarmfulBehaviors"

#pids=()
CUDA_VISIBLE_DEVICES=2 nohup python examples/5_plan_injection.py transform_original_prompts \
  --dataset_name=HarmBench \
  --target_model_name=openai/gpt-oss-20b   > logs_bash/transform-plan-injection-gpt-oss-20b-HarmBench.log 2>&1 &
#pids+=($!)
pids=()
CUDA_VISIBLE_DEVICES=3 nohup python examples/5_plan_injection.py transform_original_prompts  \
  --dataset_name=JBBHarmfulBehaviors \
  --target_model_name=openai/gpt-oss-20b > logs_bash/transform-plan-injection-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done

pids=()
CUDA_VISIBLE_DEVICES=3 nohup python examples/5_plan_injection.py transform_original_prompts \
  --dataset_name=CatQA \
  --target_model_name=openai/gpt-oss-20b   > logs_bash/transform-plan-injection-gpt-oss-20b-CatQA.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done

#echo "Finished Transform Intent Hijack JBBHarmfulBehaviors"
#exit 0

#
#
#CUDA_VISIBLE_DEVICES=0 nohup python examples/5_plan_injection.py transform_once \
#  --dataset_name=StrongReject > logs_bash/transform-plan-injection-v2-gpt-oss-20b-StrongReject.log 2>&1 &
#
#CUDA_VISIBLE_DEVICES=1 nohup python examples/5_plan_injection.py transform_once \
#  --dataset_name=HarmfulBehaviors  > logs_bash/transform-plan-injection-v2-gpt-oss-20b-HarmfulBehaviors.log 2>&1 &
#
#CUDA_VISIBLE_DEVICES=2 nohup python examples/5_plan_injection.py transform_once \
#  --dataset_name=HarmBench  > logs_bash/transform-plan-injection-v2-gpt-oss-20b-HarmBench.log 2>&1 &
#
#CUDA_VISIBLE_DEVICES=3 nohup python examples/5_plan_injection.py transform_once \
#  --dataset_name=CatQA  > logs_bash/transform-plan-injection-v2-gpt-oss-20b-CatQA.log 2>&1 &
#
#CUDA_VISIBLE_DEVICES=0 nohup  python examples/5_plan_injection.py transform_once \
#  --dataset_name=JBBHarmfulBehaviors  > logs_bash/transform-plan-injection-v2-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &
