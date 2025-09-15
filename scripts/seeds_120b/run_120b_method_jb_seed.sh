
SEED=16

pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=4 --seed=$SEED  \
  --dataset_name=StrongReject \
  --model_name=openai/gpt-oss-120b > logs_bash/seed/seed-$SEED-120b-cot_bypass-gpt-oss-120b-StrongReject.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "Finished StrongReject with gpt-oss-120b"

pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=4 --seed=$SEED  \
  --dataset_name=HarmfulBehaviors \
  --model_name=openai/gpt-oss-120b > logs_bash/seed/seed-$SEED-120b-cot_bypass-gpt-oss-120b-HarmfulBehaviors.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "Finished HarmfulBehaviors with gpt-oss-120b"

pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=4 --seed=$SEED  \
  --dataset_name=HarmBench \
  --model_name=openai/gpt-oss-120b > logs_bash/seed/seed-$SEED-120b-cot_bypass-gpt-oss-120b-HarmBench.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "Finished HarmBench with gpt-oss-120b"

pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3  nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=4 --seed=$SEED  \
  --dataset_name=CatQA \
  --model_name=openai/gpt-oss-120b > logs_bash/seed/seed-$SEED-120b-cot_bypass-gpt-oss-120b-CatQA.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "Finished CatQA with gpt-oss-120b"

pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=4 --seed=$SEED  \
  --dataset_name=JBBHarmfulBehaviors \
  --model_name=openai/gpt-oss-120b > logs_bash/seed/seed-$SEED-120b-cot_bypass-gpt-oss-120b-JBBHarmfulBehaviorstQA.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "Finished JBBHarmfulBehaviors with gpt-oss-120b"

pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=StrongReject \
  --batch_size=4 --seed=$SEED \
  --model_name=openai/gpt-oss-120b \
  --failure_cases_dir=breakoss_logs/openai_gpt-oss-120b/strong_reject_[0_-1]/CoTBypass/pure_inference_no_eval/seed_42/2025-09-06_21-56-27 > logs_bash/seed/seed-$SEED-120b-fake_overrefusal_and_cot_bypass-gpt-oss-120b-strongReject.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on StrongReject"


pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --batch_size=4 --seed=$SEED \
  --model_name=openai/gpt-oss-120b \
  --failure_cases_dir=breakoss_logs/openai_gpt-oss-120b/harmful_behaviors_gcg_[0_-1]/CoTBypass/pure_inference_no_eval/seed_42/2025-09-07_00-21-42 > logs_bash/seed/seed-$SEED-120b-fake_overrefusal_and_cot_bypass-gpt-oss-120b-HarmfulBehaviors.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on HarmfulBehaviors"


pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=HarmBench \
  --batch_size=4 --seed=$SEED \
  --model_name=openai/gpt-oss-120b \
  --failure_cases_dir=breakoss_logs/openai_gpt-oss-120b/harmbench_[0_-1]/CoTBypass/pure_inference_no_eval/seed_42/2025-09-07_03-47-45 > logs_bash/seed/seed-$SEED-120b-fake_overrefusal_and_cot_bypass-gpt-oss-120b-HarmBench.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on HarmBench"

pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=CatQA \
  --batch_size=4 --seed=$SEED \
  --model_name=openai/gpt-oss-120b \
  --failure_cases_dir=breakoss_logs/openai_gpt-oss-120b/catqa_[0_-1]/CoTBypass/pure_inference_no_eval/seed_42/2025-09-07_06-18-18/ > logs_bash/seed/seed-$SEED-120b-fake_overrefusal_and_cot_bypass-gpt-oss-120b-CatQA.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on CatQA"

pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=4 --seed=$SEED \
  --model_name=openai/gpt-oss-120b \
  --failure_cases_dir=breakoss_logs/openai_gpt-oss-120b/jbb_harmful_behaviors_[0_-1]/CoTBypass/pure_inference_no_eval/seed_42/2025-09-07_09-57-00 > logs_bash/seed/seed-$SEED-120b-fake_overrefusal_and_cot_bypass-gpt-oss-120b-JBBHarmfulBehaviors.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on JBBHarmfulBehaviors"



pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=StrongReject \
  --batch_size=4 --seed=$SEED \
  --model_name=openai/gpt-oss-120b > logs_bash/seed/seed-$SEED-120b-plan_injection-gpt-oss-120b-strongReject.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on StrongReject"


pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --batch_size=4 --seed=$SEED \
  --model_name=openai/gpt-oss-120b > logs_bash/seed/seed-$SEED-120b-plan_injection-gpt-oss-120b-HarmfulBehaviors.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on HarmfulBehaviors"


pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=HarmBench \
  --batch_size=4 --seed=$SEED \
  --model_name=openai/gpt-oss-120b > logs_bash/seed/seed-$SEED-120b-plan_injection-gpt-oss-120b-HarmBench.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on HarmBench"

pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=CatQA \
  --batch_size=4 --seed=$SEED \
  --model_name=openai/gpt-oss-120b > logs_bash/seed/seed-$SEED-120b-plan_injection-gpt-oss-120b-CatQA.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on CatQA"

pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=4 --seed=$SEED \
  --model_name=openai/gpt-oss-120b > logs_bash/seed/seed-$SEED-120b-plan_injection-gpt-oss-120b-JBBHarmfulBehaviors.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on JBBHarmfulBehaviors"




