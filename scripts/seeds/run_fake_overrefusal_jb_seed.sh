# fake-overrefusal + CoT Bypass

SEED=16
# logs_bash/seed/seed-16-fake_overrefusal_and_cot_bypass-gpt-oss-20b-strongReject.log

CUDA_VISIBLE_DEVICES=0 python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=StrongReject \
  --batch_size=10 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-20b \
  --failure_cases_dir=breakoss_logs/openai_gpt-oss-20b/strong_reject_[0_-1]/CoTBypass/pure_inference_no_eval/seed_42/2025-09-06_12-33-33 > logs_bash/seed/seed-$SEED-fake_overrefusal_and_cot_bypass-gpt-oss-20b-strongReject.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --batch_size=10 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-20b \
  --failure_cases_dir=breakoss_logs/openai_gpt-oss-20b/harmful_behaviors_gcg_[0_-1]/CoTBypass/pure_inference_no_eval/seed_42/2025-09-05_19-51-07 > logs_bash/seed/seed-$SEED-fake_overrefusal_and_cot_bypass-gpt-oss-20b-HarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=HarmBench \
  --batch_size=10 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-20b \
  --failure_cases_dir=breakoss_logs/openai_gpt-oss-20b/harmbench_[0_-1]/CoTBypass/pure_inference_no_eval/seed_42/2025-09-06_12-34-04/ > logs_bash/seed/seed-$SEED-fake_overrefusal_and_cot_bypass-gpt-oss-20b-HarmBench.log 2>&1 &

pids=()
CUDA_VISIBLE_DEVICES=3 python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=CatQA \
  --batch_size=10 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-20b \
  --failure_cases_dir=breakoss_logs/openai_gpt-oss-20b/catqa_[0_-1]/CoTBypass/pure_inference_no_eval/seed_42/2025-09-06_12-35-04 > logs_bash/seed/seed-$SEED-fake_overrefusal_and_cot_bypass-gpt-oss-20b-CatQA.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done


CUDA_VISIBLE_DEVICES=3 python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-20b \
  --failure_cases_dir=breakoss_logs/openai_gpt-oss-20b/jbb_harmful_behaviors_[0_-1]/CoTBypass/pure_inference_no_eval/seed_42/2025-09-06_12-35-04 > logs_bash/seed/seed-$SEED-fake_overrefusal_and_cot_bypass-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &

exit 0


# 120b
# fake-overrefusal + CoT Bypass]
#pids=()
#CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/2_fake_overrefusal.py main_inference \
#  --dataset_name=StrongReject \
#  --batch_size=4 \
#  --model_name=openai/gpt-oss-120b \
#  --failure_cases_dir=breakoss_logs/openai_gpt-oss-120b/strong_reject_[0_-1]/CoTBypass/pure_inference_no_eval/seed_42/2025-09-06_21-56-27 > logs_bash/fake_overrefusal_and_cot_bypass-gpt-oss-120b-strongReject.log 2>&1 &
#pids+=($!)
#for pid in "${pids[@]}"; do
#    wait $pid
#done
#echo "finished on StrongReject"
#
#
#pids=()
#CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/2_fake_overrefusal.py main_inference \
#  --dataset_name=HarmfulBehaviors \
#  --batch_size=4 \
#  --model_name=openai/gpt-oss-120b \
#  --failure_cases_dir=breakoss_logs/openai_gpt-oss-120b/harmful_behaviors_gcg_[0_-1]/CoTBypass/pure_inference_no_eval/seed_42/2025-09-07_00-21-42 > logs_bash/fake_overrefusal_and_cot_bypass-gpt-oss-120b-HarmfulBehaviors.log 2>&1 &
#pids+=($!)
#for pid in "${pids[@]}"; do
#    wait $pid
#done
#echo "finished on HarmfulBehaviors"
#
#
#pids=()
#CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/2_fake_overrefusal.py main_inference \
#  --dataset_name=HarmBench \
#  --batch_size=4 \
#  --model_name=openai/gpt-oss-120b \
#  --failure_cases_dir=breakoss_logs/openai_gpt-oss-120b/harmbench_[0_-1]/CoTBypass/pure_inference_no_eval/seed_42/2025-09-07_03-47-45 > logs_bash/fake_overrefusal_and_cot_bypass-gpt-oss-120b-HarmBench.log 2>&1 &
#pids+=($!)
#for pid in "${pids[@]}"; do
#    wait $pid
#done
#echo "finished on HarmBench"

pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=CatQA \
  --batch_size=4 \
  --model_name=openai/gpt-oss-120b \
  --failure_cases_dir=breakoss_logs/openai_gpt-oss-120b/catqa_[0_-1]/CoTBypass/pure_inference_no_eval/seed_42/2025-09-07_06-18-18/ > logs_bash/fake_overrefusal_and_cot_bypass-gpt-oss-120b-CatQA.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on CatQA"

pids=()
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=4 \
  --model_name=openai/gpt-oss-120b \
  --failure_cases_dir=breakoss_logs/openai_gpt-oss-120b/jbb_harmful_behaviors_[0_-1]/CoTBypass/pure_inference_no_eval/seed_42/2025-09-07_09-57-00 > logs_bash/fake_overrefusal_and_cot_bypass-gpt-oss-120b-JBBHarmfulBehaviors.log 2>&1 &
pids+=($!)
for pid in "${pids[@]}"; do
    wait $pid
done
echo "finished on JBBHarmfulBehaviors"




