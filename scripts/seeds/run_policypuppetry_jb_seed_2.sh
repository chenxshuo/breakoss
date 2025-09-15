SEED=28
pids=()
CUDA_VISIBLE_DEVICES=3 nohup python examples/baseline_policy_puppetry.py main_inference \
  --dataset_name=StrongReject \
  --batch_size=10 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-20b > logs_bash/seed/seed-$SEED-baseline_policy_puppetry-gpt-oss-20b-strongReject.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "finished policy strongreject"
pids=()
CUDA_VISIBLE_DEVICES=3 nohup python examples/baseline_policy_puppetry.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --batch_size=10 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-20b > logs_bash/seed/seed-$SEED-baseline_policy_puppetry-gpt-oss-20b-HarmfulBehaviors.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "finished policy harmfulbehaviors"
pids=()
CUDA_VISIBLE_DEVICES=3 nohup python examples/baseline_policy_puppetry.py main_inference \
  --dataset_name=HarmBench \
  --batch_size=10 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-20b > logs_bash/seed/seed-$SEED-baseline_policy_puppetry-gpt-oss-20b-HarmBench.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "finished policy harmbench"
pids=()
CUDA_VISIBLE_DEVICES=3 nohup python examples/baseline_policy_puppetry.py main_inference \
  --dataset_name=CatQA \
  --batch_size=10 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-20b > logs_bash/seed/seed-$SEED-baseline_policy_puppetry-gpt-oss-20b-CatQA.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "finished policy catqa"
pids=()
CUDA_VISIBLE_DEVICES=3 nohup python examples/baseline_policy_puppetry.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --seed=$SEED \
  --model_name=openai/gpt-oss-20b > logs_bash/seed/seed-$SEED-baseline_policy_puppetry-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "finished policy jbb"


#
######re eval on refusal words evaluator
#
#python examples/baseline_policy_puppetry.py eval_on_refusal_words \
#  --record_dir=breakoss_logs/openai_gpt-oss-20b/harmful_behaviors_gcg_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-12_16-50-20
#
#python examples/baseline_policy_puppetry.py eval_on_refusal_words \
#  --record_dir=breakoss_logs/openai_gpt-oss-20b/harmbench_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-12_16-50-19
#
#python examples/baseline_policy_puppetry.py eval_on_refusal_words \
#  --record_dir=breakoss_logs/openai_gpt-oss-20b/catqa_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-12_16-50-40
#
#python examples/baseline_policy_puppetry.py eval_on_refusal_words \
#  --record_dir=breakoss_logs/openai_gpt-oss-20b/jbb_harmful_behaviors_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-12_16-50-40/
#
#python examples/baseline_policy_puppetry.py eval_on_refusal_words \
#  --record_dir=breakoss_logs/openai_gpt-oss-20b/strong_reject_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-12_11-27-29/
#
#
#python examples/baseline_policy_puppetry.py eval_on_refusal_words \
#  --record_dir=breakoss_logs/openai_gpt-oss-20b/harmful_behaviors_gcg_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-12_11-27-29
#
#
#python examples/baseline_policy_puppetry.py eval_on_refusal_words \
#  --record_dir=breakoss_logs/openai_gpt-oss-20b/harmbench_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-12_11-27-29
#
#python examples/baseline_policy_puppetry.py eval_on_refusal_words \
#  --record_dir=breakoss_logs/openai_gpt-oss-20b/catqa_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-12_11-27-30
#
#
#python examples/baseline_policy_puppetry.py eval_on_refusal_words \
#  --record_dir=breakoss_logs/openai_gpt-oss-20b/strong_reject_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-11_23-16-11
#
#
####
##
#python examples/baseline_policy_puppetry.py eval_on_refusal_words \
#  --record_dir=breakoss_logs/openai_gpt-oss-120b/strong_reject_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-12_22-39-11
#
#python examples/baseline_policy_puppetry.py eval_on_refusal_words \
#  --record_dir=breakoss_logs/openai_gpt-oss-120b/harmful_behaviors_gcg_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-13_00-07-38
#
#python examples/baseline_policy_puppetry.py eval_on_refusal_words \
#  --record_dir=breakoss_logs/openai_gpt-oss-120b/harmbench_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-13_02-40-33
#
#python examples/baseline_policy_puppetry.py eval_on_refusal_words \
#  --record_dir=breakoss_logs/openai_gpt-oss-120b/catqa_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-13_06-03-02/
#
#
#python examples/baseline_policy_puppetry.py eval_on_refusal_words \
#  --record_dir=breakoss_logs/openai_gpt-oss-120b/jbb_harmful_behaviors_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-12_22-41-00
#
#
#
