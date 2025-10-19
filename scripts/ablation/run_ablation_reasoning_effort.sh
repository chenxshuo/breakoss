


CUDA_VISIBLE_DEVICES=0 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=16 \
  --dataset_name=JBBHarmfulBehaviors \
  --reasoning_effort=low \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-reasoning_effort-low-cot_bypass-gpt-oss-20b-JBBHarmfulBehaviorstQA.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=16 \
  --dataset_name=JBBHarmfulBehaviors \
  --reasoning_effort=high \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-reasoning_effort-high-cot_bypass-gpt-oss-20b-JBBHarmfulBehaviorstQA.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --reasoning_effort=low \
  --model_name=openai/gpt-oss-20b \
  --failure_cases_dir=breakoss_logs/openai_gpt-oss-20b/jbb_harmful_behaviors_[0_-1]/CoTBypass/pure_inference_no_eval/seed_42/2025-09-06_12-35-04 > logs_bash/ablation-reasoning_effort-low-fake_overrefusal_and_cot_bypass-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --reasoning_effort=high \
  --model_name=openai/gpt-oss-20b \
  --failure_cases_dir=breakoss_logs/openai_gpt-oss-20b/jbb_harmful_behaviors_[0_-1]/CoTBypass/pure_inference_no_eval/seed_42/2025-09-06_12-35-04 > logs_bash/ablation-reasoning_effort-high-fake_overrefusal_and_cot_bypass-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --reasoning_effort=low \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-reasoning_effort-low-plan_injection-v2-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --reasoning_effort=high \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-reasoning_effort-high-plan_injection-v2-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 nohup python examples/3_gcg.py ablation_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --reasoning_effort=low \
  --record_dir=breakoss_logs/openai_gpt-oss-20b/JBBHarmfulBehaviors_gcg_evaluation > logs_bash/ablation-reasoning_effort-low-gcg-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python examples/3_gcg.py ablation_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --reasoning_effort=high \
  --record_dir=breakoss_logs/openai_gpt-oss-20b/JBBHarmfulBehaviors_gcg_evaluation > logs_bash/ablation-reasoning_effort-high-gcg-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &