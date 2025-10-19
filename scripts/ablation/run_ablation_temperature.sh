CUDA_VISIBLE_DEVICES=0 python examples/1_cot_bypass.py main_inference \
  --batch_size=32 \
  --dataset_name=JBBHarmfulBehaviors \
  --temperature=0.2 --do_sample=True \
  --model_name=openai/gpt-oss-20b


CUDA_VISIBLE_DEVICES=0 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=32 \
  --dataset_name=JBBHarmfulBehaviors \
  --temperature=0.2 --do_sample=True \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-temperature-0.2-cot_bypass-gpt-oss-20b-JBBHarmfulBehaviorstQA.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=32 \
  --dataset_name=JBBHarmfulBehaviors \
  --temperature=0.4 --do_sample=True \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-temperature-0.4-cot_bypass-gpt-oss-20b-JBBHarmfulBehaviorstQA.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=32 \
  --dataset_name=JBBHarmfulBehaviors \
  --temperature=0.6 --do_sample=True \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-temperature-0.6-cot_bypass-gpt-oss-20b-JBBHarmfulBehaviorstQA.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=32 \
  --dataset_name=JBBHarmfulBehaviors \
  --temperature=0.8 --do_sample=True \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-temperature-0.8-cot_bypass-gpt-oss-20b-JBBHarmfulBehaviorstQA.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=32 \
  --dataset_name=JBBHarmfulBehaviors \
  --temperature=1.0 --do_sample=True \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-temperature-1.0-cot_bypass-gpt-oss-20b-JBBHarmfulBehaviorstQA.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=32 \
  --dataset_name=JBBHarmfulBehaviors \
  --temperature=1.2 --do_sample=True \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-temperature-1.2-cot_bypass-gpt-oss-20b-JBBHarmfulBehaviorstQA.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=32 \
  --dataset_name=JBBHarmfulBehaviors \
  --temperature=1.4 --do_sample=True \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-temperature-1.4-cot_bypass-gpt-oss-20b-JBBHarmfulBehaviorstQA.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=32 \
  --dataset_name=JBBHarmfulBehaviors \
  --temperature=1.6 --do_sample=True \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-temperature-1.6-cot_bypass-gpt-oss-20b-JBBHarmfulBehaviorstQA.log 2>&1 &


# fake over-refusal
CUDA_VISIBLE_DEVICES=0 python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --temperature=0.2 --do_sample=True \
  --model_name=openai/gpt-oss-20b \
  --failure_cases_dir=breakoss_logs/openai_gpt-oss-20b/jbb_harmful_behaviors_[0_-1]/CoTBypass/pure_inference_no_eval/seed_42/2025-09-06_12-35-04 > logs_bash/ablation-temperature-0.2-fake_overrefusal_and_cot_bypass-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --temperature=0.4 --do_sample=True \
  --model_name=openai/gpt-oss-20b \
  --failure_cases_dir=breakoss_logs/openai_gpt-oss-20b/jbb_harmful_behaviors_[0_-1]/CoTBypass/pure_inference_no_eval/seed_42/2025-09-06_12-35-04 > logs_bash/ablation-temperature-0.4-fake_overrefusal_and_cot_bypass-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --temperature=0.6 --do_sample=True \
  --model_name=openai/gpt-oss-20b \
  --failure_cases_dir=breakoss_logs/openai_gpt-oss-20b/jbb_harmful_behaviors_[0_-1]/CoTBypass/pure_inference_no_eval/seed_42/2025-09-06_12-35-04 > logs_bash/ablation-temperature-0.6-fake_overrefusal_and_cot_bypass-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --temperature=0.8 --do_sample=True \
  --model_name=openai/gpt-oss-20b \
  --failure_cases_dir=breakoss_logs/openai_gpt-oss-20b/jbb_harmful_behaviors_[0_-1]/CoTBypass/pure_inference_no_eval/seed_42/2025-09-06_12-35-04 > logs_bash/ablation-temperature-0.8-fake_overrefusal_and_cot_bypass-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --temperature=1.0 --do_sample=True \
  --model_name=openai/gpt-oss-20b \
  --failure_cases_dir=breakoss_logs/openai_gpt-oss-20b/jbb_harmful_behaviors_[0_-1]/CoTBypass/pure_inference_no_eval/seed_42/2025-09-06_12-35-04 > logs_bash/ablation-temperature-1.0-fake_overrefusal_and_cot_bypass-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --temperature=1.2 --do_sample=True \
  --model_name=openai/gpt-oss-20b \
  --failure_cases_dir=breakoss_logs/openai_gpt-oss-20b/jbb_harmful_behaviors_[0_-1]/CoTBypass/pure_inference_no_eval/seed_42/2025-09-06_12-35-04 > logs_bash/ablation-temperature-1.2-fake_overrefusal_and_cot_bypass-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --temperature=1.4 --do_sample=True \
  --model_name=openai/gpt-oss-20b \
  --failure_cases_dir=breakoss_logs/openai_gpt-oss-20b/jbb_harmful_behaviors_[0_-1]/CoTBypass/pure_inference_no_eval/seed_42/2025-09-06_12-35-04 > logs_bash/ablation-temperature-1.4-fake_overrefusal_and_cot_bypass-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --temperature=1.6 --do_sample=True \
  --model_name=openai/gpt-oss-20b \
  --failure_cases_dir=breakoss_logs/openai_gpt-oss-20b/jbb_harmful_behaviors_[0_-1]/CoTBypass/pure_inference_no_eval/seed_42/2025-09-06_12-35-04 > logs_bash/ablation-temperature-1.6-fake_overrefusal_and_cot_bypass-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &

# plan injection

CUDA_VISIBLE_DEVICES=0 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --temperature=0.2 --do_sample=True \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-temperature-0.2-plan_injection-v2-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --temperature=0.4 --do_sample=True \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-temperature-0.4-plan_injection-v2-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --temperature=0.6 --do_sample=True \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-temperature-0.6-plan_injection-v2-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --temperature=0.8 --do_sample=True \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-temperature-0.8-plan_injection-v2-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --temperature=1.0 --do_sample=True \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-temperature-1.0-plan_injection-v2-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --temperature=1.2 --do_sample=True \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-temperature-1.2-plan_injection-v2-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --temperature=1.4 --do_sample=True \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-temperature-1.4-plan_injection-v2-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --temperature=1.6 --do_sample=True \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-temperature-1.6-plan_injection-v2-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &


# gcg
#CUDA_VISIBLE_DEVICES=0 python examples/3_gcg.py ablation_inference \
#  --dataset_name=JBBHarmfulBehaviors \
#  --batch_size=10 \
#  --temperature=0.4 --do_sample=True \
#  --record_dir=breakoss_logs/openai_gpt-oss-20b/JBBHarmfulBehaviors_gcg_evaluation

CUDA_VISIBLE_DEVICES=0 nohup python examples/3_gcg.py ablation_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --temperature=0.2 --do_sample=True \
  --record_dir=breakoss_logs/openai_gpt-oss-20b/JBBHarmfulBehaviors_gcg_evaluation > logs_bash/ablation-temperature-0.2-gcg-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/3_gcg.py ablation_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --temperature=0.4 --do_sample=True \
  --record_dir=breakoss_logs/openai_gpt-oss-20b/JBBHarmfulBehaviors_gcg_evaluation > logs_bash/ablation-temperature-0.4-gcg-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python examples/3_gcg.py ablation_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --temperature=0.6 --do_sample=True \
  --record_dir=breakoss_logs/openai_gpt-oss-20b/JBBHarmfulBehaviors_gcg_evaluation > logs_bash/ablation-temperature-0.6-gcg-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python examples/3_gcg.py ablation_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --temperature=0.8 --do_sample=True \
  --record_dir=breakoss_logs/openai_gpt-oss-20b/JBBHarmfulBehaviors_gcg_evaluation > logs_bash/ablation-temperature-0.8-gcg-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python examples/3_gcg.py ablation_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --temperature=1.0 --do_sample=True \
  --record_dir=breakoss_logs/openai_gpt-oss-20b/JBBHarmfulBehaviors_gcg_evaluation > logs_bash/ablation-temperature-1.0-gcg-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/3_gcg.py ablation_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --temperature=1.2 --do_sample=True \
  --record_dir=breakoss_logs/openai_gpt-oss-20b/JBBHarmfulBehaviors_gcg_evaluation > logs_bash/ablation-temperature-1.2-gcg-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python examples/3_gcg.py ablation_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --temperature=1.4 --do_sample=True \
  --record_dir=breakoss_logs/openai_gpt-oss-20b/JBBHarmfulBehaviors_gcg_evaluation > logs_bash/ablation-temperature-1.4-gcg-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python examples/3_gcg.py ablation_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --temperature=1.6 --do_sample=True \
  --record_dir=breakoss_logs/openai_gpt-oss-20b/JBBHarmfulBehaviors_gcg_evaluation > logs_bash/ablation-temperature-1.6-gcg-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &

