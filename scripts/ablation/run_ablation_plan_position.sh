# reasoning hijack method
# the position of the plan.
# not in the reasoning section, put it in the user prompt section.
# the model is gpt-oss-20b
# the dataset is JBBHarmfulBehaviors
# aka, plan injection's data but use cot bypass method.

#CUDA_VISIBLE_DEVICES=0 python examples/1_cot_bypass.py main_inference \
#  --batch_size=8 \
#  --dataset_name=JBBHarmfulBehaviors \
#  --repeat_user_prompt=False \
#  --dataset_path=transformed_datasets/jbb-harmful-behaviors_PlanInjection_pure_plan.csv \
#  --model_name=openai/gpt-oss-20b

CUDA_VISIBLE_DEVICES=0 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=8 \
  --dataset_name=JBBHarmfulBehaviors \
  --repeat_user_prompt=False \
  --dataset_path=transformed_datasets/jbb-harmful-behaviors_PlanInjection_pure_plan.csv \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-reasoning_position-cot_bypass-gpt-oss-20b-JBBHarmfulBehaviorstQA.log 2>&1 &



CUDA_VISIBLE_DEVICES=1 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=8 \
  --dataset_name=StrongReject \
  --repeat_user_prompt=False \
  --dataset_path=transformed_datasets/strongreject_dataset_PlanInjection_pure_plan.csv \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-reasoning_position-cot_bypass-gpt-oss-20b-StrongReject.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=8 \
  --dataset_name=HarmfulBehaviors \
  --repeat_user_prompt=False \
  --dataset_path=transformed_datasets/harmful_behaviors_PlanInjection_pure_plan.csv \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-reasoning_position-cot_bypass-gpt-oss-20b-HarmfulBehaviors.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=8 \
  --dataset_name=HarmBench \
  --repeat_user_prompt=False \
  --dataset_path=transformed_datasets/harmbench_behaviors_text_all_PlanInjection_pure_plan.csv \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-reasoning_position-cot_bypass-gpt-oss-20b-HarmBench.log 2>&1 &
