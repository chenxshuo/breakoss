CUDA_VISIBLE_DEVICES=0 nohup python examples/baseline_mousetrap.py main_inference \
  --dataset_name=StrongReject \
  --batch_size=10 \
  --model_name=openai/gpt-oss-20b > logs_bash/baseline_mousetrap-gpt-oss-20b-strongReject.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/baseline_mousetrap.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --batch_size=10 \
  --model_name=openai/gpt-oss-20b > logs_bash/baseline_mousetrap-gpt-oss-20b-HarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python examples/baseline_mousetrap.py main_inference \
  --dataset_name=HarmBench \
  --batch_size=10 \
  --model_name=openai/gpt-oss-20b > logs_bash/baseline_mousetrap-gpt-oss-20b-HarmBench.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python examples/baseline_mousetrap.py main_inference \
  --dataset_name=CatQA \
  --batch_size=10 \
  --model_name=openai/gpt-oss-20b > logs_bash/baseline_mousetrap-gpt-oss-20b-CatQA.log 2>&1 &


CUDA_VISIBLE_DEVICES=4 nohup python examples/baseline_mousetrap.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=10 \
  --model_name=openai/gpt-oss-20b > logs_bash/baseline_mousetrap-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &


#CUDA_VISIBLE_DEVICES=0 python examples/baseline_mousetrap.py main_eval \
#  --record_dir=breakoss_logs/openai_gpt-oss-20b/strong_reject_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-11_23-16-11
#


#CUDA_VISIBLE_DEVICES=4 python examples/baseline_mousetrap.py re_eval_answer_match \
#  --record_dir=breakoss_logs/openai_gpt-oss-20b/strong_reject_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-11_23-16-11/re_evaluation_records_before_evaluation

CUDA_VISIBLE_DEVICES=4 nohup python examples/baseline_mousetrap.py re_eval_answer_match \
  --record_dir=breakoss_logs/openai_gpt-oss-20b/strong_reject_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-11_23-16-11/re_evaluation_records_before_evaluation > logs_bash/baseline_mousetrap_re_eval_related_strongreject.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup python examples/baseline_mousetrap.py re_eval_answer_match \
  --record_dir=breakoss_logs/openai_gpt-oss-20b/harmful_behaviors_gcg_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-11_23-16-12/re_evaluation_records_before_evaluation > logs_bash/baseline_mousetrap_re_eval_related_harmful_behaviors 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup python examples/baseline_mousetrap.py re_eval_answer_match \
  --record_dir=breakoss_logs/openai_gpt-oss-20b/harmbench_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-11_23-16-11/re_evaluation_records_before_evaluation > logs_bash/baseline_mousetrap_re_eval_related_harmbench 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup python examples/baseline_mousetrap.py re_eval_answer_match \
  --record_dir=breakoss_logs/openai_gpt-oss-20b/catqa_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-11_23-16-11/re_evaluation_records_before_evaluation > logs_bash/baseline_mousetrap_re_eval_related_catqa 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python examples/baseline_mousetrap.py re_eval_answer_match \
  --record_dir=breakoss_logs/openai_gpt-oss-20b/jbb_harmful_behaviors_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-11_23-16-11/re_evaluation_records_before_evaluation > logs_bash/baseline_mousetrap_re_eval_related_jbb 2>&1 &






