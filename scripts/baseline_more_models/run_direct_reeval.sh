#
#CUDA_VISIBLE_DEVICES=0 nohup python examples/0_direct.py main_evals \
#  --records_dir=breakoss_logs/microsoft_Phi-4-reasoning-plus/strong_reject_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-16_18-29-13/ > logs_bash/eval-phi-4-reasoning-plus-StrongReject.log 2>&1 &
#
#CUDA_VISIBLE_DEVICES=1 nohup python examples/0_direct.py main_evals \
#  --records_dir=breakoss_logs/microsoft_Phi-4-reasoning-plus/harmful_behaviors_gcg_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-16_19-04-12 > logs_bash/eval-phi-4-reasoning-plus-HarmfulBehaviors.log 2>&1 &
#
#CUDA_VISIBLE_DEVICES=2 nohup python examples/0_direct.py main_evals \
#  --records_dir=breakoss_logs/microsoft_Phi-4-reasoning-plus/harmbench_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-16_20-05-59 > logs_bash/eval-phi-4-reasoning-plus-HarmBench.log 2>&1 &
#
#CUDA_VISIBLE_DEVICES=3 nohup python examples/0_direct.py main_evals \
#  --records_dir=breakoss_logs/microsoft_Phi-4-reasoning-plus/catqa_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-16_20-49-48 > logs_bash/eval-phi-4-reasoning-plus-CatQA.log 2>&1 &
#
#
#
#CUDA_VISIBLE_DEVICES=0 nohup python examples/0_direct.py main_evals \
#  --records_dir=breakoss_logs/microsoft_Phi-4-reasoning-plus/jbb_harmful_behaviors_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-16_21-51-14 > logs_bash/eval-phi-4-reasoning-plus-JBB.log 2>&1 &
#
#


CUDA_VISIBLE_DEVICES=1 python examples/0_direct.py main_evals \
  --records_dir=breakoss_logs/Qwen_Qwen3-4B-Thinking-2507/strong_reject_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-16_18-30-13 > logs_bash/eval-Qwen3-4B-Thinking-2507-StrongReject.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python examples/0_direct.py main_evals \
  --records_dir=breakoss_logs/Qwen_Qwen3-4B-Thinking-2507/harmful_behaviors_gcg_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-16_19-04-21 > logs_bash/eval-Qwen3-4B-Thinking-2507-HarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python examples/0_direct.py main_evals \
  --records_dir=breakoss_logs/Qwen_Qwen3-4B-Thinking-2507/harmbench_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-16_20-00-01 > logs_bash/eval-Qwen3-4B-Thinking-2507-HarmBench.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python examples/0_direct.py main_evals \
  --records_dir=breakoss_logs/Qwen_Qwen3-4B-Thinking-2507/catqa_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-16_20-00-01 > logs_bash/eval-Qwen3-4B-Thinking-2507-CatQA.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python examples/0_direct.py main_evals \
  --records_dir=breakoss_logs/Qwen_Qwen3-4B-Thinking-2507/jbb_harmful_behaviors_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-16_20-49-44 > logs_bash/eval-Qwen3-4B-Thinking-2507-JBB.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 python examples/0_direct.py main_evals \
  --records_dir=breakoss_logs/deepseek-ai_DeepSeek-R1-Distill-Llama-8B/strong_reject_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-16_18-31-12 > logs_bash/eval-DeepSeek-R1-Distill-Llama-8B-StrongReject.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python examples/0_direct.py main_evals \
  --records_dir=breakoss_logs/deepseek-ai_DeepSeek-R1-Distill-Llama-8B/harmful_behaviors_gcg_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-16_19-08-33 > logs_bash/eval-DeepSeek-R1-Distill-Llama-8B-HarmfulBehaviors.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python examples/0_direct.py main_evals \
  --records_dir=breakoss_logs/deepseek-ai_DeepSeek-R1-Distill-Llama-8B/harmbench_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-16_20-01-05 > logs_bash/eval-DeepSeek-R1-Distill-Llama-8B-HarmBench.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python examples/0_direct.py main_evals \
  --records_dir=breakoss_logs/deepseek-ai_DeepSeek-R1-Distill-Llama-8B/catqa_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-16_20-40-21 > logs_bash/eval-DeepSeek-R1-Distill-Llama-8B-CatQA.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python examples/0_direct.py main_evals \
  --records_dir=breakoss_logs/deepseek-ai_DeepSeek-R1-Distill-Llama-8B/jbb_harmful_behaviors_[0_-1]/direct/pure_inference_no_eval/seed_42/2025-09-16_21-34-57 > logs_bash/eval-DeepSeek-R1-Distill-Llama-8B-JBB.log 2>&1 &
