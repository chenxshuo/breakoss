python examples/3_gcg.py ablation_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --model_name=openai/gpt-oss-20b \
  --provider=openrouter \
  --batch_size=10 \
  --record_dir=breakoss_logs/openai_gpt-oss-20b/JBBHarmfulBehaviors_gcg_evaluation


nohup python examples/3_gcg.py ablation_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --model_name=gpt-oss-20b \
  --provider=openrouter \
  --batch_size=10 \
  --record_dir=breakoss_logs/openai_gpt-oss-20b/JBBHarmfulBehaviors_gcg_evaluation > logs_bash/gcg-api-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &
