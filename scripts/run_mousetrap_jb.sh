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
