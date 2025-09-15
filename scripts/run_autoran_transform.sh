CUDA_VISIBLE_DEVICES=4 nohup python examples/baseline_autoran.py transform_original_prompts StrongReject > logs_bash/transform-autoran-gpt-oss-20b-StrongReject.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python examples/baseline_autoran.py transform_original_prompts HarmfulBehaviors > logs_bash/transform-autoran-gpt-oss-20b-HarmfulBehaviors.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python examples/baseline_autoran.py transform_original_prompts HarmBench > logs_bash/transform-autoran-gpt-oss-20b-HarmBench.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python examples/baseline_autoran.py transform_original_prompts JBBHarmfulBehaviors > logs_bash/transform-autoran-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python examples/baseline_autoran.py transform_original_prompts CatQA > logs_bash/transform-autoran-gpt-oss-20b-CatQA.log 2>&1 &