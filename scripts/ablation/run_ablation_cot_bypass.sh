#CUDA_VISIBLE_DEVICES=0 python examples/1_cot_bypass.py main_inference \
#  --batch_size=8 \
#  --dataset_name=JBBHarmfulBehaviors \
#  --ablation_transform_type=-1 \
#  --model_name=openai/gpt-oss-20b


CUDA_VISIBLE_DEVICES=0 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=8 \
  --dataset_name=JBBHarmfulBehaviors \
  --ablation_transform_type=0 \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-cot_bypass-0-gpt-oss-20b-JBBHarmfulBehaviorstQA.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=8 \
  --dataset_name=JBBHarmfulBehaviors \
  --ablation_transform_type=1 \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-cot_bypass-1-gpt-oss-20b-JBBHarmfulBehaviorstQA.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=8 \
  --dataset_name=JBBHarmfulBehaviors \
  --ablation_transform_type=2 \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-cot_bypass-2-gpt-oss-20b-JBBHarmfulBehaviorstQA.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=8 \
  --dataset_name=JBBHarmfulBehaviors \
  --ablation_transform_type=3 \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-cot_bypass-3-gpt-oss-20b-JBBHarmfulBehaviorstQA.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=8 \
  --dataset_name=JBBHarmfulBehaviors \
  --ablation_transform_type=4 \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-cot_bypass-4-gpt-oss-20b-JBBHarmfulBehaviorstQA.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=8 \
  --dataset_name=JBBHarmfulBehaviors \
  --ablation_transform_type=5 \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-cot_bypass-5-gpt-oss-20b-JBBHarmfulBehaviorstQA.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=8 \
  --dataset_name=JBBHarmfulBehaviors \
  --ablation_transform_type=-1 \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-cot_bypass-minus1-gpt-oss-20b-JBBHarmfulBehaviorstQA.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=8 \
  --dataset_name=JBBHarmfulBehaviors \
  --ablation_transform_type=-2 \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-cot_bypass-minus2-gpt-oss-20b-JBBHarmfulBehaviorstQA.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=8 \
  --dataset_name=JBBHarmfulBehaviors \
  --ablation_transform_type=-3 \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-cot_bypass-minus3-gpt-oss-20b-JBBHarmfulBehaviorstQA.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=8 \
  --dataset_name=JBBHarmfulBehaviors \
  --ablation_transform_type=-4 \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-cot_bypass-minus4-gpt-oss-20b-JBBHarmfulBehaviorstQA.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=8 \
  --dataset_name=JBBHarmfulBehaviors \
  --ablation_transform_type=-5 \
  --model_name=openai/gpt-oss-20b > logs_bash/ablation-cot_bypass-minus5-gpt-oss-20b-JBBHarmfulBehaviorstQA.log 2>&1 &

