CUDA_VISIBLE_DEVICES=0 python examples/3_gcg.py main_eval \
  --dataset_name=StrongReject

CUDA_VISIBLE_DEVICES=0 python examples/3_gcg.py main_eval \
  --dataset_name=JBBHarmfulBehaviors

CUDA_VISIBLE_DEVICES=0 python examples/3_gcg.py main_eval \
  --dataset_name=HarmBench

CUDA_VISIBLE_DEVICES=0 python examples/3_gcg.py main_eval \
  --dataset_name=CatQA

CUDA_VISIBLE_DEVICES=1 python examples/3_gcg.py main_eval \
  --dataset_name=HarmfulBehaviors

CUDA_VISIBLE_DEVICES=0 python examples/3_gcg.py main_eval \
  --dataset_name=HarmfulBehaviors-d
