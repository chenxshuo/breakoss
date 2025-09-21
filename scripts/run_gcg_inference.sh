
#DEBUG=1 CUDA_VISIBLE_DEVICES=0 python examples/3_gcg.py main_inference \
#  --dataset_name=StrongReject

CUDA_VISIBLE_DEVICES=0 nohup python examples/3_gcg.py main_inference \
  --dataset_name=StrongReject \
  --starting_index=0 \
  --end_index=80  > logs_bash/gcg-inference_gpt-oss-20b_StrongReject_start-0_end-80.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/3_gcg.py main_inference \
  --dataset_name=StrongReject \
  --starting_index=80 \
  --end_index=160  > logs_bash/gcg-inference_gpt-oss-20b_StrongReject_start-80_end-160.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python examples/3_gcg.py main_inference \
  --dataset_name=StrongReject \
  --starting_index=160 \
  --end_index=240  > logs_bash/gcg-inference_gpt-oss-20b_StrongReject_start-160_end-240.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python examples/3_gcg.py main_inference \
  --dataset_name=StrongReject \
  --starting_index=240 \
  --end_index=320  > logs_bash/gcg-inference_gpt-oss-20b_StrongReject_start-240_end-320.log 2>&1 &

# left
CUDA_VISIBLE_DEVICES=0 nohup python examples/3_gcg.py main_inference \
  --dataset_name=StrongReject \
  --starting_index=278 \
  --end_index=295  > logs_bash/gcg-inference_gpt-oss-20b_StrongReject_start-277_end-295.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/3_gcg.py main_inference \
  --dataset_name=StrongReject \
  --starting_index=295 \
  --end_index=320  > logs_bash/gcg-inference_gpt-oss-20b_StrongReject_start-295_end-320.log 2>&1 &


# JBB

CUDA_VISIBLE_DEVICES=0 nohup python examples/3_gcg.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --starting_index=0 \
  --end_index=25  > logs_bash/gcg-inference_gpt-oss-20b_JBBHarmfulBehaviors_start-0_end-25.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/3_gcg.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --starting_index=25 \
  --end_index=50  > logs_bash/gcg-inference_gpt-oss-20b_JBBHarmfulBehaviors_start-25_end-50.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python examples/3_gcg.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --starting_index=50 \
  --end_index=75  > logs_bash/gcg-inference_gpt-oss-20b_JBBHarmfulBehaviors_start-50_end-75.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python examples/3_gcg.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --starting_index=75 \
  --end_index=100  > logs_bash/gcg-inference_gpt-oss-20b_JBBHarmfulBehaviors_start-75_end-100.log 2>&1 &




# harmbench
DEBUG=1 CUDA_VISIBLE_DEVICES=0 python examples/3_gcg.py main_inference \
  --dataset_name=HarmBench

CUDA_VISIBLE_DEVICES=0 nohup python examples/3_gcg.py main_inference \
  --dataset_name=HarmBench \
  --starting_index=0 \
  --end_index=100  > logs_bash/gcg-inference_gpt-oss-20b_HarmBench_start-0_end-100.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/3_gcg.py main_inference \
  --dataset_name=HarmBench \
  --starting_index=100 \
  --end_index=200  > logs_bash/gcg-inference_gpt-oss-20b_HarmBench_start-100_end-200.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python examples/3_gcg.py main_inference \
  --dataset_name=HarmBench \
  --starting_index=200 \
  --end_index=300  > logs_bash/gcg-inference_gpt-oss-20b_HarmBench_start-200_end-300.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python examples/3_gcg.py main_inference \
  --dataset_name=HarmBench \
  --starting_index=300 \
  --end_index=400  > logs_bash/gcg-inference_gpt-oss-20b_HarmBench_start-300_end-400.log 2>&1 &


# AdvBench, HarmfulBehaviors
# hess:0
CUDA_VISIBLE_DEVICES=0 nohup python examples/3_gcg.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --starting_index=0 \
  --end_index=50  > logs_bash/gcg-inference_gpt-oss-20b_HarmfulBehaviors_start-0_end-50.log 2>&1 &
# hess:1
CUDA_VISIBLE_DEVICES=1 nohup python examples/3_gcg.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --starting_index=50 \
  --end_index=100  > logs_bash/gcg-inference_gpt-oss-20b_HarmfulBehaviors_start-50_end-100.log 2>&1 &
# hess:2
CUDA_VISIBLE_DEVICES=2 nohup python examples/3_gcg.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --starting_index=100 \
  --end_index=150  > logs_bash/gcg-inference_gpt-oss-20b_HarmfulBehaviors_start-100_end-150.log 2>&1 &
# hess:3
CUDA_VISIBLE_DEVICES=3 nohup python examples/3_gcg.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --starting_index=150 \
  --end_index=200  > logs_bash/gcg-inference_gpt-oss-20b_HarmfulBehaviors_start-150_end-200.log 2>&1 &
# lrz below

CUDA_VISIBLE_DEVICES=0 nohup python examples/3_gcg.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --starting_index=200 \
  --end_index=250  > logs_bash/gcg-inference_gpt-oss-20b_HarmfulBehaviors_start-200_end-250.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/3_gcg.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --starting_index=250 \
  --end_index=300  > logs_bash/gcg-inference_gpt-oss-20b_HarmfulBehaviors_start-250_end-300.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python examples/3_gcg.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --starting_index=300 \
  --end_index=350  > logs_bash/gcg-inference_gpt-oss-20b_HarmfulBehaviors_start-300_end-350.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python examples/3_gcg.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --starting_index=350 \
  --end_index=400  > logs_bash/gcg-inference_gpt-oss-20b_HarmfulBehaviors_start-350_end-400.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python examples/3_gcg.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --starting_index=400 \
  --end_index=450  > logs_bash/gcg-inference_gpt-oss-20b_HarmfulBehaviors_start-400_end-450.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/3_gcg.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --starting_index=450 \
  --end_index=520  > logs_bash/gcg-inference_gpt-oss-20b_HarmfulBehaviors_start-450_end-520.log 2>&1 &

# CatQA below
CUDA_VISIBLE_DEVICES=0 nohup python examples/3_gcg.py main_inference \
  --dataset_name=CatQA \
  --starting_index=0 \
  --end_index=50  > logs_bash/gcg-inference_gpt-oss-20b_CatQA_start-0_end-50.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/3_gcg.py main_inference \
  --dataset_name=CatQA \
  --starting_index=50 \
  --end_index=100  > logs_bash/gcg-inference_gpt-oss-20b_CatQA_start-50_end-100.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python examples/3_gcg.py main_inference \
  --dataset_name=CatQA \
  --starting_index=100 \
  --end_index=150  > logs_bash/gcg-inference_gpt-oss-20b_CatQA_start-100_end-150.log 2>&1 &

###
CUDA_VISIBLE_DEVICES=3 nohup python examples/3_gcg.py main_inference \
  --dataset_name=CatQA \
  --starting_index=150 \
  --end_index=200  > logs_bash/gcg-inference_gpt-oss-20b_CatQA_start-150_end-200.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python examples/3_gcg.py main_inference \
  --dataset_name=CatQA \
  --starting_index=166 \
  --end_index=183  > logs_bash/gcg-inference_gpt-oss-20b_CatQA_start-166_end-183.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 nohup python examples/3_gcg.py main_inference \
  --dataset_name=CatQA \
  --starting_index=183 \
  --end_index=200  > logs_bash/gcg-inference_gpt-oss-20b_CatQA_start-183_end-200.log 2>&1 &

###
CUDA_VISIBLE_DEVICES=0 nohup python examples/3_gcg.py main_inference \
  --dataset_name=CatQA \
  --starting_index=200 \
  --end_index=250  > logs_bash/gcg-inference_gpt-oss-20b_CatQA_start-200_end-250.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/3_gcg.py main_inference \
  --dataset_name=CatQA \
  --starting_index=250 \
  --end_index=300  > logs_bash/gcg-inference_gpt-oss-20b_CatQA_start-250_end-300.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python examples/3_gcg.py main_inference \
  --dataset_name=CatQA \
  --starting_index=300 \
  --end_index=350  > logs_bash/gcg-inference_gpt-oss-20b_CatQA_start-300_end-350.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python examples/3_gcg.py main_inference \
  --dataset_name=CatQA \
  --starting_index=350 \
  --end_index=400  > logs_bash/gcg-inference_gpt-oss-20b_CatQA_start-350_end-400.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python examples/3_gcg.py main_inference \
  --dataset_name=CatQA \
  --starting_index=400 \
  --end_index=450  > logs_bash/gcg-inference_gpt-oss-20b_CatQA_start-400_end-450.log 2>&1 &
