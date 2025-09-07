#!/bin/bash

# Run cot_bypass inference
# SUPPORTED_MODELS = [
  #    "openai/gpt-oss-20b",
  #    "openai/gpt-oss-120b",
  #    "microsoft/phi-4-reasoning",
  #    "qwen/qwen3-4b-thinking-2507",
  #    "RealSafe/RealSafe-R1-7B"
  #]


#DEBUG=1 python examples/1_cot_bypass.py main_inference \
#  --dataset_name=StrongReject \
#  --model_name=openai/gpt-oss-20b

# gpt-oss-20b
#pids=()
#CUDA_VISIBLE_DEVICES=0 nohup python examples/1_cot_bypass.py main_inference \
#  --batch_size=32 \
#  --dataset_name=StrongReject \
#  --model_name=openai/gpt-oss-20b > logs_bash/cot_bypass-gpt-oss-20b-StrongReject.log 2>&1 &
#pids+=($!)
#
#nohup python examples/1_cot_bypass.py main_inference \
#  --dataset_name=HarmfulBehaviors \
#  --model_name=openai/gpt-oss-20b > logs_bash/cot_bypass-gpt-oss-20b-HarmfulBehaviors.log 2>&1 &
#pids+=($!)
#for ((i=${#pids[@]}; i>0; i--)) ; do
#    wait -n
#done
#echo "Finished StrongReject, HarmfulBehaviors with gpt-oss-20b"
#
#pids=()
#CUDA_VISIBLE_DEVICES=1 nohup python examples/1_cot_bypass.py main_inference \
#  --batch_size=32 \
#  --dataset_name=HarmBench \
#  --model_name=openai/gpt-oss-20b > logs_bash/cot_bypass-gpt-oss-20b-HarmBench.log 2>&1 &
#pids+=($!)
#
#CUDA_VISIBLE_DEVICES=2 nohup python examples/1_cot_bypass.py main_inference \
#  --batch_size=32 \
#  --dataset_name=CatQA \
#  --model_name=openai/gpt-oss-20b > logs_bash/cot_bypass-gpt-oss-20b-CatQA.log 2>&1 &
#
#for ((i=${#pids[@]}; i>0; i--)) ; do
#    wait -n
#done
#echo "Finished HarmBench CatQA with gpt-oss-20b"
#
#pids=()
#CUDA_VISIBLE_DEVICES=3 nohup python examples/1_cot_bypass.py main_inference \
#  --batch_size=32 \
#  --dataset_name=JBBHarmfulBehaviors \
#  --model_name=openai/gpt-oss-20b > logs_bash/cot_bypass-gpt-oss-20b-CaJBBHarmfulBehaviorstQA.log 2>&1 &
#pids+=($!)


## gpt-oss-120b

CUDA_VISIBLE_DEVICES=2,3,4,5 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=8 \
  --dataset_name=StrongReject \
  --model_name=openai/gpt-oss-120b > logs_bash/cot_bypass-gpt-oss-120b-StrongReject.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "Finished StrongReject with gpt-oss-120b"

pids=()
CUDA_VISIBLE_DEVICES=2,3,4,5 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=8 \
  --dataset_name=HarmfulBehaviors \
  --model_name=openai/gpt-oss-120b > logs_bash/cot_bypass-gpt-oss-120b-HarmfulBehaviors.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "Finished HarmfulBehaviors with gpt-oss-120b"

pids=()
CUDA_VISIBLE_DEVICES=2,3,4,5 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=8 \
  --dataset_name=HarmBench \
  --model_name=openai/gpt-oss-120b > logs_bash/cot_bypass-gpt-oss-120b-HarmBench.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "Finished HarmBench with gpt-oss-120b"

pids=()
CUDA_VISIBLE_DEVICES=2,3,4,5  nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=8 \
  --dataset_name=CatQA \
  --model_name=openai/gpt-oss-120b > logs_bash/cot_bypass-gpt-oss-120b-CatQA.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "Finished CatQA with gpt-oss-120b"

pids=()
CUDA_VISIBLE_DEVICES=2,3,4,5 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=8 \
  --dataset_name=JBBHarmfulBehaviors \
  --model_name=openai/gpt-oss-120b > logs_bash/cot_bypass-gpt-oss-120b-JBBHarmfulBehaviorstQA.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "Finished JBBHarmfulBehaviors with gpt-oss-120b"
exit 0

# phi-4-reasoning
pids=()

nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=StrongReject \
  --model_name=microsoft/phi-4-reasoning > logs_bash/cot_bypass-phi-4-reasoning-StrongReject.log 2>&1 &
pids+=($!)

nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --model_name=microsoft/phi-4-reasoning > logs_bash/cot_bypass-phi-4-reasoning-HarmfulBehaviors.log 2>&1 &
pids+=($!)

nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=HarmBench \
  --model_name=microsoft/phi-4-reasoning > logs_bash/cot_bypass-phi-4-reasoning-HarmBench.log 2>&1 &
pids+=($!)

for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done

echo "finished StrongReject, HarmfulBehaviors, HarmBench with phi-4-reasoning"


pids=()
nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=CatQA \
  --model_name=microsoft/phi-4-reasoning > logs_bash/cot_bypass-phi-4-reasoning-CatQA.log 2>&1 &
pids+=($!)

nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --model_name=microsoft/phi-4-reasoning > logs_bash/cot_bypass-phi-4-reasoning-JBBHarmfulBehaviorstQA.log 2>&1 &
pids+=($!)

for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done

echo "finished CatQA, JBBHarmfulBehaviors with phi-4-reasoning"

# qwen3-4b-thinking-2507
pids=()
nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=StrongReject \
  --model_name=qwen/qwen3-4b-thinking-2507 > logs_bash/cot_bypass-qwen3-4b-thinking-2507-StrongReject.log 2>&1 &
pids+=($!)

nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --model_name=qwen/qwen3-4b-thinking-2507 > logs_bash/cot_bypass-qwen3-4b-thinking-2507-HarmfulBehaviors.log 2>&1 &
pids+=($!)

nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=HarmBench \
  --model_name=qwen/qwen3-4b-thinking-2507 > logs_bash/cot_bypass-qwen3-4b-thinking-2507-HarmBench.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done

echo "finished StrongReject, HarmfulBehaviors, HarmBench with qwen3-4b-thinking-2507"

pids=()
nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=CatQA \
  --model_name=qwen/qwen3-4b-thinking-2507 > logs_bash/cot_bypass-qwen3-4b-thinking-2507-CatQA.log 2>&1 &
pids+=($!)

nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --model_name=qwen/qwen3-4b-thinking-2507 > logs_bash/cot_bypass-qwen3-4b-thinking-2507-JBBHarmfulBehaviorstQA.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "finished CatQA, JBBHarmfulBehaviors with qwen3-4b-thinking-2507"

# RealSafe-R1-7B

pids=()
nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=StrongReject \
  --model_name=RealSafe/RealSafe-R1-7B > logs_bash/cot_bypass-RealSafe-R1-7B-StrongReject.log 2>&1 &
pids+=($!)

nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --model_name=RealSafe/RealSafe-R1-7B > logs_bash/cot_bypass-RealSafe-R1-7B-HarmfulBehaviors.log 2>&1 &
pids+=($!)

nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=HarmBench \
  --model_name=RealSafe/RealSafe-R1-7B > logs_bash/cot_bypass-RealSafe-R1-7B-HarmBench.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "finished StrongReject, HarmfulBehaviors, HarmBench with RealSafe-R1-7B"


pids=()
nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=CatQA \
  --model_name=RealSafe/RealSafe-R1-7B > logs_bash/cot_bypass-RealSafe-R1-7B-CatQA.log 2>&1 &
pids+=($!)

nohup python examples/1_cot_bypass.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --model_name=RealSafe/RealSafe-R1-7B > logs_bash/cot_bypass-RealSafe-R1-7B-JBBHarmfulBehaviorstQA.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "finished CatQA, JBBHarmfulBehaviors with RealSafe-R1-7B"