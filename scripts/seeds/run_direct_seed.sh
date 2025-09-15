#!/bin/bash

SEED=16
pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/0_direct.py main_inference \
  --dataset_name=StrongReject \
  --seed=$SEED \
  --model_name=openai/gpt-oss-20b > logs_bash/seed/seed-$SEED-direct-gpt-oss-20b-StrongReject.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "finished direct strongreject"

pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/0_direct.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --seed=$SEED \
  --model_name=openai/gpt-oss-20b > logs_bash/seed/seed-$SEED-direct-gpt-oss-20b-HarmfulBehaviors.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "finished direct harmfulbehaviors"

pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/0_direct.py main_inference \
  --dataset_name=HarmBench \
  --seed=$SEED \
  --model_name=openai/gpt-oss-20b > logs_bash/seed/seed-$SEED-direct-gpt-oss-20b-HarmBench.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "finished direct harmbench"

pids=()
 CUDA_VISIBLE_DEVICES=0 nohup python examples/0_direct.py main_inference \
  --dataset_name=CatQA \
  --seed=$SEED \
  --model_name=openai/gpt-oss-20b > logs_bash/seed/seed-$SEED-direct-gpt-oss-20b-CatQA.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "finished direct catqa"

pids=()
 CUDA_VISIBLE_DEVICES=0 nohup python examples/0_direct.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --seed=$SEED \
  --model_name=openai/gpt-oss-20b > logs_bash/seed/seed-$SEED-direct-gpt-oss-20b-CaJBBHarmfulBehaviorstQA.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "finished direct jbb"







## gpt-oss-120b
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
#pids=()
#nohup python examples/0_direct.py main_inference \
#  --dataset_name=StrongReject \
#  --model_name=openai/gpt-oss-120b > logs_bash/direct-gpt-oss-120b-StrongReject.log 2>&1 &
#pids+=($!)
#for ((i=${#pids[@]}; i>0; i--)) ; do
#    wait -n
#done
#echo "Finished StrongReject with gpt-oss-120b"
#
#pids=()
#nohup python examples/0_direct.py main_inference \
#  --dataset_name=HarmfulBehaviors \
#  --model_name=openai/gpt-oss-120b > logs_bash/direct-gpt-oss-120b-HarmfulBehaviors.log 2>&1 &
#pids+=($!)
#for ((i=${#pids[@]}; i>0; i--)) ; do
#    wait -n
#done
#echo "Finished HarmfulBehaviors with gpt-oss-120b"
#
#pids=()
#nohup python examples/0_direct.py main_inference \
#  --dataset_name=HarmBench \
#  --model_name=openai/gpt-oss-120b > logs_bash/direct-gpt-oss-120b-HarmBench.log 2>&1 &
#pids+=($!)
#for ((i=${#pids[@]}; i>0; i--)) ; do
#    wait -n
#done
#echo "Finished HarmBench with gpt-oss-120b"
#
#pids=()
#nohup python examples/0_direct.py main_inference \
#  --dataset_name=CatQA \
#  --model_name=openai/gpt-oss-120b > logs_bash/direct-gpt-oss-120b-CatQA.log 2>&1 &
#pids+=($!)
#for ((i=${#pids[@]}; i>0; i--)) ; do
#    wait -n
#done
#echo "Finished CatQA with gpt-oss-120b"
#
#pids=()
#nohup python examples/0_direct.py main_inference \
#  --dataset_name=JBBHarmfulBehaviors \
#  --model_name=openai/gpt-oss-120b > logs_bash/direct-gpt-oss-120b-JBBHarmfulBehaviorstQA.log 2>&1 &
#pids+=($!)
#for ((i=${#pids[@]}; i>0; i--)) ; do
#    wait -n
#done
#echo "Finished JBBHarmfulBehaviors with gpt-oss-120b"
#

#exit 0

## phi-4-reasoning
#pids=()
#
#nohup python examples/0_direct.py main_inference \
#  --dataset_name=StrongReject \
#  --model_name=microsoft/phi-4-reasoning > logs_bash/direct-phi-4-reasoning-StrongReject.log 2>&1 &
#pids+=($!)
#
#nohup python examples/0_direct.py main_inference \
#  --dataset_name=HarmfulBehaviors \
#  --model_name=microsoft/phi-4-reasoning > logs_bash/direct-phi-4-reasoning-HarmfulBehaviors.log 2>&1 &
#pids+=($!)
#
#nohup python examples/0_direct.py main_inference \
#  --dataset_name=HarmBench \
#  --model_name=microsoft/phi-4-reasoning > logs_bash/direct-phi-4-reasoning-HarmBench.log 2>&1 &
#pids+=($!)
#
#for ((i=${#pids[@]}; i>0; i--)) ; do
#    wait -n
#done
#
#echo "finished StrongReject, HarmfulBehaviors, HarmBench with phi-4-reasoning"
#
#
#pids=()
#nohup python examples/0_direct.py main_inference \
#  --dataset_name=CatQA \
#  --model_name=microsoft/phi-4-reasoning > logs_bash/direct-phi-4-reasoning-CatQA.log 2>&1 &
#pids+=($!)
#
#nohup python examples/0_direct.py main_inference \
#  --dataset_name=JBBHarmfulBehaviors \
#  --model_name=microsoft/phi-4-reasoning > logs_bash/direct-phi-4-reasoning-JBBHarmfulBehaviorstQA.log 2>&1 &
#pids+=($!)
#
#for ((i=${#pids[@]}; i>0; i--)) ; do
#    wait -n
#done
#
#echo "finished CatQA, JBBHarmfulBehaviors with phi-4-reasoning"
#
## qwen3-4b-thinking-2507
#pids=()
#nohup python examples/0_direct.py main_inference \
#  --dataset_name=StrongReject \
#  --model_name=qwen/qwen3-4b-thinking-2507 > logs_bash/direct-qwen3-4b-thinking-2507-StrongReject.log 2>&1 &
#pids+=($!)
#
#nohup python examples/0_direct.py main_inference \
#  --dataset_name=HarmfulBehaviors \
#  --model_name=qwen/qwen3-4b-thinking-2507 > logs_bash/direct-qwen3-4b-thinking-2507-HarmfulBehaviors.log 2>&1 &
#pids+=($!)
#
#nohup python examples/0_direct.py main_inference \
#  --dataset_name=HarmBench \
#  --model_name=qwen/qwen3-4b-thinking-2507 > logs_bash/direct-qwen3-4b-thinking-2507-HarmBench.log 2>&1 &
#pids+=($!)
#for ((i=${#pids[@]}; i>0; i--)) ; do
#    wait -n
#done
#
#echo "finished StrongReject, HarmfulBehaviors, HarmBench with qwen3-4b-thinking-2507"
#
#pids=()
#nohup python examples/0_direct.py main_inference \
#  --dataset_name=CatQA \
#  --model_name=qwen/qwen3-4b-thinking-2507 > logs_bash/direct-qwen3-4b-thinking-2507-CatQA.log 2>&1 &
#pids+=($!)
#
#nohup python examples/0_direct.py main_inference \
#  --dataset_name=JBBHarmfulBehaviors \
#  --model_name=qwen/qwen3-4b-thinking-2507 > logs_bash/direct-qwen3-4b-thinking-2507-JBBHarmfulBehaviorstQA.log 2>&1 &
#pids+=($!)
#for ((i=${#pids[@]}; i>0; i--)) ; do
#    wait -n
#done
#echo "finished CatQA, JBBHarmfulBehaviors with qwen3-4b-thinking-2507"
#
## RealSafe-R1-7B
#
#pids=()
#nohup python examples/0_direct.py main_inference \
#  --dataset_name=StrongReject \
#  --model_name=RealSafe/RealSafe-R1-7B > logs_bash/direct-RealSafe-R1-7B-StrongReject.log 2>&1 &
#pids+=($!)
#
#nohup python examples/0_direct.py main_inference \
#  --dataset_name=HarmfulBehaviors \
#  --model_name=RealSafe/RealSafe-R1-7B > logs_bash/direct-RealSafe-R1-7B-HarmfulBehaviors.log 2>&1 &
#pids+=($!)
#
#nohup python examples/0_direct.py main_inference \
#  --dataset_name=HarmBench \
#  --model_name=RealSafe/RealSafe-R1-7B > logs_bash/direct-RealSafe-R1-7B-HarmBench.log 2>&1 &
#pids+=($!)
#for ((i=${#pids[@]}; i>0; i--)) ; do
#    wait -n
#done
#echo "finished StrongReject, HarmfulBehaviors, HarmBench with RealSafe-R1-7B"
#
#
#pids=()
#nohup python examples/0_direct.py main_inference \
#  --dataset_name=CatQA \
#  --model_name=RealSafe/RealSafe-R1-7B > logs_bash/direct-RealSafe-R1-7B-CatQA.log 2>&1 &
#pids+=($!)
#
#nohup python examples/0_direct.py main_inference \
#  --dataset_name=JBBHarmfulBehaviors \
#  --model_name=RealSafe/RealSafe-R1-7B > logs_bash/direct-RealSafe-R1-7B-JBBHarmfulBehaviorstQA.log 2>&1 &
#pids+=($!)
#for ((i=${#pids[@]}; i>0; i--)) ; do
#    wait -n
#done
#echo "finished CatQA, JBBHarmfulBehaviors with RealSafe-R1-7B"