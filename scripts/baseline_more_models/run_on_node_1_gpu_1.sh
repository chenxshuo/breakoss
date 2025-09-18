
#CUDA_VISIBLE_DEVICES=0 python examples/0_direct.py main_inference \
#  --dataset_name=StrongReject \
#  --batch_size=16 \
#  --model_name=microsoft/Phi-4-reasoning-plus

pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/0_direct.py main_inference \
  --dataset_name=StrongReject \
  --batch_size=16 \
  --model_name=microsoft/Phi-4-reasoning-plus \
   > logs_bash/direct-phi-4-reasnoning-plus-StrongReject.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/0_direct.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --batch_size=16 \
  --model_name=microsoft/Phi-4-reasoning-plus \
   > logs_bash/direct-phi-4-reasnoning-plus-HarmfulBehaviors.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/0_direct.py main_inference \
  --dataset_name=HarmBench \
  --batch_size=16 \
  --model_name=microsoft/Phi-4-reasoning-plus \
   > logs_bash/direct-phi-4-reasnoning-plus-HarmBench.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/0_direct.py main_inference \
  --dataset_name=CatQA \
  --batch_size=16 \
  --model_name=microsoft/Phi-4-reasoning-plus \
   > logs_bash/direct-phi-4-reasnoning-plus-CatQA.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/0_direct.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=16 \
  --model_name=microsoft/Phi-4-reasoning-plus \
   > logs_bash/direct-phi-4-reasnoning-plus-JBBHarmfulBehaviors.log 2>&1 &

pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
