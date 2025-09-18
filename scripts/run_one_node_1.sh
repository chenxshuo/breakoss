
pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=StrongReject \
  --batch_size=16 \
  --model_name=microsoft/Phi-4-reasoning-plus \
  --apply_chat_template=False > logs_bash/fake_overrefusal_and_cot_bypass-phi-4-reasnoning-plus-StrongReject.log 2>&1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=1 nohup python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --batch_size=16 \
  --model_name=microsoft/Phi-4-reasoning-plus \
  --apply_chat_template=False > logs_bash/fake_overrefusal_and_cot_bypass-phi-4-reasnoning-plus-HarmfulBehaviors.log 2>&1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=2 nohup python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=HarmBench \
  --batch_size=16 \
  --model_name=microsoft/Phi-4-reasoning-plus \
  --apply_chat_template=False > logs_bash/fake_overrefusal_and_cot_bypass-phi-4-reasnoning-plus-HarmBench.log 2>&1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=3 nohup python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=CatQA \
  --batch_size=16 \
  --model_name=microsoft/Phi-4-reasoning-plus \
  --apply_chat_template=False > logs_bash/fake_overrefusal_and_cot_bypass-phi-4-reasnoning-plus-CatQA.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "finished 4 for phi-4"
pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=16 \
  --model_name=microsoft/Phi-4-reasoning-plus \
  --apply_chat_template=False > logs_bash/fake_overrefusal_and_cot_bypass-phi-4-reasnoning-plus-JBBHarmfulBehaviors.log 2>&1 &



#
#CUDA_VISIBLE_DEVICES=0 python examples/2_fake_overrefusal.py main_inference \
#  --dataset_name=StrongReject \
#  --batch_size=4 \
#  --model_name=Qwen/Qwen3-4B-Thinking-2507 \
#  --apply_chat_template=False


pids+=($!)
CUDA_VISIBLE_DEVICES=1 nohup python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=StrongReject \
  --batch_size=16 \
  --model_name=Qwen/Qwen3-4B-Thinking-2507 \
  --apply_chat_template=False > logs_bash/fake_overrefusal_and_cot_bypass-Qwen3-4B-Thinking-2507-StrongReject.log 2>&1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=2 nohup python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --batch_size=16 \
  --model_name=Qwen/Qwen3-4B-Thinking-2507 \
  --apply_chat_template=False > logs_bash/fake_overrefusal_and_cot_bypass-Qwen3-4B-Thinking-2507-HarmfulBehaviors.log 2>&1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=3 nohup python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=HarmBench \
  --batch_size=16 \
  --model_name=Qwen/Qwen3-4B-Thinking-2507 \
  --apply_chat_template=False > logs_bash/fake_overrefusal_and_cot_bypass-Qwen3-4B-Thinking-2507-HarmBench.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "finished the second 4 for phi-4"

CUDA_VISIBLE_DEVICES=0 nohup python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=CatQA \
  --batch_size=16 \
  --model_name=Qwen/Qwen3-4B-Thinking-2507 \
  --apply_chat_template=False > logs_bash/fake_overrefusal_and_cot_bypass-Qwen3-4B-Thinking-2507-CatQA.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=16 \
  --model_name=Qwen/Qwen3-4B-Thinking-2507 \
  --apply_chat_template=False > logs_bash/fake_overrefusal_and_cot_bypass-Qwen3-4B-Thinking-2507-JBBHarmfulBehaviors.log 2>&1 &

