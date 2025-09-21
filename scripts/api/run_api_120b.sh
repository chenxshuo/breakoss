
#CUDA_VISIBLE_DEVICES=0 python examples/1_cot_bypass.py main_inference \
#  --dataset_name=StrongReject \
#  --model_name=openai/gpt-oss-120b \
#  --provider=openrouter


 #gpt-oss-120b
pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=32 \
  --dataset_name=StrongReject \
  --model_name=openai/gpt-oss-120b \
  --provider=openrouter > logs_bash/cot_bypass-api-gpt-oss-120b-StrongReject.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
CUDA_VISIBLE_DEVICES=0 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=32 \
  --dataset_name=HarmfulBehaviors \
  --model_name=openai/gpt-oss-120b \
  --provider=openrouter > logs_bash/cot_bypass-api-gpt-oss-120b-HarmfulBehaviors.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=32 \
  --dataset_name=HarmBench \
  --model_name=openai/gpt-oss-120b \
  --provider=openrouter > logs_bash/cot_bypass-api-gpt-oss-120b-HarmBench.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=32 \
  --dataset_name=CatQA \
  --model_name=openai/gpt-oss-120b \
  --provider=openrouter > logs_bash/cot_bypass-api-gpt-oss-120b-CatQA.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/1_cot_bypass.py main_inference \
  --batch_size=32 \
  --dataset_name=JBBHarmfulBehaviors \
  --model_name=openai/gpt-oss-120b \
  --provider=openrouter > logs_bash/cot_bypass-api-gpt-oss-120b-CaJBBHarmfulBehaviorstQA.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "finished on cotbypass"
pids=()

# fakeoverrefusal
#DEBUG=1 CUDA_VISIBLE_DEVICES=0 python examples/2_fake_overrefusal.py main_inference \
#  --dataset_name=StrongReject \
#  --batch_size=2 \
#  --model_name=openai/gpt-oss-120b \
#  --provider=openrouter

CUDA_VISIBLE_DEVICES=0 nohup python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=StrongReject \
  --batch_size=16 \
  --model_name=openai/gpt-oss-120b \
  --provider=openrouter > logs_bash/fake_overrefusal_and_cot_bypass-api-gpt-oss-120b-StrongReject.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --batch_size=16 \
  --model_name=openai/gpt-oss-120b \
  --provider=openrouter > logs_bash/fake_overrefusal_and_cot_bypass-api-gpt-oss-120b-HarmfulBehaviors.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=HarmBench \
  --batch_size=16 \
  --model_name=openai/gpt-oss-120b \
  --provider=openrouter > logs_bash/fake_overrefusal_and_cot_bypass-api-gpt-oss-120b-HarmBench.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=CatQA \
  --batch_size=16 \
  --model_name=openai/gpt-oss-120b \
  --provider=openrouter > logs_bash/fake_overrefusal_and_cot_bypass-api-gpt-oss-120b-CatQA.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/2_fake_overrefusal.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=16 \
  --model_name=openai/gpt-oss-120b \
  --provider=openrouter > logs_bash/fake_overrefusal_and_cot_bypass-api-gpt-oss-120b-JBBHarmfulBehaviors.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "finished on fake overrefusal"
pids=()
# reason hijack

#DEBUG=1 CUDA_VISIBLE_DEVICES=0 python examples/5_plan_injection.py main_inference \
#  --dataset_name=StrongReject \
#  --batch_size=2 \
#  --model_name=openai/gpt-oss-120b \
#  --provider=openrouter

CUDA_VISIBLE_DEVICES=0 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=StrongReject \
  --batch_size=16 \
  --model_name=openai/gpt-oss-120b \
  --provider=openrouter > logs_bash/plan_injection-api-gpt-oss-120b-StrongReject.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --batch_size=16 \
  --model_name=openai/gpt-oss-120b \
  --provider=openrouter > logs_bash/plan_injection-api-gpt-oss-120b-HarmfulBehaviors.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=HarmBench \
  --batch_size=16 \
  --model_name=openai/gpt-oss-120b \
  --provider=openrouter > logs_bash/plan_injection-api-gpt-oss-120b-HarmBench.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=CatQA \
  --batch_size=16 \
  --model_name=openai/gpt-oss-120b \
  --provider=openrouter > logs_bash/plan_injection-api-gpt-oss-120b-CatQA.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
pids=()
CUDA_VISIBLE_DEVICES=0 nohup python examples/5_plan_injection.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=16 \
  --model_name=openai/gpt-oss-120b \
  --provider=openrouter > logs_bash/plan_injection-api-gpt-oss-120b-JBBHarmfulBehaviors.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
pids=()
