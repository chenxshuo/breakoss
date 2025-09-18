# H-CoT on Phi
pids=()
CUDA_VISIBLE_DEVICES=2 nohup python examples/baseline_hcot.py main_inference \
  --dataset_name=StrongReject \
  --batch_size=16 \
  --model_name=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
   > logs_bash/baseline_hcot-DeepSeek-R1-Distill-Llama-8B-StrongReject.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
pids=()
CUDA_VISIBLE_DEVICES=2 nohup python examples/baseline_hcot.py main_inference \
  --dataset_name=HarmfulBehaviors \
  --batch_size=16 \
  --model_name=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
   > logs_bash/baseline_hcot-DeepSeek-R1-Distill-Llama-8B-HarmfulBehaviors.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
pids=()
CUDA_VISIBLE_DEVICES=2 nohup python examples/baseline_hcot.py main_inference \
  --dataset_name=HarmBench \
  --batch_size=16 \
  --model_name=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
   > logs_bash/baseline_hcot-DeepSeek-R1-Distill-Llama-8B-HarmBench.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
pids=()
CUDA_VISIBLE_DEVICES=2 nohup python examples/baseline_hcot.py main_inference \
  --dataset_name=CatQA \
  --batch_size=16 \
  --model_name=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
   > logs_bash/baseline_hcot-DeepSeek-R1-Distill-Llama-8B-CatQA.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
pids=()
CUDA_VISIBLE_DEVICES=2 nohup python examples/baseline_hcot.py main_inference \
  --dataset_name=JBBHarmfulBehaviors \
  --batch_size=16 \
  --model_name=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
   > logs_bash/baseline_hcot-DeepSeek-R1-Distill-Llama-8B-JBBHarmfulBehaviors.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
