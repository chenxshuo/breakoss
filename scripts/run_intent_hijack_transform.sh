

#CUDA_VISIBLE_DEVICES=4 python examples/4_intent_hijack.py transform_original_prompts StrongReject

pids=()
CUDA_VISIBLE_DEVICES=4 nohup python examples/4_intent_hijack.py transform_original_prompts StrongReject > logs_bash/transform-intent-hijack-gpt-oss-20b-StrongReject.log 2>&1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=5 python examples/4_intent_hijack.py transform_original_prompts HarmfulBehaviors > logs_bash/transform-intent-hijack-gpt-oss-20b-HarmfulBehaviors.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done
echo "Finished Transform Intent Hijack StrongReject, HarmfulBehaviors"

pids=()
CUDA_VISIBLE_DEVICES=4 python examples/4_intent_hijack.py transform_original_prompts HarmBench > logs_bash/transform-intent-hijack-gpt-oss-20b-HarmBench.log 2>&1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=5 python examples/4_intent_hijack.py transform_original_prompts CatQA > logs_bash/transform-intent-hijack-gpt-oss-20b-CatQA.log 2>&1 &
pids+=($!)
for ((i=${#pids[@]}; i>0; i--)) ; do
    wait -n
done

echo "Finished Transform Intent Hijack HarmBench, CatQA"

CUDA_VISIBLE_DEVICES=0 python examples/4_intent_hijack.py transform_original_prompts JBBHarmfulBehaviors > logs_bash/transform-intent-hijack-gpt-oss-20b-JBBHarmfulBehaviors.log 2>&1 &

echo "Finished Transform Intent Hijack JBBHarmfulBehaviors"