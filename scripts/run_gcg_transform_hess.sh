


CUDA_VISIBLE_DEVICES=0 nohup python examples/3_gcg.py transform \
  --dataset_name=HarmfulBehaviors \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=0 --end_ind=50 > logs_bash/gcg_optimization_transform_HarmfulBehaviors_gpt-oss-20b_start_0_end_50.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/3_gcg.py transform \
  --dataset_name=HarmfulBehaviors \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=50 --end_ind=100 > logs_bash/gcg_optimization_transform_HarmfulBehaviors_gpt-oss-20b_start_50_end_100.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python examples/3_gcg.py transform \
  --dataset_name=HarmfulBehaviors \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=100 --end_ind=150 > logs_bash/gcg_optimization_transform_HarmfulBehaviors_gpt-oss-20b_start_50_end_150.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 nohup python examples/3_gcg.py transform \
  --dataset_name=HarmfulBehaviors \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=150 --end_ind=200 > logs_bash/gcg_optimization_transform_HarmfulBehaviors_gpt-oss-20b_start_150_end_200.log 2>&1 &



#
#pids=()
#CUDA_VISIBLE_DEVICES=0 nohup python examples/3_gcg.py transform \
#  --dataset_name=HarmBench \
#  --target_model_name=openai/gpt-oss-20b \
#  --start_ind=0 --end_ind=50 > logs_bash/gcg_optimization_transform_HarmBench_gpt-oss-20b_start_0_end_50.log 2>&1 &
#pids+=($!)
#for pid in "${pids[@]}"; do
#    wait $pid
#done
#echo "finished on 0-50"
#
#
#
#CUDA_VISIBLE_DEVICES=0 nohup python examples/3_gcg.py transform \
#  --dataset_name=HarmBench \
#  --target_model_name=openai/gpt-oss-20b \
#  --start_ind=50 --end_ind=100 > logs_bash/gcg_optimization_transform_HarmBench_gpt-oss-20b_start_50_end_100.log 2>&1 &
#
#pids=()
#CUDA_VISIBLE_DEVICES=1 nohup python examples/3_gcg.py transform \
#  --dataset_name=HarmBench \
#  --target_model_name=openai/gpt-oss-20b \
#  --start_ind=101 --end_ind=150 > logs_bash/gcg_optimization_transform_HarmBench_gpt-oss-20b_start_101_end_150.log 2>&1 &
#pids+=($!)
#for pid in "${pids[@]}"; do
#    wait $pid
#done
#echo "finished on 101-150"
#
#
#CUDA_VISIBLE_DEVICES=1 nohup python examples/3_gcg.py transform \
#  --dataset_name=HarmBench \
#  --target_model_name=openai/gpt-oss-20b \
#  --start_ind=151 --end_ind=200 > logs_bash/gcg_optimization_transform_HarmBench_gpt-oss-20b_start_151_end_200.log 2>&1 &
#
#
#pids=()
#CUDA_VISIBLE_DEVICES=2 nohup python examples/3_gcg.py transform \
#  --dataset_name=HarmBench \
#  --target_model_name=openai/gpt-oss-20b \
#  --start_ind=201 --end_ind=250 > logs_bash/gcg_optimization_transform_HarmBench_gpt-oss-20b_start_201_end_250.log 2>&1 &
#pids+=($!)
#for pid in "${pids[@]}"; do
#    wait $pid
#done
#echo "finished on 101-150"
#
#CUDA_VISIBLE_DEVICES=2 nohup python examples/3_gcg.py transform \
#  --dataset_name=HarmBench \
#  --target_model_name=openai/gpt-oss-20b \
#  --start_ind=251 --end_ind=300 > logs_bash/gcg_optimization_transform_HarmBench_gpt-oss-20b_start_251_end_300.log 2>&1 &
#
#pids=()
#CUDA_VISIBLE_DEVICES=3 nohup python examples/3_gcg.py transform \
#  --dataset_name=HarmBench \
#  --target_model_name=openai/gpt-oss-20b \
#  --start_ind=301 --end_ind=350 > logs_bash/gcg_optimization_transform_HarmBench_gpt-oss-20b_start_301_end_350.log 2>&1 &
#pids+=($!)
#for pid in "${pids[@]}"; do
#    wait $pid
#done
#echo "finished on 101-150"
#
#CUDA_VISIBLE_DEVICES=3 nohup python examples/3_gcg.py transform \
#  --dataset_name=HarmBench \
#  --target_model_name=openai/gpt-oss-20b \
#  --start_ind=351 --end_ind=400 > logs_bash/gcg_optimization_transform_HarmBench_gpt-oss-20b_start_351_end_400.log 2>&1 &
