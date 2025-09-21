CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/3_gcg.py transform \
  --dataset_name=JBBHarmfulBehaviors \
  --target_model_name=openai/gpt-oss-120b \
  --num_steps=150 --gcg_batch_size=12 \
  --start_ind=0 --end_ind=20 > logs_bash/gcg_optimization_transform_JBBHarmfulBehaviors_gpt-oss-120b_start_0_end_20.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/3_gcg.py transform \
  --dataset_name=JBBHarmfulBehaviors \
  --target_model_name=openai/gpt-oss-120b \
  --num_steps=100 --gcg_batch_size=12 --search_width=64 \
  --start_ind=5 --end_ind=20 > logs_bash/gcg_optimization_transform_JBBHarmfulBehaviors_gpt-oss-120b_start_5_end_20.log 2>&1 &





CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/3_gcg.py transform \
  --dataset_name=JBBHarmfulBehaviors \
  --target_model_name=openai/gpt-oss-120b \
  --num_steps=150 \
  --start_ind=20 --end_ind=40 > logs_bash/gcg_optimization_transform_JBBHarmfulBehaviors_gpt-oss-120b_start_20_end_40.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/3_gcg.py transform \
  --dataset_name=JBBHarmfulBehaviors \
  --target_model_name=openai/gpt-oss-120b \
  --num_steps=100 --gcg_batch_size=12 --search_width=64 \
  --start_ind=28 --end_ind=40 > logs_bash/gcg_optimization_transform_JBBHarmfulBehaviors_gpt-oss-120b_start_28_end_40.log 2>&1 &


CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/3_gcg.py transform \
  --dataset_name=JBBHarmfulBehaviors \
  --target_model_name=openai/gpt-oss-120b \
  --num_steps=150 \
  --start_ind=40 --end_ind=60 > logs_bash/gcg_optimization_transform_JBBHarmfulBehaviors_gpt-oss-120b_start_40_end_60.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/3_gcg.py transform \
  --dataset_name=JBBHarmfulBehaviors \
  --target_model_name=openai/gpt-oss-120b \
  --num_steps=100 --gcg_batch_size=12 --search_width=64 \
  --start_ind=42 --end_ind=60 > logs_bash/gcg_optimization_transform_JBBHarmfulBehaviors_gpt-oss-120b_start_42_end_60.log 2>&1 &



CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/3_gcg.py transform \
  --dataset_name=JBBHarmfulBehaviors \
  --target_model_name=openai/gpt-oss-120b \
  --num_steps=100 --gcg_batch_size=12 --search_width=64 \
  --start_ind=60 --end_ind=80 > logs_bash/gcg_optimization_transform_JBBHarmfulBehaviors_gpt-oss-120b_start_60_end_80.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/3_gcg.py transform \
  --dataset_name=JBBHarmfulBehaviors \
  --target_model_name=openai/gpt-oss-120b \
  --num_steps=150 \
  --start_ind=80 --end_ind=100 > logs_bash/gcg_optimization_transform_JBBHarmfulBehaviors_gpt-oss-120b_start_80_end_100.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python examples/3_gcg.py transform \
  --dataset_name=JBBHarmfulBehaviors \
  --target_model_name=openai/gpt-oss-120b \
  --num_steps=100 --gcg_batch_size=12 --search_width=64 \
  --start_ind=82 --end_ind=100 > logs_bash/gcg_optimization_transform_JBBHarmfulBehaviors_gpt-oss-120b_start_82_end_100.log 2>&1 &

