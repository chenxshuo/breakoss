
# harmfulbehaviors
CUDA_VISIBLE_DEVICES=0 nohup python examples/3_gcg.py transform \
  --dataset_name=HarmfulBehaviors \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=200 --end_ind=250 > logs_bash/gcg_optimization_transform_HarmfulBehaviors_gpt-oss-20b_start_200_end_250.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 nohup python examples/3_gcg.py transform \
  --dataset_name=HarmfulBehaviors \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=250 --end_ind=300 > logs_bash/gcg_optimization_transform_HarmfulBehaviors_gpt-oss-20b_start_250_end_300.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 nohup python examples/3_gcg.py transform \
  --dataset_name=HarmfulBehaviors \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=300 --end_ind=350 > logs_bash/gcg_optimization_transform_HarmfulBehaviors_gpt-oss-20b_start_300_end_350.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python examples/3_gcg.py transform \
  --dataset_name=HarmfulBehaviors \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=350 --end_ind=400 > logs_bash/gcg_optimization_transform_HarmfulBehaviors_gpt-oss-20b_start_350_end_400.log 2>&1 &



CUDA_VISIBLE_DEVICES=0 nohup python examples/3_gcg.py transform \
  --dataset_name=HarmfulBehaviors \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=400 --end_ind=450 > logs_bash/gcg_optimization_transform_HarmfulBehaviors_gpt-oss-20b_start_400_end_450.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 nohup python examples/3_gcg.py transform \
  --dataset_name=HarmfulBehaviors \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=450 --end_ind=523 > logs_bash/gcg_optimization_transform_HarmfulBehaviors_gpt-oss-20b_start_450_end_523.log 2>&1 &

# catqa

CUDA_VISIBLE_DEVICES=0 nohup python examples/3_gcg.py transform \
  --dataset_name=CatQA \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=0 --end_ind=50 > logs_bash/gcg_optimization_transform_CatQA_gpt-oss-20b_start_0_end_50.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/3_gcg.py transform \
  --dataset_name=CatQA \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=50 --end_ind=100 > logs_bash/gcg_optimization_transform_CatQA_gpt-oss-20b_start_50_end_100.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python examples/3_gcg.py transform \
  --dataset_name=CatQA \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=100 --end_ind=150 > logs_bash/gcg_optimization_transform_CatQA_gpt-oss-20b_start_50_end_150.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 nohup python examples/3_gcg.py transform \
  --dataset_name=CatQA \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=150 --end_ind=200 > logs_bash/gcg_optimization_transform_CatQA_gpt-oss-20b_start_150_end_200.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python examples/3_gcg.py transform \
  --dataset_name=CatQA \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=200 --end_ind=250 > logs_bash/gcg_optimization_transform_CatQA_gpt-oss-20b_start_200_end_250.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 nohup python examples/3_gcg.py transform \
  --dataset_name=CatQA \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=250 --end_ind=300 > logs_bash/gcg_optimization_transform_CatQA_gpt-oss-20b_start_250_end_300.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 nohup python examples/3_gcg.py transform \
  --dataset_name=CatQA \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=300 --end_ind=350 > logs_bash/gcg_optimization_transform_CatQA_gpt-oss-20b_start_300_end_350.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python examples/3_gcg.py transform \
  --dataset_name=CatQA \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=350 --end_ind=400 > logs_bash/gcg_optimization_transform_CatQA_gpt-oss-20b_start_350_end_400.log 2>&1 &


# wait
CUDA_VISIBLE_DEVICES=1 nohup python examples/3_gcg.py transform \
  --dataset_name=CatQA \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=400 --end_ind=450 > logs_bash/gcg_optimization_transform_CatQA_gpt-oss-20b_start_400_end_450.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python examples/3_gcg.py transform \
  --dataset_name=CatQA \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=450 --end_ind=500 > logs_bash/gcg_optimization_transform_CatQA_gpt-oss-20b_start_450_end_500.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/3_gcg.py transform \
  --dataset_name=CatQA \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=500 --end_ind=550 > logs_bash/gcg_optimization_transform_CatQA_gpt-oss-20b_start_500_end_550.log 2>&1 &

# wait  done


# try 120b
#python examples/3_gcg.py transform \
#  --dataset_name=StrongReject \
#  --target_model_name=openai/gpt-oss-120b \
#  --start_ind=0 --end_ind=50
#
#python examples/3_gcg.py transform \
#  --dataset_name=StrongReject \
#  --target_model_name=openai/gpt-oss-120b \
#  --start_ind=0 --end_ind=50





# StrongREJECT

CUDA_VISIBLE_DEVICES=2 nohup python examples/3_gcg.py transform \
  --dataset_name=StrongReject \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=0 --end_ind=50 > logs_bash/gcg_optimization_transform_StrongReject_gpt-oss-20b_start_0_end_50.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 nohup python examples/3_gcg.py transform \
  --dataset_name=StrongReject \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=50 --end_ind=100 > logs_bash/gcg_optimization_transform_StrongReject_gpt-oss-20b_start_50_end_100.log 2>&1 &



# DISK ERROR below

CUDA_VISIBLE_DEVICES=0 nohup python examples/3_gcg.py transform \
  --dataset_name=StrongReject \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=130 --end_ind=150 > logs_bash/gcg_optimization_transform_StrongReject_gpt-oss-20b_start_130_end_150.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 nohup python examples/3_gcg.py transform \
  --dataset_name=StrongReject \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=151 --end_ind=200 > logs_bash/gcg_optimization_transform_StrongReject_gpt-oss-20b_start_151_end_200.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python examples/3_gcg.py transform \
  --dataset_name=StrongReject \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=230 --end_ind=250 > logs_bash/gcg_optimization_transform_StrongReject_gpt-oss-20b_start_201_end_250.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 nohup python examples/3_gcg.py transform \
  --dataset_name=StrongReject \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=277 --end_ind=315 > logs_bash/gcg_optimization_transform_StrongReject_gpt-oss-20b_start_277_end_315.log 2>&1 &



# JBB
CUDA_VISIBLE_DEVICES=0 nohup python examples/3_gcg.py transform \
  --dataset_name=JBBHarmfulBehaviors \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=0 --end_ind=50 > logs_bash/gcg_optimization_transform_JBBHarmfulBehaviors_gpt-oss-20b_start_0_end_50.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 nohup python examples/3_gcg.py transform \
  --dataset_name=JBBHarmfulBehaviors \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=50 --end_ind=101 > logs_bash/gcg_optimization_transform_JBBHarmfulBehaviors_gpt-oss-20b_start_50_end_101.log 2>&1 &