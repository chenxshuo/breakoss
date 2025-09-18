#!/bin/bash
#SBATCH --job-name 400-450
#SBATCH -o slurm_logs/cat-qa-gcg-400-450
#SBATCH -e slurm_logs/cat-qa-gcg-400-450
#SBATCH -N 1 # do not change
##SBATCH -p mcml-hgx-a100-80x4 # select a partition
#SBATCH -p lrz-dgx-a100-80x8 # select a partition
##SBATCH --qos=mcml
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=400G



python examples/3_gcg.py transform \
  --dataset_name=CatQA \
  --target_model_name=openai/gpt-oss-20b \
  --start_ind=400 --end_ind=450

#python examples/3_gcg.py transform \
#  --dataset_name=CatQA \
#  --target_model_name=openai/gpt-oss-20b \
#  --start_ind=450 --end_ind=500
#python examples/3_gcg.py transform \
#  --dataset_name=CatQA \
#  --target_model_name=openai/gpt-oss-20b \
#  --start_ind=500 --end_ind=550

#  > logs_bash/gcg_optimization_transform_CatQA_gpt-oss-20b_start_400_end_450.log 2>&1 &
