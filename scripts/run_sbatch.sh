#!/bin/bash
#SBATCH --job-name cot-bypass-strongreject
#SBATCH -o slurm_logs/cot-bypass-strongreject
#SBATCH -e slurm_logs/cot-bypass-strongreject
#SBATCH -N 1 # do not change
#SBATCH -p mcml-hgx-a100-80x4 # select a partition
#SBATCH --qos=mcml
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=400G

python examples/1_cot_bypass.py --dataset_name=StrongReject