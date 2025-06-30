#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --job-name=run_incremental_cifar_sap
#SBATCH --output=slurm_logs_sap/run_incremental_cifar-%A.out   # %A = master job ID, %a = array index
#SBATCH --error=slurm_logs_sap/run_incremental_cifar-%A.err
export PYTHONUNBUFFERED=TRUE

# activate your environment
module load cuda cudnn
module load python/3.11.0s-ixrhc3q
source ../.venv/bin/activate
# run
python3 incremental_cifar_experiment_jax.py --config ./cfg/shrink_and_perturb.json --verbose --experiment-index 0