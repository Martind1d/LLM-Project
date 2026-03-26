#!/bin/bash
#SBATCH -J hydra_log_test
#SBATCH --gres=gpu:L40s
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

mkdir -p slurm_logs
nvidia-smi -L
source ~/miniforge3/etc/profile.d/conda.sh

conda activate theenv

srun python -u mt/eval.py
