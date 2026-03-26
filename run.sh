#!/bin/bash
#SBATCH -J hydra_log_test
#SBATCH --gres=gpu:L40s
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

mkdir -p slurm_logs

# Optional but helpful
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

source ~/.envs/en-sv-mt/bin/activate

OUTBASE="${SCRATCH:-$PWD}/hydra_outputs"

srun python -u mt/train.py \
  hydra.run.dir="${OUTBASE}/${SLURM_JOB_ID}"
