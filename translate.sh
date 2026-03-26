#!/bin/bash
#SBATCH -J hydra_log_test
#SBATCH --gres=gpu:L40s
#SBATCH --cpus-per-task=4
#SBATCH --output=translate_logs/%x_%j.out
#SBATCH --error=translate_logs/%x_%j.err

set -euo pipefail

mkdir -p translate_logs

# Optional but helpful
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

source ~/.envs/en-sv-mt/bin/activate

OUTBASE="${SCRATCH:-$PWD}/translate_outputs"

python -u mt/translate.py \
  hydra.run.dir="${OUTBASE}/${SLURM_JOB_ID}"
