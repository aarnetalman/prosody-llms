#!/bin/bash
# Run one line of a job file per array task: a generic template for any Slurm cluster.
#
#   python scripts/reproduce.py list probe-linear > jobs.txt
#   sbatch --array=1-$(wc -l < jobs.txt) --account=<account> --partition=<gpu partition> \
#     slurm/array.sbatch jobs.txt
#
# Adjust the resources below (or override them on the sbatch command line). The largest
# models need one GPU with about 95 GB of memory; the commands themselves pick the GPU.
# Set PROSODY_DATA, PROSODY_RUNS and HF_HOME before submitting (they are exported to
# the job), and activate the Python environment in ~/.bashrc or below.
#SBATCH --job-name=prosody-llms
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
JOBS=${1:?usage: sbatch --array=1-N slurm/array.sbatch JOBFILE}
cd "${SLURM_SUBMIT_DIR:-.}"
# source .venv/bin/activate                 # uncomment if the environment is not active
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}  # models were downloaded beforehand (group "models")
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD=$(sed -n "${SLURM_ARRAY_TASK_ID:?run as an array job}p" "$JOBS")
[[ -n $CMD ]] || { echo "no line $SLURM_ARRAY_TASK_ID in $JOBS" >&2; exit 1; }
echo "task $SLURM_ARRAY_TASK_ID: $CMD"
eval "$CMD"
