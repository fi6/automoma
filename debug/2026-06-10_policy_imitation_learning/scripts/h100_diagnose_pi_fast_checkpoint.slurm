#!/usr/bin/env bash
#SBATCH --partition=h100
#SBATCH --job-name=pifast_ckpt_diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/home/xinhai/projects/automoma/debug/2026-06-10_policy_imitation_learning/output/slurm/%x_%j.out
#SBATCH --error=/home/xinhai/projects/automoma/debug/2026-06-10_policy_imitation_learning/output/slurm/%x_%j.err

set -Eeuo pipefail

CHECKPOINT="${1:?Usage: sbatch h100_diagnose_pi_fast_checkpoint.slurm <checkpoint> <dataset-root> [indices]}"
DATASET_ROOT="${2:?Usage: sbatch h100_diagnose_pi_fast_checkpoint.slurm <checkpoint> <dataset-root> [indices]}"
INDICES="${3:-0,1,2,10,100,500,1000,2000,3000,4000}"

set +e
if [ -r /cm/local/apps/environment-modules/4.5.3/init/bash ]; then
    source /cm/local/apps/environment-modules/4.5.3/init/bash
fi
module load cuda12.8/toolkit/12.8.0 2>/dev/null
set -e

REPO_ROOT="/home/xinhai/projects/automoma"
TASK_DIR="$REPO_ROOT/debug/2026-06-10_policy_imitation_learning"
ENV_PREFIX="/mnt/nfs_project_a/xinhai/env/automoma-train"
PYTHON_BIN="$ENV_PREFIX/bin/python"

export PATH="$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="${HF_HOME:-/mnt/nfs_project_a/xinhai/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/nfs_project_a/xinhai/hf_home/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/mnt/nfs_project_a/xinhai/hf_home/transformers}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/third_party/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"

echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "checkpoint=$CHECKPOINT"
echo "dataset_root=$DATASET_ROOT"
echo "indices=$INDICES"
nvidia-smi || true

cd "$REPO_ROOT"
exec "$PYTHON_BIN" "$TASK_DIR/scripts/diagnose_pi_fast_checkpoint.py" \
    --checkpoint "$CHECKPOINT" \
    --dataset-root "$DATASET_ROOT" \
    --device cuda \
    --indices "$INDICES"
