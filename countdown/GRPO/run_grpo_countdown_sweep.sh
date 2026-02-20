#!/bin/bash
#SBATCH --job-name=trl_grpo_countdown_1p5b
#SBATCH --output=logs/trl_grpo_countdown_1p5b_%A_%a.log
#SBATCH --error=logs/trl_grpo_countdown_1p5b_%A_%a.err
#SBATCH --account=kempner_sham_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --mem=256GB
#SBATCH --partition=kempner_h100
#SBATCH --constraint=h100
#SBATCH --mail-user=jbejjani@college.harvard.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0-0%1

set -euo pipefail

# Load modules
module load python/3.10.13-fasrc01
module load cuda/12.9.1-fasrc01
module load cudnn/9.10.2.21_cuda12-fasrc01

# Activate conda environment
source activate /n/holylabs/LABS/sham_lab/Users/jbejjani/envs/evolutionary-alignment

CONFIG=${CONFIG:-grpo_countdown_trl.yaml}

BETAS=(5e-3)
LRS=(1e-6)
SEEDS=(${SEEDS:-0})

NB=${#BETAS[@]}
NS=1
IDX=${SLURM_ARRAY_TASK_ID}

BIDX=$(( IDX % NB ))
SIDX=0
BETA=${BETAS[$BIDX]}
LR=${LRS[$BIDX]}
SEED=${SEEDS[$SIDX]}

echo "[SLURM] idx=$IDX -> beta=$BETA lr=$LR seed=$SEED"

mkdir -p logs

# Minimal accelerate config (single node, ZeRO-2)
ACCELERATE_CONFIG=accelerate_trl_grpo_countdown.yaml
cat > $ACCELERATE_CONFIG << ACC
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
num_machines: 1
machine_rank: 0
gpu_ids: all
use_cpu: false
ACC

# Env
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export WANDB__SERVICE_WAIT=300
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8

DS_CFG=$(realpath ds_zero2.json)

accelerate launch \
  --config_file $ACCELERATE_CONFIG \
  --num_processes 4 \
  --deepspeed_config_file $DS_CFG \
  train_grpo_countdown_trl.py \
  --config ${CONFIG} \
  --beta ${BETA} \
  --learning_rate ${LR} \
  --seed ${SEED} \
  --chat_template
