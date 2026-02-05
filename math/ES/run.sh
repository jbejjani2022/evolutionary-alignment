#!/bin/bash
#SBATCH --job-name=es_math
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --output=logs/es_math_%A_%a.log
#SBATCH --mail-type=ALL
#SBATCH --array=0

# Create logs directory if it doesn't exist
mkdir -p logs

# Configuration
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
SIGMA=0.001
ALPHA=0.0005
POP_SIZE=30
NUM_ITERS=1000
SAVE_EVERY=200
BATCH_SIZE=200

# Output directory
OUTPUT_DIR="/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/math/es-ft-experiment"

python math/ES/es-fine-tuning_math.py \
    --policy_model_path ${MODEL} \
    --sigma ${SIGMA} \
    --alpha ${ALPHA} \
    --population_size ${POP_SIZE} \
    --num_iterations ${NUM_ITERS} \
    --save_every ${SAVE_EVERY} \
    --save_replay_log \
    --batch_size ${BATCH_SIZE} \
    --num_engines 4 \
    --cuda_devices "0,1,2,3" \
    --experiment_dir ${OUTPUT_DIR} \
    --train_dataset "DigitalLearningGmbH/MATH-lighteval" \
    --wandb_project "math_es_training" \
    --global_seed ${SLURM_ARRAY_TASK_ID}
