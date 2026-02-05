#!/bin/bash
#SBATCH --job-name=es_math_eval
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=6:00:00
#SBATCH --mem=256G
#SBATCH --output=logs/es_math_eval_%A_%a.log
#SBATCH --mail-type=ALL

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules
module load python/3.12.11-fasrc02
module load cuda/12.9.1-fasrc01
module load cudnn/9.10.2.21_cuda12-fasrc01

# Activate environment
mamba deactivate
mamba activate /n/holylabs/LABS/sham_lab/Users/jbejjani/envs/evolutionary-alignment

cd ..

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Configuration
MODEL_PATH="${1:-/path/to/your/checkpoint}"  # Pass as first argument or set default
BASELINE_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
OUTPUT_DIR="./eval_results"

# Evaluation parameters
DATASETS="MATH500,AIME2024,Minerva,OlympiadBench"
NUM_SAMPLES=64
KMAX=64
TEMPERATURE=1.0

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run evaluation
python math_eval.py \
    --model_path ${MODEL_PATH} \
    --baseline_model ${BASELINE_MODEL} \
    --datasets ${DATASETS} \
    --num_samples ${NUM_SAMPLES} \
    --kmax ${KMAX} \
    --temperature ${TEMPERATURE} \
    --max_new_tokens 2048 \
    --output_dir ${OUTPUT_DIR} \
    --gpu_memory_utilization 0.9 \
    --seed 42

echo ""
echo "Evaluation complete. Results saved to ${OUTPUT_DIR}/"
