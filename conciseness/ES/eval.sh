#!/bin/bash
#SBATCH --job-name=es_conciseness_eval
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=3:00:00
#SBATCH --mem=256G
#SBATCH --output=logs/es_conciseness_eval_%A_%a.log
#SBATCH --mail-user=jbejjani@college.harvard.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0-3

# Load modules
module load python/3.12.11-fasrc02
module load cuda/12.9.1-fasrc01
module load cudnn/9.10.2.21_cuda12-fasrc01

# Activate conda environment
mamba deactivate
mamba activate /n/holylabs/LABS/sham_lab/Users/jbejjani/envs/evolutionary-alignment

cd ..

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Define ES hyperparameters
sigma=0.001
alpha=0.0005

python conciseness_eval_es_seeds_sweep.py \
        --baseline_model_name Qwen/Qwen2.5-7B-Instruct \
        --hf_cache_dir /n/netscratch/sham_lab/Everyone/jbejjani/hf_cache \
        --precision bf16 \
        --max_new_tokens 128 \
        --num_samples 20 \
        --eval_data_path data/eval.jsonl \
        --print-examples \
        --output_json ES/evals/alpha${alpha}_sigma${sigma}.json \
        --seed 42 \
        --sigma ${sigma} \
        --alpha ${alpha} \
        --do_sample
