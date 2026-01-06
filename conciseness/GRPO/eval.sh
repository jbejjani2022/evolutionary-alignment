#!/bin/bash
#SBATCH --job-name=grpo_conciseness_eval
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=3:00:00
#SBATCH --mem=256G
#SBATCH --output=logs/grpo_conciseness_eval_%A_%a.log
#SBATCH --mail-user=jbejjani@college.harvard.edu
#SBATCH --mail-type=ALL

# Load modules
module load python/3.12.11-fasrc02
module load cuda/12.9.1-fasrc01
module load cudnn/9.10.2.21_cuda12-fasrc01

# Activate conda environment
mamba deactivate
mamba activate /n/holylabs/LABS/sham_lab/Users/jbejjani/envs/evolutionary-alignment

cd ..

# Set PyTorch memory allocator configuration for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for beta in 0.1 0.0464 0.0
do
    python conciseness_eval_grpo_seeds_sweep.py \
        --baseline_model_name Qwen/Qwen2.5-7B-Instruct \
        --hf_cache_dir /n/netscratch/sham_lab/Everyone/jbejjani/hf_cache \
        --precision bf16 \
        --max_new_tokens 128 \
        --num_samples 20 \
        --eval_data_path /n/holylabs/LABS/sham_lab/Users/jbejjani/evolutionary-alignment/conciseness/data/eval.jsonl \
        --print-examples \
        --output_json GRPO/evals/temp_0.7_beta${beta}.json \
        --seed 42 \
        --beta ${beta} \
        --temperature 1.0 \
        --top_p 1.0 \
        --do_sample
done
