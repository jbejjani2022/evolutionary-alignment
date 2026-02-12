#!/bin/bash
#SBATCH --job-name=es_countdown_eval
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=3:00:00
#SBATCH --mem=256G
#SBATCH -o output/job.%N.%j.out          # STDOUT
#SBATCH -e error/job.%N.%j.err           # STDERR
#SBATCH --mail-user=jbejjani@college.harvard.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0

# Load modules
module load python/3.12.11-fasrc02
module load cuda/12.9.1-fasrc01
module load cudnn/9.10.2.21_cuda12-fasrc01

# Activate conda environment
source activate /n/holylabs/LABS/sham_lab/Users/jbejjani/envs/evolutionary-alignment

for model in "Qwen/Qwen2.5-0.5B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-3B-Instruct" "Qwen/Qwen2.5-7B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-3B-Instruct"; do
    python eval_countdown_vllm.py \
        --model_id ${model} \
        --eval_data_path "/n/holylabs/LABS/sham_lab/Users/jbejjani/evolutionary-alignment/countdown/data/countdown.json" \
        --eval_samples 2000 \
        --eval_offset -2000 \
        --max_new_tokens 1024 \
        --batch_size 1024 \
        --save_responses \
        --show_examples 5 \
        --hf_cache_dir /n/netscratch/sham_lab/Everyone/jbejjani/hf_cache \
        --output_dir /n/netscratch/sham_lab/Everyone/jbejjani/evolutionary-alignment/countdown/ES/accl/eval \
        --dtype float16 \
        --seed 0
done;

checkpoint="Todo"

python eval_countdown_vllm.py \
    --model_id "Qwen/Qwen2.5-3B-Instruct" \
    --trained_model_path ${checkpoint} \
    --eval_data_path "/n/holylabs/LABS/sham_lab/Users/jbejjani/evolutionary-alignment/countdown/data/countdown.json" \
    --eval_samples 2000 \
    --eval_offset -2000 \
    --max_new_tokens 1024 \
    --batch_size 1024 \
    --save_responses \
    --show_examples 5 \
    --output_dir /n/netscratch/sham_lab/Everyone/jbejjani/evolutionary-alignment/countdown/ES/accl/eval \
    --dtype float16 \
    --seed 0
