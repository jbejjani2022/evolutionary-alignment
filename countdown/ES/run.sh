#!/bin/bash
#SBATCH --job-name=es_countdown
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH -t 3-00:00:00
#SBATCH --mem=256G
#SBATCH -o output/job.%N.%j.out          # STDOUT
#SBATCH -e error/job.%N.%j.err           # STDERR
#SBATCH --mail-user=jbejjani@college.harvard.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0

# Load modules
module load python/3.10.13-fasrc01
module load cuda/12.9.1-fasrc01
module load cudnn/9.10.2.21_cuda12-fasrc01

# Activate conda environment
source activate /n/holylabs/LABS/sham_lab/Users/jbejjani/envs/evolutionary-alignment

accelerate launch \
    --num_processes 4 \
    --num_machines 1 \
    --machine_rank 0 \
    es_fine-tuning_countdown_iid.py \
    --train_samples 200 \
    --data_path /n/holylabs/LABS/sham_lab/Users/jbejjani/evolutionary-alignment/countdown/data/countdown.json \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --hf_cache_dir /n/netscratch/sham_lab/Everyone/jbejjani/hf_cache \
    --experiment_dir /n/netscratch/sham_lab/Everyone/jbejjani/evolutionary-alignment/countdown/ES/normal \
    --gpu_threads 1 \
    --max_new_tokens 1024 \
    --iterations 500 \
    --eval_interval 25 \
    --save_steps 250 \
    --log_wandb \
    --wandb_project es_accl_countdown \
    --wandb_entity KURE-Spring-25 \
    --population_size 30 \
    --sigma 0.001 \
    --alpha 0.0005 \
    --global_seed $SLURM_ARRAY_TASK_ID \
    --precision float16
