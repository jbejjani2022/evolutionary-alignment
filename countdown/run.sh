#!/bin/bash
#SBATCH --job-name=es_countdown_accl
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=1:00:00
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

# Multi-GPU run (one vLLM engine per GPU)
python es_accl_static.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --sigma 0.001 \
  --alpha 0.0005 \
  --population_size 30 \
  --num_engines 4 \
  --cuda_devices "0,1,2,3" \
  --num_iterations 500 \
  --eval_interval 25 \
  --max_new_tokens 1024 \
  --precision float16 \
  --global_seed $SLURM_ARRAY_TASK_ID \
  --experiment_dir /n/netscratch/sham_lab/Everyone/jbejjani/evolutionary-alignment/countdown/ES/accl \
  --hf_cache_dir /n/netscratch/sham_lab/Everyone/jbejjani/hf_cache \
  --wandb_project es_accl_countdown \
  --wandb_entity KURE-Spring-25
