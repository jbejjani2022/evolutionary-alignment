#!/usr/bin/env python
"""
Evolution Strategies fine-tuning for language model conciseness.

Uses ES to perturb model weights with Gaussian noise, evaluates perturbations on
question-answer pairs, and updates weights based on reward: -|len(generated) - len(target)|.

Features: Distributed training (Accelerate), multi-threaded evaluation, wandb logging,
checkpoint saving, memory-efficient batching.

Dataset: JSONL at 'conciseness/data/train.jsonl' with {"question": str, "answer": str} format.

Args:
    --model_name: HF model ID (default: 'Qwen/Qwen2.5-3B-Instruct')
    --output_dir: Output directory for checkpoints
    --iterations: ES iterations (default: 1000)
    --population_size: Perturbations per iteration (default: 30)
    --sigma: Perturbation noise scale (default: 0.001)
    --alpha: Learning rate (default: 0.0005)
    --gpu_threads: Threads per GPU (default: 4)
    --max_new_tokens: Max generation length (default: 100)
    --wandb_project: W&B project name

Output: Checkpoints and final model saved to specified output_dir
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import numpy as np
import copy
import os
import argparse
from accelerate import Accelerator
import time
import torch.multiprocessing as mp
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import math
import gc
import json
import wandb

logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-3B-Instruct')
parser.add_argument('--hf_cache_dir', type=str, default='hf_cache')
parser.add_argument('--output_dir', type=str, default='/n/netscratch/sham_lab/Everyone/jbejjani/evolutionary-alignment/checkpoints', help='Output directory for checkpoints and final model')
parser.add_argument('--precision', type=str, default='bf16')
parser.add_argument('--gpu_threads', type=int, default=4, help='Number of parallel threads per GPU')
parser.add_argument('--verbose', action='store_true', help='Print verbose logs')
parser.add_argument('--save_steps', type=int, default=100, help='Save checkpoint every n iterations')
parser.add_argument('--iterations', type=int, default=1000, help='Number of ES iterations (generations)')
parser.add_argument('--population_size', type=int, default=30, help='Population size (number of perturbations per iteration)')
parser.add_argument('--sigma', type=float, default=0.001, help='Standard deviation for weight perturbations (noise scale)')
parser.add_argument('--alpha', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--initial_seed', type=int, default=33, help='Initial random seed')
parser.add_argument('--do_sample', default=False, action='store_true', help='Whether sampling is allowed in generating tokens, default to be not allowed (greedy decoding for ES)')
parser.add_argument('--max_new_tokens', type=int, default=100, help='Maximum number of tokens allowed to be generated')
parser.add_argument('--wandb_project', type=str, default='es-fine-tuning-conciseness', help='W&B project name')
parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name (default: auto-generated)')
parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity name')
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
parser.add_argument('--eval_jsonl', type=str, default='../data/eval.jsonl',
                    help='Path to eval JSONL with {"question","answer"} lines (default: ../data/eval.jsonl).')
parser.add_argument('--eval_steps', type=int, default=50,
                    help='Evaluate every N iterations. Set <=0 to disable. (default: 50)')
parser.add_argument('--eval_log_completions', type=int, default=8,
                    help='How many eval completions to log to W&B each eval. (default: 8)')
parser.add_argument('--eval_log_max_chars', type=int, default=500,
                    help='Truncate logged question/target/generated strings to this many chars. (default: 500)')

args = parser.parse_args()


# Hyperparameters for ES
NUM_ITERATIONS = args.iterations             # Number of ES iterations (generations)
POPULATION_SIZE = args.population_size       # Population size (number of perturbations per iteration)
SIGMA = args.sigma                           # Standard deviation for weight perturbations (noise scale)
ALPHA = args.alpha                           # Learning rate
max_new_tokens = args.max_new_tokens         # Maximum number of tokens allowed to be generated
do_sample = args.do_sample                   # Whether sampling is allowed in generating tokens, default to be not allowed (greedy decoding for ES)
initial_seed = args.initial_seed             # Initial random seed


# --- Load Dataset from JSONL File ---
dataset = []
with open('../data/train.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            dataset.append((data['question'], data['answer']))
            
# --- Load Eval Dataset from JSONL File ---
eval_dataset = []
_eval_path = args.eval_jsonl

def _load_jsonl_pairs(path):
    pairs = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                pairs.append((d["question"], d["answer"]))
    return pairs

if os.path.exists(_eval_path):
    eval_dataset = _load_jsonl_pairs(_eval_path)
else:
    eval_dataset = None


# Global tokenizer reference (set in main, used for chat template)
_tokenizer = None


def format_prompt_with_chat_template(question: str) -> str:
    """Apply chat template to raw question for Qwen2.5-Instruct."""
    global _tokenizer
    messages = [{"role": "user", "content": question}]
    formatted = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return formatted


def clean_completion(text: str) -> str:
    """Remove EOS/chat-end tokens and strip whitespace."""
    global _tokenizer
    text = text.strip()
    # Qwen2.5 uses <|im_end|> as the chat end token
    im_end_token = "<|im_end|>"
    eos_token = _tokenizer.eos_token or ""
    
    for token in [im_end_token, eos_token]:
        if token and text.endswith(token):
            text = text[:-len(token)].strip()
    return text


def compute_reward(generated_text: str, target_text: str) -> float:
    """
    Reward: negative absolute difference in character length.
    generated_text should be the completion only (after cleaning).
    """
    gen_clean = clean_completion(generated_text)
    target_clean = target_text.strip()
    return -abs(len(gen_clean) - len(target_clean))


def get_save_dir(model_name, initial_seed, iteration, dataset_size, args, is_final=False):
    """Generate consistent save directory path for checkpoints and final model"""
    question_num = dataset_size
    suffix = "final" if is_final else "checkpoint"
    save_dir = os.path.join(
        args.output_dir,
        f"conciseness/{model_name}/{initial_seed}",
        f"es_random_seed{initial_seed}_pop{POPULATION_SIZE}_iter{iteration}_sigma{SIGMA}_alpha{ALPHA}_{args.precision}_threads{args.gpu_threads}_question_num{question_num}_{suffix}"
    )
    return save_dir


def save_model_checkpoint(model, tokenizer, iteration, model_name, initial_seed, args, dataset_size, wandb_run=None):
    """Save model checkpoint at specified iteration"""
    save_dir = get_save_dir(model_name, initial_seed, iteration, dataset_size, args, is_final=False)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving checkpoint at iteration {iteration} to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Checkpoint saved successfully.")
    if wandb_run is not None:
        wandb_run.log({"checkpoint_saved": True, "checkpoint_iteration": iteration, "checkpoint_path": save_dir}, step=iteration)


def force_memory_cleanup():
    """Force aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def evaluate_model(model, tokenizer, input_text, target_text, accelerator, seed_idx=None, thread_id=None, verbose=False, return_text=False):
    """
    Generate a response from the model given an input (single or batch) and compute rewards.
    
    - input_text should be RAW questions (chat template applied here)
    - Returns completion only (prompt stripped), cleaned of special tokens
    """
    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} evaluating seed {seed_idx}")

    # Handle both single input and batch input
    is_batch = isinstance(input_text, list)
    input_texts_raw = input_text if is_batch else [input_text]
    target_texts = target_text if is_batch else [target_text]

    # Apply chat template to all inputs
    input_texts_formatted = [format_prompt_with_chat_template(q) for q in input_texts_raw]

    # Batch tokenization
    tokenized_inputs = tokenizer(
        input_texts_formatted, 
        return_tensors="pt", 
        padding=True,
        add_special_tokens=False  # Chat template already includes special tokens
    )
    input_ids = tokenized_inputs["input_ids"].to(accelerator.device)
    attention_mask = tokenized_inputs["attention_mask"].to(accelerator.device)
    
    # Store input length for completion extraction
    prompt_length = input_ids.shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            input_ids, 
            attention_mask=attention_mask, 
            max_new_tokens=max_new_tokens, 
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)

    # Decode completion only (strip the prompt)
    generated_texts = []
    for i in range(len(input_texts_formatted)):
        # Get only the new tokens (completion)
        completion_ids = outputs[i][prompt_length:]
        
        try:
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=False)
        except TypeError:
            tokens = tokenizer.convert_ids_to_tokens(completion_ids)
            filtered = [t for t in tokens if t is not None]
            completion_text = tokenizer.convert_tokens_to_string(filtered)
        
        # Clean the completion (remove <|im_end|>, EOS, etc.)
        completion_text = clean_completion(completion_text)
        generated_texts.append(completion_text)

    del input_ids, outputs
    torch.cuda.empty_cache()

    # Compute rewards using cleaned completions
    rewards = [compute_reward(gen_text, tgt_text) for gen_text, tgt_text in zip(generated_texts, target_texts)]

    if return_text:
        return rewards, generated_texts
    else:
        return rewards


def _truncate(s: str, n: int) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else (s[:n] + "…")


def evaluate_eval_set_distributed(model, tokenizer, eval_dataset, accelerator, verbose=False):
    """
    Evaluate current (unperturbed) model on eval_dataset across processes.
    Returns (mean, min, max, std, rewards_list).
    """
    if eval_dataset is None or len(eval_dataset) == 0:
        return None

    n = len(eval_dataset)
    device = accelerator.device

    # Assign eval examples by index to processes
    local_indices = [i for i in range(n) if (i % accelerator.num_processes) == accelerator.process_index]
    local_inputs = [eval_dataset[i][0] for i in local_indices]
    local_targets = [eval_dataset[i][1] for i in local_indices]

    if verbose:
        print(f"[Eval] process {accelerator.process_index} evaluating {len(local_indices)}/{n} examples")

    # Compute local rewards (batched)
    local_rewards = []
    if len(local_inputs) > 0:
        local_rewards = evaluate_model(
            model, tokenizer,
            local_inputs, local_targets,
            accelerator,
            seed_idx=None, thread_id=None,
            verbose=False,
            return_text=False
        )

    # Build a full rewards tensor and reduce across processes
    rewards_tensor = torch.zeros(n, device=device, dtype=torch.float32)
    for idx, r in zip(local_indices, local_rewards):
        rewards_tensor[idx] = float(r)

    if accelerator.num_processes > 1:
        torch.distributed.all_reduce(rewards_tensor, op=torch.distributed.ReduceOp.SUM)

    rewards = rewards_tensor.detach().cpu().numpy()
    mean = float(rewards.mean())
    mn = float(rewards.min())
    mx = float(rewards.max())
    std = float(rewards.std())

    del rewards_tensor
    force_memory_cleanup()

    return mean, mn, mx, std, rewards


def log_eval_completions_to_wandb(model, tokenizer, eval_dataset, accelerator, wandb_run, step, max_rows, max_chars, seed):
    """
    Logs a small table of eval completions to wandb (main process only).
    """
    if wandb_run is None or eval_dataset is None or len(eval_dataset) == 0:
        return
    if not accelerator.is_main_process:
        return

    n = len(eval_dataset)
    k = max(1, min(max_rows, n))

    rng = np.random.RandomState(int(seed) + int(step))
    sample_indices = rng.choice(n, size=k, replace=False).tolist()

    input_texts = [eval_dataset[i][0] for i in sample_indices]
    target_texts = [eval_dataset[i][1] for i in sample_indices]

    rewards, generated_texts = evaluate_model(
        model, tokenizer,
        input_texts, target_texts,
        accelerator,
        seed_idx=None, thread_id=None,
        verbose=False,
        return_text=True
    )

    table = wandb.Table(columns=["idx", "question", "target", "generated", "gen_len", "target_len", "reward"])
    for i, q, t, g, r in zip(sample_indices, input_texts, target_texts, generated_texts, rewards):
        table.add_data(
            int(i),
            _truncate(q, max_chars),
            _truncate(t, max_chars),
            _truncate(g, max_chars),
            len(g),
            len(t),
            float(r),
        )

    wandb_run.log({"eval/completions": table}, step=step)


def process_seed(seed_args):
    """Function to process a single seed, used for thread pool"""
    seed_idx, seed, model, tokenizer, accelerator, thread_id, verbose = seed_args

    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} processing seed {seed_idx} (value: {seed})")

    # Weight perturbation
    seed_shift = 0
    for name, param in model.named_parameters():
        gen = torch.Generator(device=param.device)
        gen.manual_seed(int(seed+seed_shift))
        seed_shift += 1

        noise = torch.randn(
            param.shape,
            generator=gen,
            device=param.device,
            dtype=param.dtype
        )
        param.data.add_(SIGMA * noise)

    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)

    # Evaluate all prompts with perturbed weights in batch
    # Pass raw questions - chat template applied inside evaluate_model
    input_texts = [input_text for input_text, _ in dataset]
    target_texts = [target_text for _, target_text in dataset]
    rewards = evaluate_model(model, tokenizer, input_texts, target_texts, accelerator,
                           seed_idx=seed_idx, thread_id=thread_id, verbose=verbose, return_text=False)
    total_reward = sum(rewards)

    # Restore original weights
    seed_shift = 0
    for name, param in model.named_parameters():
        gen = torch.Generator(device=param.device)
        gen.manual_seed(int(seed+seed_shift))
        seed_shift += 1

        noise = torch.randn(
            param.shape,
            generator=gen,
            device=param.device,
            dtype=param.dtype
        )
        param.data.add_(-SIGMA * noise)

    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)

    average_reward = total_reward / len(dataset)

    force_memory_cleanup()

    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} completed seed {seed_idx} with reward {average_reward:.4f}")

    return seed_idx, average_reward


def verify_setup(tokenizer, accelerator, args):
    """
    Print verification info for debugging. Call on rank 0 only.
    Matches GRPO script output format for easy comparison.
    """
    print("\n" + "=" * 60)
    print("SETUP VERIFICATION")
    print("=" * 60)
    
    # Tokenizer checks
    print(f"\n[Tokenizer]")
    print(f"  pad_token: {repr(tokenizer.pad_token)} (id={tokenizer.pad_token_id})")
    print(f"  eos_token: {repr(tokenizer.eos_token)} (id={tokenizer.eos_token_id})")
    print(f"  padding_side: {tokenizer.padding_side}")
    
    # Check for Qwen chat tokens
    im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    print(f"  <|im_end|> token ids: {im_end_id}")
    
    # Sample prompt check
    if len(dataset) > 0:
        sample_q, sample_a = dataset[0]
        formatted = format_prompt_with_chat_template(sample_q)
        
        print(f"\n[Sample Formatted Prompt]")
        print(f"  Prompt:")
        print(f"    {repr(formatted)[:200]}...")
        print(f"  Answer: {repr(sample_a)}")
        
        # Verify chat template applied
        if "<|im_start|>" not in formatted:
            print("  ⚠️  WARNING: Prompt missing <|im_start|> - chat template may not be applied!")
        else:
            print("  ✓ Chat template detected")
        
        if "<|im_start|>assistant" in formatted:
            print("  ✓ Generation prompt added (ends with assistant turn)")
        else:
            print("  ⚠️  WARNING: Missing assistant generation prompt!")
    
    # Training config summary
    print(f"\n[Training Config]")
    print(f"  algorithm: Evolution Strategies")
    print(f"  sigma (noise scale): {SIGMA}")
    print(f"  alpha (learning rate): {ALPHA}")
    print(f"  population_size: {POPULATION_SIZE}")
    print(f"  iterations: {NUM_ITERATIONS}")
    print(f"  max_new_tokens: {max_new_tokens}")
    print(f"  do_sample: {do_sample}")
    print(f"  seed: {initial_seed}")
    
    # Dataset info
    print(f"\n[Dataset]")
    print(f"  train_size: {len(dataset)}")
    print(f"  eval_size: {len(eval_dataset) if eval_dataset else 0}")
    print(f"  eval_steps: {args.eval_steps}")
    
    # Distributed info
    print(f"\n[Distributed]")
    print(f"  num_processes: {accelerator.num_processes}")
    print(f"  gpu_threads: {args.gpu_threads}")
    
    print("\n" + "=" * 60 + "\n")


def verify_completion_extraction(model, tokenizer, accelerator):
    """
    Run a single test generation to verify completion extraction is working.
    Call on rank 0 only.
    """
    if len(dataset) == 0:
        return
    
    print("[Completion Extraction Test]")
    
    sample_q, sample_a = dataset[0]
    
    # Get completion using our evaluate_model function
    rewards, completions = evaluate_model(
        model, tokenizer,
        [sample_q], [sample_a],
        accelerator,
        seed_idx=None, thread_id=None,
        verbose=False,
        return_text=True
    )
    
    completion = completions[0]
    reward = rewards[0]
    
    print(f"  Input question: {repr(sample_q)}")
    print(f"  Target answer: {repr(sample_a)} (len={len(sample_a)})")
    print(f"  Generated completion: {repr(completion[:100])}{'...' if len(completion) > 100 else ''}")
    print(f"  Completion length: {len(completion)}")
    print(f"  Reward: {reward}")
    
    # Sanity check: completion should NOT contain the prompt
    if sample_q in completion:
        print("  ⚠️  WARNING: Completion contains the input question - extraction may be broken!")
    else:
        print("  ✓ Completion does not contain input (extraction working)")
    
    # Check if completion contains chat template artifacts
    if "<|im_start|>" in completion:
        print("  ⚠️  WARNING: Completion contains <|im_start|> - may need additional cleaning")
    else:
        print("  ✓ No chat template artifacts in completion")
    
    print()


# --- Main Evolution Strategies Loop ---
def main():
    global _tokenizer
    
    accelerator = Accelerator()

    # Initialize wandb only on main process
    wandb_run = None
    if accelerator.is_main_process and not args.no_wandb:
        run_name = args.wandb_run_name
        if run_name is None:
            run_name = f"es_seed{args.initial_seed}_sigma{SIGMA}_alpha{ALPHA}_pop{POPULATION_SIZE}"
        
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            entity=args.wandb_entity,
            config={
                "algorithm": "evolution_strategies",
                "model_name": args.model_name,
                "initial_seed": args.initial_seed,
                "sigma": SIGMA,
                "alpha": ALPHA,
                "population_size": POPULATION_SIZE,
                "iterations": NUM_ITERATIONS,
                "save_steps": args.save_steps,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "precision": args.precision,
                "gpu_threads": args.gpu_threads,
                "num_processes": accelerator.num_processes,
                "dataset_size": len(dataset),
                "eval_dataset_size": len(eval_dataset) if eval_dataset else 0,
            }
        )
        wandb_run = wandb.run

    # Load model
    model_name = args.model_name
    hf_cache_dir = args.hf_cache_dir

    if accelerator.is_main_process:
        print(f"Loading model {model_name}...")

    model_list = []
    for model_index in range(args.gpu_threads):
        model_list.append(AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=hf_cache_dir,
            device_map={"": accelerator.process_index},
            torch_dtype=torch.float16 if args.precision == 'fp16' else (torch.bfloat16 if args.precision == 'bf16' else torch.float32),
        ))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=hf_cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Set global tokenizer for chat template functions
    _tokenizer = tokenizer

    if accelerator.is_main_process:
        print("Model loaded successfully\n")
        
        # Run verification checks
        verify_setup(tokenizer, accelerator, args)
        
        # Test completion extraction with a real generation
        verify_completion_extraction(model_list[0], tokenizer, accelerator)

    for model in model_list:
        model.eval()

    force_memory_cleanup()

    # Synchronize before starting training
    if accelerator.num_processes > 1:
        torch.distributed.barrier()

    if accelerator.is_main_process:
        print("=" * 60)
        print("STARTING TRAINING")
        print("=" * 60 + "\n")

    training_start_time = time.time()
    np.random.seed(initial_seed)

    for iteration in range(NUM_ITERATIONS):
        iter_start_time = time.time()
        force_memory_cleanup()

        if args.verbose:
            print(f"Process {accelerator.process_index} starting iteration {iteration + 1}/{NUM_ITERATIONS}")

        # Generate seeds on main process only
        if accelerator.is_main_process:
            if args.verbose:
                print(f"Main process {accelerator.process_index} generating seeds")
            seeds = np.random.randint(0, 2**30, size=POPULATION_SIZE, dtype=np.int64).tolist()
            seeds_tensor = torch.tensor(seeds, device=accelerator.device)
        else:
            if args.verbose:
                print(f"Worker process {accelerator.process_index} waiting for seeds")
            seeds_tensor = torch.zeros(POPULATION_SIZE, dtype=torch.long, device=accelerator.device)

        if accelerator.num_processes > 1:
            torch.distributed.broadcast(seeds_tensor, src=0)
        seeds = seeds_tensor.cpu().tolist()

        if args.verbose:
            print(f"Process {accelerator.process_index} received seeds")

        # Assign seeds to each process
        local_seeds = []
        for seed_idx, seed in enumerate(seeds):
            if seed_idx % accelerator.num_processes == accelerator.process_index:
                local_seeds.append((seed_idx, seed))

        if args.verbose:
            print(f"Process {accelerator.process_index} assigned {len(local_seeds)} seeds: {[idx for idx, _ in local_seeds]}")

        # Process seeds in batches
        local_rewards = []
        batch_size = max(1, min(args.gpu_threads, len(local_seeds)))

        for batch_start in range(0, len(local_seeds), batch_size):
            batch_end = min(batch_start + batch_size, len(local_seeds))
            batch_seeds = local_seeds[batch_start:batch_end]

            with ThreadPoolExecutor(max_workers=len(batch_seeds)) as executor:
                thread_args = []
                for thread_id, (seed_idx, seed) in enumerate(batch_seeds):
                    thread_args.append((seed_idx, seed, model_list[thread_id], tokenizer, accelerator, thread_id, args.verbose))

                results = list(executor.map(process_seed, thread_args))
                local_rewards.extend(results)

            force_memory_cleanup()

        # Collect rewards from all processes
        all_rewards = torch.zeros(POPULATION_SIZE, device=accelerator.device)

        for seed_idx, reward in local_rewards:
            all_rewards[seed_idx] = reward

        if accelerator.num_processes > 1:
            torch.distributed.all_reduce(all_rewards, op=torch.distributed.ReduceOp.SUM)

        rewards = all_rewards.cpu().tolist()
        del all_rewards
        force_memory_cleanup()

        # Normalize rewards
        rewards_tensor = np.array(rewards, dtype=np.float32)
        rewards_normalized = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        # Update model weights
        if args.verbose:
            print(f"Process {accelerator.process_index} updating model weights")
        original_model = model_list[0]
        seed_shift = 0
        for name, param in original_model.named_parameters():
            gen = torch.Generator(device=param.device)
            update = torch.zeros_like(param)
            for seed_idx in range(POPULATION_SIZE):
                r_norm = rewards_normalized[seed_idx]
                seed = seeds[seed_idx]
                gen.manual_seed(int(seed+seed_shift))

                noise = torch.randn(
                    param.shape,
                    generator=gen,
                    device=param.device,
                    dtype=param.dtype
                )
                noise.mul_(float(r_norm))
                update.add_(noise)
                del noise
            update.div_(POPULATION_SIZE)
            param.data.add_(ALPHA * update)
            torch.cuda.empty_cache()
            seed_shift += 1

        # Sync weights across model copies
        for model_idx in range(1, len(model_list)):
            original_model_tmp = model_list[model_idx]
            for name, param in original_model_tmp.named_parameters():
                param.data.copy_(original_model.get_parameter(name).data.clone())

        if torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)

        force_memory_cleanup()
        
        # --- Periodic Eval ---
        eval_metrics = None
        if eval_dataset is not None and args.eval_steps > 0 and ((iteration + 1) % args.eval_steps == 0):
            eval_metrics = evaluate_eval_set_distributed(
                original_model, tokenizer, eval_dataset, accelerator, verbose=args.verbose
            )

            if eval_metrics is not None:
                eval_mean, eval_min, eval_max, eval_std, _ = eval_metrics

                if accelerator.is_main_process:
                    print(f"[Eval] Iteration {iteration + 1}: "
                          f"Mean {eval_mean:.2f}  Min {eval_min:.2f}  Max {eval_max:.2f}  Std {eval_std:.2f}")

                    if wandb_run is not None:
                        wandb_run.log({
                            "eval/reward/mean": eval_mean,
                            "eval/reward/min": eval_min,
                            "eval/reward/max": eval_max,
                            "eval/reward/std": eval_std,
                            "eval/size": len(eval_dataset),
                        }, step=iteration + 1)

                        log_eval_completions_to_wandb(
                            original_model, tokenizer, eval_dataset, accelerator,
                            wandb_run=wandb_run,
                            step=iteration + 1,
                            max_rows=int(args.eval_log_completions),
                            max_chars=int(args.eval_log_max_chars),
                            seed=int(initial_seed),
                        )

        iter_time = time.time() - iter_start_time

        mean_reward = rewards_tensor.mean().item()
        min_reward = rewards_tensor.min().item()
        max_reward = rewards_tensor.max().item()
        std_reward = rewards_tensor.std().item()

        del rewards_tensor, rewards_normalized
        force_memory_cleanup()

        if accelerator.is_main_process:
            print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}, Time: {iter_time:.2f}s, Mean: {mean_reward:.2f}, Min: {min_reward:.2f}, Max: {max_reward:.2f}")
            print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}MB allocated, {torch.cuda.max_memory_allocated() / 1024**2:.2f}MB peak")

            if wandb_run is not None:
                wandb_run.log({
                    "iteration": iteration + 1,
                    "reward/mean": mean_reward,
                    "reward/min": min_reward,
                    "reward/max": max_reward,
                    "reward/std": std_reward,
                    "time/iteration": iter_time,
                    "gpu_memory/allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "gpu_memory/peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
                }, step=iteration + 1)

            # Save checkpoint every save_steps iterations
            if args.save_steps > 0 and (iteration + 1) % args.save_steps == 0:
                save_model_checkpoint(
                    original_model, tokenizer, iteration + 1,
                    model_name, initial_seed, args, len(dataset), wandb_run
                )

    total_time = time.time() - training_start_time

    # Save the final model
    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
        save_dir = get_save_dir(model_name, initial_seed, NUM_ITERATIONS, len(dataset), args, is_final=True)
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving final model to {save_dir}...")
        original_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Final model saved successfully.")
        print("=" * 60)
        
        if wandb_run is not None:
            wandb_run.log({
                "final_model_saved": True,
                "final_model_path": save_dir,
                "total_training_time_seconds": total_time,
                "total_training_time_minutes": total_time / 60,
            }, step=NUM_ITERATIONS)
            wandb_run.finish()


if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore"
    mp.set_start_method('spawn', force=True)
    main()
