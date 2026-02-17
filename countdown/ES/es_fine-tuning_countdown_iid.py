"""
ES (Evolution Strategies) fine-tuning for the Countdown math task.
Baseline (non-accelerated) version using HuggingFace generate + Accelerate
for multi-GPU data parallelism.

NOTE on data loading: The JSON data file contains a pre-baked "context"
field with raw (un-templated) prompts. This script ignores that field and
always constructs prompts via _process_context(), which applies the model's
chat template with proper special tokens. Any separate evaluation scripts
should do the same — do NOT pass the JSON "context" field directly to the model.
"""

import argparse
import gc
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch
import torch.multiprocessing as mp
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging

# Ensure repo root is on sys.path so `countdown.*` imports work regardless of cwd
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Also set PYTHONPATH for accelerate's subprocesses
existing = os.environ.get("PYTHONPATH", "")
if _REPO_ROOT not in existing.split(os.pathsep):
    os.environ["PYTHONPATH"] = _REPO_ROOT + (os.pathsep + existing if existing else "")

from countdown.countdown_task import reward_function

logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

# ---------------------------------------------------------------------------
# Prompt construction — aligned with es_accl_static.py (GRPOZero format)
# ---------------------------------------------------------------------------
SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)
USER_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. "
    "And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
)
RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"

# ---------------------------------------------------------------------------
# Default Hyperparameters — aligned with es_accl_static.py
# ---------------------------------------------------------------------------
SIGMA = 0.001
ALPHA = 0.0005
POPULATION_SIZE = 30
NUM_ITERATIONS = 500
TRAIN_SAMPLES = 200
MAX_NEW_TOKENS = 1024
GLOBAL_SEED = 42

PRECISION_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="ES Fine-tuning for Countdown Task (baseline / non-accelerated)"
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--hf_cache_dir", type=str, default="hf_cache")
    parser.add_argument(
        "--precision", type=str, default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Precision for model weights",
    )
    parser.add_argument("--gpu_threads", type=int, default=4,
                        help="Number of parallel threads per GPU")
    parser.add_argument("--verbose", action="store_true", help="Print verbose logs")

    # Data args
    parser.add_argument("--data_path", type=str,
                        default=os.path.join(os.path.dirname(__file__), "/n/holylabs/LABS/sham_lab/Users/jbejjani/evolutionary-alignment/countdown/data/countdown.json"),
                        help="Path to JSON data file")
    parser.add_argument("--train_samples", type=int, default=TRAIN_SAMPLES,
                        help="Number of samples for training (rest used for eval)")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS,
                        help="Maximum number of tokens allowed to be generated")

    # ES args
    parser.add_argument("--iterations", type=int, default=NUM_ITERATIONS,
                        help="Number of ES iterations (generations)")
    parser.add_argument("--population_size", type=int, default=POPULATION_SIZE,
                        help="Population size (number of perturbations per iteration)")
    parser.add_argument("--sigma", type=float, default=SIGMA,
                        help="Standard deviation for weight perturbations (noise scale)")
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help="Learning rate")
    parser.add_argument("--global_seed", type=int, default=GLOBAL_SEED,
                        help="Global random seed")
    parser.add_argument("--do_sample", default=False, action="store_true",
                        help="Whether to use sampling (default: greedy decoding)")

    # Checkpointing
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every n iterations")
    parser.add_argument("--experiment_dir", type=str, default="es-countdown-iid",
                        help="Base directory for experiment outputs")

    # Eval args
    parser.add_argument("--eval_interval", type=int, default=25,
                        help="Evaluate on held-out set every N iterations (0 disables)")
    parser.add_argument("--eval_batch_size", type=int, default=512,
                        help="Batch size for evaluation generation")

    # W&B args
    parser.add_argument("--log_wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="es-countdown",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity (team or username)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Prompt processing — applies chat template, returns string
# ---------------------------------------------------------------------------
def _process_context(task_data: Dict[str, Any], tokenizer) -> str:
    """Build a fully-templated prompt string (with special tokens) from task data.

    This mirrors es_accl_static.py's _process_context() but returns a string
    instead of a TokensPrompt, since HF generate works from text inputs.
    """
    numbers = task_data["numbers"]
    target = task_data["target"]
    user_content = USER_TEMPLATE.format(numbers=numbers, target=target)
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]
    prompt_str = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_str += RESPONSE_PROMPT
    return prompt_str


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def force_memory_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def get_save_dir(model_name: str, seed: int, iteration: int, dataset_size: int,
                 args, is_final: bool = False) -> str:
    suffix = "final" if is_final else "checkpoint"
    return os.path.join(
        args.experiment_dir,
        f"{model_name}/{seed}/max_tokens_{args.max_new_tokens}",
        (f"es_random_seed{seed}_pop{args.population_size}_iter{iteration}"
         f"_sigma{args.sigma}_alpha{args.alpha}_{args.precision}"
         f"_threads{args.gpu_threads}_question_num{dataset_size}_{suffix}"),
    )


def save_model_checkpoint(model, tokenizer, iteration, model_name, seed, args,
                          dataset_size):
    save_dir = get_save_dir(model_name, seed, iteration, dataset_size, args,
                            is_final=False)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving checkpoint at iteration {iteration} to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Checkpoint saved successfully.")


# ---------------------------------------------------------------------------
# Generation + reward scoring helpers
# ---------------------------------------------------------------------------
def _generate_and_decode(model, tokenizer, prompt_texts, device, max_new_tokens,
                         do_sample):
    """Tokenize, generate, return list of decoded new-token strings."""
    tokenized = tokenizer(
        prompt_texts, return_tensors="pt", padding=True, padding_side="left",
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    input_length = input_ids.shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

    new_token_ids = outputs[:, input_length:]
    decoded = [
        tokenizer.decode(new_token_ids[i], skip_special_tokens=True)
        for i in range(len(prompt_texts))
    ]

    del input_ids, attention_mask, outputs, new_token_ids
    torch.cuda.empty_cache()

    return decoded


def _score_responses(decoded_texts: List[str],
                     task_datas: List[Dict]) -> Dict[str, Any]:
    """Score decoded model outputs. Prepends RESPONSE_PROMPT before calling
    reward_function, aligned with es_accl_static.py."""
    all_rewards = []
    format_rewards = []
    answer_rewards = []

    for gen_text, data in zip(decoded_texts, task_datas):
        response = RESPONSE_PROMPT + gen_text
        r = reward_function(response, data["numbers"], data["target"])
        all_rewards.append(r["reward"])
        if "reward_info" in r:
            format_rewards.append(r["reward_info"].get("format_reward", 0.0))
            answer_rewards.append(r["reward_info"].get("answer_reward", 0.0))

    avg_reward = float(np.mean(all_rewards)) if all_rewards else 0.0
    avg_format = float(np.mean(format_rewards)) if format_rewards else 0.0
    avg_answer = float(np.mean(answer_rewards)) if answer_rewards else 0.0
    accuracy = (
        (sum(1 for a in answer_rewards if a > 0) / len(answer_rewards) * 100.0)
        if answer_rewards else 0.0
    )
    return {
        "rewards": all_rewards,
        "avg_reward": avg_reward,
        "avg_format": avg_format,
        "avg_answer": avg_answer,
        "accuracy": accuracy,
    }


# ---------------------------------------------------------------------------
# Evaluation on held-out set
# ---------------------------------------------------------------------------
def evaluate_on_eval_split(model, tokenizer, eval_prompts, eval_task_datas,
                           device, step, args, wandb_mod=None):
    """Run greedy generation on the eval set, compute and log metrics."""
    if not eval_task_datas:
        return

    batch_size = max(1, args.eval_batch_size)
    all_rewards = []
    format_rewards = []
    answer_rewards = []
    start = time.time()

    for b in range(0, len(eval_task_datas), batch_size):
        batch_datas = eval_task_datas[b : b + batch_size]
        batch_prompts = eval_prompts[b : b + batch_size]

        decoded = _generate_and_decode(
            model, tokenizer, batch_prompts, device,
            max_new_tokens=args.max_new_tokens, do_sample=False,
        )

        for idx, (gen_text, data) in enumerate(zip(decoded, batch_datas)):
            response = RESPONSE_PROMPT + gen_text
            r = reward_function(response, data["numbers"], data["target"])
            all_rewards.append(r["reward"])
            if "reward_info" in r:
                format_rewards.append(
                    r["reward_info"].get("format_reward", 0.0))
                answer_rewards.append(
                    r["reward_info"].get("answer_reward", 0.0))

            # Print first sample for inspection
            if b == 0 and idx == 0:
                print(f"Eval Sample Response:\n{response}\n---")
                print(f"Target Answer: {data['target']}")
                print(f"The rewards: {r}\n===")

    elapsed = time.time() - start

    avg_reward = float(np.mean(all_rewards)) if all_rewards else 0.0
    std_reward = float(np.std(all_rewards)) if all_rewards else 0.0
    min_reward = float(np.min(all_rewards)) if all_rewards else 0.0
    max_reward = float(np.max(all_rewards)) if all_rewards else 0.0
    avg_format = float(np.mean(format_rewards)) if format_rewards else 0.0
    avg_answer = float(np.mean(answer_rewards)) if answer_rewards else 0.0
    accuracy = (
        (sum(1 for a in answer_rewards if a > 0) / len(answer_rewards) * 100.0)
        if answer_rewards else 0.0
    )

    print(
        f"[Eval @ step {step}] avg_reward={avg_reward:.4f} ± {std_reward:.4f} "
        f"range=[{min_reward:.4f},{max_reward:.4f}] format={avg_format:.4f} "
        f"answer={avg_answer:.4f} acc={accuracy:.1f}% time={elapsed:.2f}s"
    )

    if wandb_mod is not None:
        wandb_mod.log(
            {
                "eval/avg_reward": avg_reward,
                "eval/std_reward": std_reward,
                "eval/min_reward": min_reward,
                "eval/max_reward": max_reward,
                "eval/format_reward": avg_format,
                "eval/answer_reward": avg_answer,
                "eval/accuracy": accuracy,
                "eval/time": elapsed,
            },
            step=step,
        )


# ---------------------------------------------------------------------------
# Per-seed perturbation + evaluation worker
# ---------------------------------------------------------------------------
def process_seed(seed_args):
    """Process a single seed: perturb weights, evaluate, restore weights."""
    (seed_idx, seed, model, tokenizer, accelerator, thread_id,
     verbose, prompt_texts, task_datas, args) = seed_args

    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} "
              f"processing seed {seed_idx} (value: {seed})")

    # --- Perturb weights ---
    seed_shift = 0
    for name, param in model.named_parameters():
        gen = torch.Generator(device=param.device)
        gen.manual_seed(int(seed + seed_shift))
        seed_shift += 1
        noise = torch.randn(
            param.shape, generator=gen, device=param.device, dtype=param.dtype,
        )
        param.data.add_(args.sigma * noise)

    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)

    # --- Evaluate ---
    decoded = _generate_and_decode(
        model, tokenizer, prompt_texts, accelerator.device,
        max_new_tokens=args.max_new_tokens, do_sample=args.do_sample,
    )
    metrics = _score_responses(decoded, task_datas)

    # --- Restore weights ---
    seed_shift = 0
    for name, param in model.named_parameters():
        gen = torch.Generator(device=param.device)
        gen.manual_seed(int(seed + seed_shift))
        seed_shift += 1
        noise = torch.randn(
            param.shape, generator=gen, device=param.device, dtype=param.dtype,
        )
        param.data.add_(-args.sigma * noise)

    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)

    force_memory_cleanup()

    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} "
              f"completed seed {seed_idx} with reward "
              f"{metrics['avg_reward']:.4f}")

    return seed_idx, metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    accelerator = Accelerator()

    # --- Load dataset ---
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset file not found: {args.data_path}")

    with open(args.data_path, "r") as f:
        all_task_datas = json.load(f)

    # --- Tokenizer (needed for prompt construction) ---
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=False, cache_dir=args.hf_cache_dir,
    )

    # --- Build train / eval splits ---
    train_task_datas = all_task_datas[: args.train_samples]
    train_prompts = [_process_context(d, tokenizer) for d in train_task_datas]

    eval_task_datas = []
    eval_prompts = []
    if args.eval_interval > 0:
        eval_task_datas = all_task_datas[args.train_samples :]
        eval_prompts = [_process_context(d, tokenizer) for d in eval_task_datas]

    if accelerator.is_main_process:
        print(f"Loaded {len(train_task_datas)} train samples from {args.data_path}")
        print(f"Loaded {len(eval_task_datas)} eval samples from {args.data_path}")
        print(f"Total processes: {accelerator.num_processes}, "
              f"GPU threads per process: {args.gpu_threads}")
        print(f"Population size: {args.population_size}, "
              f"Iterations: {args.iterations}")
        print(f"Sigma: {args.sigma}, Alpha: {args.alpha}, "
              f"Precision: {args.precision}")

    # --- Logging setup ---
    wandb_mod = None
    logging_dir = None
    if accelerator.is_main_process:
        logging_dir = os.path.join(
            args.experiment_dir,
            f"countdown_iid_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        os.makedirs(logging_dir, exist_ok=True)

        # Dump args
        with open(os.path.join(logging_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

        if args.log_wandb:
            import wandb

            short_model_name = (
                args.model_name.split("/")[-1]
                if "/" in args.model_name
                else args.model_name
            )
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"{short_model_name}-iid",
                config=vars(args),
                dir=logging_dir,
            )
            wandb_mod = wandb

    # --- Load model replicas (one per GPU thread) ---
    torch_dtype = PRECISION_MAP[args.precision]
    model_list = []
    for _ in range(args.gpu_threads):
        model_list.append(
            AutoModelForCausalLM.from_pretrained(
                args.model_name,
                cache_dir=args.hf_cache_dir,
                device_map={"": accelerator.process_index},
                torch_dtype=torch_dtype,
                attn_implementation="sdpa",
            )
        )

    if accelerator.is_main_process:
        print("Model loaded successfully")

    for model in model_list:
        model.eval()

    force_memory_cleanup()

    # --- Training loop ---
    training_start_time = time.time()

    for iteration in range(args.iterations):
        iter_start_time = time.time()
        force_memory_cleanup()

        if args.verbose:
            print(f"Process {accelerator.process_index} starting "
                  f"iteration {iteration}/{args.iterations}")

        # --- Generate seeds (deterministic per-iteration, aligned with accl) ---
        loop_rng = np.random.default_rng(seed=args.global_seed + iteration)
        seeds = loop_rng.integers(
            0, 2**30, size=args.population_size, dtype=np.int64,
        ).tolist()

        # Broadcast seeds to all processes (for multi-GPU)
        if accelerator.num_processes > 1:
            seeds_tensor = torch.tensor(
                seeds, device=accelerator.device, dtype=torch.long,
            )
            torch.distributed.broadcast(seeds_tensor, src=0)
            seeds = seeds_tensor.cpu().tolist()
            del seeds_tensor

        # --- Assign seeds to this process ---
        local_seeds = [
            (seed_idx, seed)
            for seed_idx, seed in enumerate(seeds)
            if seed_idx % accelerator.num_processes == accelerator.process_index
        ]

        if args.verbose:
            print(f"Process {accelerator.process_index} assigned "
                  f"{len(local_seeds)} seeds")

        # --- Process seeds in threaded batches ---
        local_results: Dict[int, Dict[str, Any]] = {}
        batch_size = max(1, min(args.gpu_threads, len(local_seeds)))

        for batch_start in range(0, len(local_seeds), batch_size):
            batch_seeds = local_seeds[batch_start : batch_start + batch_size]

            with ThreadPoolExecutor(max_workers=len(batch_seeds)) as executor:
                thread_args = [
                    (seed_idx, seed, model_list[thread_id], tokenizer,
                     accelerator, thread_id, args.verbose,
                     train_prompts, train_task_datas, args)
                    for thread_id, (seed_idx, seed) in enumerate(batch_seeds)
                ]
                results = list(executor.map(process_seed, thread_args))
                for seed_idx, metrics in results:
                    local_results[seed_idx] = metrics

            force_memory_cleanup()

        # --- Aggregate rewards across all processes ---
        all_rewards_t = torch.zeros(
            args.population_size, device=accelerator.device)
        all_format_t = torch.zeros(
            args.population_size, device=accelerator.device)
        all_answer_t = torch.zeros(
            args.population_size, device=accelerator.device)
        all_accuracy_t = torch.zeros(
            args.population_size, device=accelerator.device)

        for seed_idx, metrics in local_results.items():
            all_rewards_t[seed_idx] = metrics["avg_reward"]
            all_format_t[seed_idx] = metrics["avg_format"]
            all_answer_t[seed_idx] = metrics["avg_answer"]
            all_accuracy_t[seed_idx] = metrics["accuracy"]

        if accelerator.num_processes > 1:
            for t in (all_rewards_t, all_format_t, all_answer_t, all_accuracy_t):
                torch.distributed.all_reduce(
                    t, op=torch.distributed.ReduceOp.SUM)

        rewards = all_rewards_t.cpu().numpy()
        formats_arr = all_format_t.cpu().numpy()
        answers_arr = all_answer_t.cpu().numpy()
        accuracy_arr = all_accuracy_t.cpu().numpy()

        del all_rewards_t, all_format_t, all_answer_t, all_accuracy_t

        # --- Normalize rewards ---
        rewards_normalized = (
            (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        )

        # --- Log metrics ---
        mean_reward = float(rewards.mean())
        std_reward = float(rewards.std())
        min_reward = float(rewards.min())
        max_reward = float(rewards.max())
        mean_format = float(formats_arr.mean())
        mean_answer = float(answers_arr.mean())
        mean_accuracy = float(accuracy_arr.mean())

        if accelerator.is_main_process:
            print(f"\n=== Iteration {iteration}/{args.iterations} ===")
            print(
                f"Mean reward: {mean_reward:.4f}, std: {std_reward:.4f}, "
                f"format: {mean_format:.4f}, answer: {mean_answer:.4f}, "
                f"acc: {mean_accuracy:.1f}%"
            )

            if wandb_mod is not None:
                wandb_mod.log(
                    {
                        "reward/mean": mean_reward,
                        "reward/std": std_reward,
                        "reward/min": min_reward,
                        "reward/max": max_reward,
                        "train/format_reward": mean_format,
                        "train/answer_reward": mean_answer,
                        "train/accuracy": mean_accuracy,
                    },
                    step=iteration,
                )

        # --- Update weights on the primary model (model_list[0]) ---
        original_model = model_list[0]
        seed_shift = 0
        for name, param in original_model.named_parameters():
            gen = torch.Generator(device=param.device)
            update = torch.zeros_like(param)
            for seed_idx in range(args.population_size):
                r_norm = rewards_normalized[seed_idx]
                seed = seeds[seed_idx]
                gen.manual_seed(int(seed + seed_shift))
                noise = torch.randn(
                    param.shape, generator=gen,
                    device=param.device, dtype=param.dtype,
                )
                noise.mul_(float(r_norm))
                update.add_(noise)
                del noise
            update.div_(args.population_size)
            param.data.add_(args.alpha * update)
            torch.cuda.empty_cache()
            seed_shift += 1

        # --- Sync weights to other model replicas ---
        for model_idx in range(1, len(model_list)):
            for name, param in model_list[model_idx].named_parameters():
                param.data.copy_(original_model.get_parameter(name).data)

        if torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)

        force_memory_cleanup()

        iter_time = time.time() - iter_start_time

        if accelerator.is_main_process:
            print(f"Iteration {iteration} finished in {iter_time:.2f}s")
            print(
                f"GPU Memory: "
                f"{torch.cuda.memory_allocated() / 1024**2:.2f}MB allocated, "
                f"{torch.cuda.max_memory_allocated() / 1024**2:.2f}MB peak"
            )

            # --- Checkpoint ---
            if (iteration + 1) % args.save_steps == 0:
                save_model_checkpoint(
                    original_model, tokenizer, iteration + 1,
                    args.model_name, args.global_seed, args,
                    len(train_task_datas),
                )

            # --- Eval ---
            if args.eval_interval > 0 and (
                iteration % args.eval_interval == 0
                or iteration == args.iterations - 1
            ):
                evaluate_on_eval_split(
                    original_model, tokenizer,
                    eval_prompts, eval_task_datas,
                    device=accelerator.device,
                    step=iteration, args=args,
                    wandb_mod=wandb_mod,
                )

        accelerator.wait_for_everyone()

        del rewards, rewards_normalized, formats_arr, answers_arr, accuracy_arr
        force_memory_cleanup()

    # --- Final save ---
    total_time = time.time() - training_start_time

    if accelerator.is_main_process:
        print(f"\nTraining completed in {total_time:.2f}s "
              f"({total_time / 60:.2f} minutes)")
        save_dir = get_save_dir(
            args.model_name, args.global_seed, args.iterations,
            len(train_task_datas), args, is_final=True,
        )
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving final model to {save_dir}...")
        original_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Final model saved successfully.")

        if wandb_mod is not None:
            wandb_mod.finish()


if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore"
    mp.set_start_method("spawn", force=True)
    main()
