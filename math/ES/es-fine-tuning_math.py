#!/usr/bin/env python
"""
ES Fine-tuning for Math RLVR (Training Only).

Uses Evolution Strategies with multi-engine NCCL sync to train on math problems.
Saves checkpoints periodically for later evaluation with math_eval.py.

Features:
- Multi-GPU training via Ray + vLLM
- NCCL-based weight synchronization
- Periodic HF-format checkpoint saving
- Compact replay logs for checkpoint reconstruction
- Basic train accuracy logging to wandb/tensorboard

Usage:
    python es-fine-tuning_math.py \
        --policy_model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --num_engines 4 \
        --num_iterations 1000 \
        --save_every 100 \
        --save_replay_log

Replay Log:
    Enables checkpoint reconstruction from compact logs (~39,000x smaller than full weights).
    Use reconstruct_checkpoint.py to rebuild any iteration's weights.

Evaluation should be done separately using math_eval.py on saved checkpoints.
"""

import argparse
from datetime import datetime
import gc
import json
import os
import random
import shutil
import signal
import sys
import time

import numpy as np
import ray
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.utils import get_ip, get_open_port
import importlib.util

from datasets import load_dataset

# Default Hyperparameters
SIGMA = 0.001
ALPHA = 0.0005
POPULATION_SIZE = 30
NUM_ENGINES = 4
NUM_ITERATIONS = 1000
EXPERIMENT_DIR = "/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/math/es-ft-experiment"


class ESReplayLog:
    """
    Manages saving of compact replay logs for ES training.
    
    Instead of saving full model weights at every checkpoint, we save:
    - One-time metadata (base model path, hyperparameters)
    - Per-iteration: seeds and update coefficients (w_n = alpha/N * Z_n)
    
    This allows reconstructing any checkpoint by replaying the updates
    from the base model, with ~39,000x storage reduction.
    
    Use reconstruct_checkpoint.py to rebuild weights from replay logs.
    """
    
    def __init__(self, log_dir: str, base_model_path: str, args):
        self.log_dir = log_dir
        self.replay_dir = os.path.join(log_dir, "replay_logs")
        os.makedirs(self.replay_dir, exist_ok=True)
        
        # Save one-time metadata
        self.metadata = {
            "base_model_path": base_model_path,
            "sigma": args.sigma,
            "alpha": args.alpha,
            "population_size": args.population_size,
            "num_iterations": args.num_iterations,
            "global_seed": args.global_seed,
            "train_dataset": args.train_dataset,
            "created_at": datetime.now().strftime('%Y%m%d_%H%M%S'),
        }
        self._save_metadata()
        
        # Track iterations logged
        self.iterations_logged = 0
    
    def _save_metadata(self):
        """Save one-time run metadata."""
        meta_path = os.path.join(self.replay_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
    
    def log_iteration(self, iteration: int, seeds: list, update_coeffs: list):
        """
        Log a single iteration's update information.
        
        Args:
            iteration: Current iteration number
            seeds: List of seeds used for perturbations [s_1, ..., s_N]
            update_coeffs: List of update coefficients [w_1, ..., w_N]
                          where w_n = (alpha / N) * Z_n
        """
        entry = {
            "iteration": iteration,
            "seeds": seeds,
            "update_coeffs": update_coeffs,
        }
        
        # Append to JSONL file (one line per iteration for streaming)
        log_path = os.path.join(self.replay_dir, "iteration_logs.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        
        self.iterations_logged += 1
    
    def save_full_checkpoint_marker(self, iteration: int, checkpoint_path: str):
        """
        Record that a full checkpoint was saved at this iteration.
        This helps with efficient reconstruction (start from nearest full checkpoint).
        """
        markers_path = os.path.join(self.replay_dir, "full_checkpoints.json")
        markers = []
        if os.path.exists(markers_path):
            with open(markers_path, "r") as f:
                markers = json.load(f)
        
        markers.append({
            "iteration": iteration,
            "checkpoint_path": checkpoint_path,
            "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S'),
        })
        
        with open(markers_path, "w") as f:
            json.dump(markers, f, indent=2)
    
    def get_storage_stats(self):
        """Return storage statistics for the replay log."""
        log_path = os.path.join(self.replay_dir, "iteration_logs.jsonl")
        meta_path = os.path.join(self.replay_dir, "metadata.json")
        
        total_bytes = 0
        if os.path.exists(log_path):
            total_bytes += os.path.getsize(log_path)
        if os.path.exists(meta_path):
            total_bytes += os.path.getsize(meta_path)
        
        return {
            "num_iterations": self.iterations_logged,
            "log_size_bytes": total_bytes,
            "log_size_kb": total_bytes / 1024,
            "log_size_mb": total_bytes / (1024 * 1024),
            "replay_dir": self.replay_dir,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="ES Fine-tuning for Math RLVR (Training Only)"
    )
    parser.add_argument("--policy_model_path", type=str, 
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                        help="Base model to fine-tune")
    parser.add_argument("--sigma", type=float, default=SIGMA,
                        help="Perturbation noise scale")
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help="Learning rate")
    parser.add_argument("--population_size", type=int, default=POPULATION_SIZE,
                        help="Number of perturbations per iteration")
    parser.add_argument("--num_engines", type=int, default=NUM_ENGINES,
                        help="Number of vLLM engines (GPUs)")
    parser.add_argument("--num_iterations", type=int, default=NUM_ITERATIONS,
                        help="Total ES iterations")
    parser.add_argument("--experiment_dir", type=str, default=EXPERIMENT_DIR,
                        help="Directory to save checkpoints and logs")
    parser.add_argument("--cuda_devices", type=str, default="0,1,2,3",
                        help="Comma-separated list of CUDA device IDs")
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose logs')
    parser.add_argument('--max_new_tokens', type=int, default=1024,
                        help='Maximum tokens to generate')
    parser.add_argument('--prompt_max_len', type=int, default=1024,
                        help='Maximum tokens in prompt')
    
    # Dataset args
    parser.add_argument('--train_samples', type=int, default=None,
                        help='Limit train samples (None = full split)')
    parser.add_argument('--train_dataset', type=str, 
                        default='DigitalLearningGmbH/MATH-lighteval',
                        help='Training dataset name')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='Prompts per iteration for ES fitness')
    parser.add_argument('--standardize_within_batch', action='store_true', default=False,
                        help='Standardize reward within mini-batch')
    
    # Logging and checkpointing
    parser.add_argument('--wandb_project', type=str, default='math_es_training',
                        help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default='',
                        help='W&B run name')
    parser.add_argument('--save_every', type=int, default=100,
                        help='Save HF checkpoint every N iters (0=off)')
    parser.add_argument('--save_replay_log', action='store_true',
                        help='Save compact replay logs for checkpoint reconstruction')
    parser.add_argument("--global_seed", type=int, default=42,
                        help="Global random seed")
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # Set global random seed
    if args.global_seed is not None:
        random.seed(args.global_seed)
        np.random.seed(args.global_seed)
        torch.manual_seed(args.global_seed)
        torch.cuda.manual_seed_all(args.global_seed)

    return args


class ESNcclLLM(LLM):
    def __init__(self, *args, **kwargs):
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)


def launch_engines(num_engines, model_name):
    """Launch vLLM engines with NCCL worker extension."""
    engines = [
        ray.remote(num_cpus=8, num_gpus=1)(ESNcclLLM).remote(
            model=model_name,
            tensor_parallel_size=1,
            distributed_executor_backend="mp",
            worker_extension_cls="utils.worker_extn.WorkerExtension",
            dtype="float16",
            enable_prefix_caching=False,
            enforce_eager=False,
        )
        for _ in range(num_engines)
    ]
    return engines


def load_math_verifier_module():
    """Dynamically load math_verifier.py to avoid clashing with Python stdlib 'math'."""
    # math_verifier.py is one level up from ES/
    verifier_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "math_verifier.py")
    spec = importlib.util.spec_from_file_location("math_verifier_local", verifier_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _extract_reward_list(obj):
    """Normalize outputs from math_verifier.reward_func into a list of floats."""
    if isinstance(obj, dict):
        for key in ("rewards", "scores", "values"):
            if key in obj:
                obj = obj[key]
                break
        else:
            if obj:
                obj = next(iter(obj.values()))
            else:
                return []
    if hasattr(obj, "detach"):
        obj = obj.detach()
    if hasattr(obj, "cpu"):
        obj = obj.cpu()
    if hasattr(obj, "tolist"):
        try:
            return [float(x) for x in obj.tolist()]
        except Exception:
            pass
    if isinstance(obj, (list, tuple)):
        try:
            return [float(x) for x in obj]
        except Exception:
            pass
    raise TypeError(f"Unsupported reward output type: {type(obj)}")


def compute_rewards_math(responses, prompts, answers, batch_size: int = 256):
    """Compute rewards via math_verifier: 1.0 if correct, else 0.0."""
    mv = load_math_verifier_module()
    rewards = []
    for i in range(0, len(responses), batch_size):
        reward_out = mv.reward_func(
            queries=responses[i:i+batch_size],
            prompts=prompts[i:i+batch_size],
            answers=answers[i:i+batch_size],
        )
        rewards.extend(_extract_reward_list(reward_out))
    return {
        "per_sample_reward": rewards,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "num_samples": len(rewards),
    }


def evaluate_alignment_handle(llm, prompts, max_tokens):
    """Start async generation with greedy decoding."""
    sampling_params = SamplingParams(
        temperature=0.0,
        seed=42,
        max_tokens=max_tokens,
    )
    handle = llm.generate.remote(prompts, sampling_params, use_tqdm=False)
    return handle, time.time()


def build_responses(outputs):
    """Extract response texts from vLLM outputs."""
    return [out.outputs[0].text for out in outputs]


def main(args):
    # Ensure local Ray
    os.environ.pop("RAY_ADDRESS", None)
    os.environ.pop("RAY_HEAD_IP", None)
    os.environ.pop("RAY_GCS_SERVER_ADDRESS", None)
    ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)

    # Verify GPU availability
    resources = ray.cluster_resources()
    gpu_count = resources.get('GPU', 0)
    required_gpus = args.num_engines

    print("=" * 80)
    print(f"Ray Cluster Resources:")
    print(f"  GPUs: {gpu_count}")
    print(f"  CPUs: {resources.get('CPU', 0)}")
    print(f"  Memory: {resources.get('memory', 0) / (1024**3):.2f} GB")
    print(f"  Required: {required_gpus} GPUs for {args.num_engines} engines")
    if gpu_count < required_gpus:
        print(f"\n❌ ERROR: Need {required_gpus} GPUs but only {gpu_count} available!")
        ray.shutdown()
        sys.exit(1)
    print(f"✅ Found {gpu_count} GPUs - sufficient for training")
    print("=" * 80)

    # Logging setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging_dir = f"{args.experiment_dir}/es_math_{timestamp}"
    writer = SummaryWriter(log_dir=logging_dir)
    
    if args.wandb_project:
        default_name = args.wandb_run_name or f"es_math_{os.path.basename(args.policy_model_path).replace('/', '_')}_{timestamp}"
        wandb.init(project=args.wandb_project, name=default_name, config={
            'policy_model_path': args.policy_model_path,
            'sigma': args.sigma,
            'alpha': args.alpha,
            'population_size': args.population_size,
            'num_engines': args.num_engines,
            'batch_size': args.batch_size,
            'train_dataset': args.train_dataset,
            'standardize_within_batch': args.standardize_within_batch,
            'global_seed': args.global_seed,
        })
        wandb.define_metric("iteration")
        wandb.define_metric("*", step_metric="iteration")

    # Model saves directory
    model_saves_dir = f"{logging_dir}/model_saves"
    os.makedirs(model_saves_dir, exist_ok=True)

    # Resolve base model path
    if os.path.isdir(args.policy_model_path):
        base_model_path = args.policy_model_path
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.policy_model_path, torch_dtype=torch.float16
        ).to("cpu")
        tokenizer = AutoTokenizer.from_pretrained(args.policy_model_path)
        base_model_path = f"{model_saves_dir}/base_model"
        if os.path.exists(base_model_path):
            shutil.rmtree(base_model_path)
        os.makedirs(base_model_path, exist_ok=True)
        tokenizer.save_pretrained(base_model_path)
        base_model.save_pretrained(base_model_path)
        del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    policy_tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # Initialize replay log if enabled
    replay_log = None
    if args.save_replay_log:
        replay_log = ESReplayLog(logging_dir, base_model_path, args)
        print(f"📝 Replay log enabled - saving seeds + coefficients to {replay_log.replay_dir}")

    def truncate_prompt(prompt_text: str) -> str:
        try:
            ids = policy_tokenizer(prompt_text, add_special_tokens=False).get("input_ids", [])
            if len(ids) <= args.prompt_max_len:
                return prompt_text
            ids = ids[-args.prompt_max_len:]
            return policy_tokenizer.decode(ids, skip_special_tokens=True)
        except Exception:
            return prompt_text

    def save_checkpoint(engine, iteration: int, checkpoint_name: str = None):
        """Save current engine weights to HF-format checkpoint."""
        tmp_path = f"{model_saves_dir}/tmp_iter_{iteration}.pth"
        ray.get(engine.collective_rpc.remote("save_self_weights_to_disk", args=(tmp_path,)))
        
        mdl = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16).to("cpu")
        state = torch.load(tmp_path, map_location="cpu")
        mdl.load_state_dict(state, strict=True)
        
        ckpt_name = checkpoint_name or f"checkpoint_iter_{iteration}"
        ckpt_dir = f"{model_saves_dir}/{ckpt_name}"
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        os.makedirs(ckpt_dir, exist_ok=True)
        
        policy_tokenizer.save_pretrained(ckpt_dir)
        mdl.save_pretrained(ckpt_dir)
        with open(os.path.join(ckpt_dir, "ckpt_meta.json"), "w") as f:
            json.dump({
                "iteration": iteration,
                "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S'),
                "sigma": args.sigma,
                "alpha": args.alpha,
                "population_size": args.population_size,
            }, f)
        
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        del mdl, state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"✅ Checkpoint saved: {ckpt_dir}")
        return ckpt_dir

    def extract_qa(record: dict):
        """Extract question and answer from dataset record."""
        qkeys = ['problem']
        akeys = ['solution']
        question = None
        answer = None
        for k in qkeys:
            if k in record and record[k] is not None:
                question = record[k]
                break
        for k in akeys:
            if k in record and record[k] is not None:
                answer = record[k]
                break
        if question is None and isinstance(record.get('sample'), dict):
            s = record['sample']
            for k in qkeys:
                if k in s and s[k] is not None:
                    question = s[k]
                    break
            for k in akeys:
                if k in s and s[k] is not None:
                    answer = s[k]
                    break
        return question, answer

    def load_train_data(dataset_name: str, n: int | None):
        """Load training data from HuggingFace dataset."""
        ds = load_dataset(dataset_name, split='train')
        total = len(ds)
        if n is None or n >= total:
            indices = list(range(total))
        else:
            rng = random.Random(args.global_seed)
            indices = rng.sample(range(total), n)
        pairs = []
        for idx in indices:
            raw = ds[int(idx)]
            q, a = extract_qa(raw)
            if q is None or a is None:
                continue
            conv = truncate_prompt(q)
            pairs.append((conv, a))
        return pairs

    # Load training data
    train_pairs = load_train_data(args.train_dataset, args.train_samples)
    train_total = len(train_pairs)
    print(f"\n📊 Loaded {train_total} training samples from {args.train_dataset}")

    if args.wandb_project and wandb.run is not None:
        wandb.config.update({
            "train_sample_count": train_total,
            "train_samples_arg": args.train_samples,
        }, allow_val_change=True)

    # Launch vLLM engines
    print(f"\nLaunching {args.num_engines} vLLM engines...")
    engines = launch_engines(args.num_engines, base_model_path)
    print(f"✅ Submitted {len(engines)} engines")

    # Init inter-engine NCCL communicator
    master_address = get_ip()
    master_port = get_open_port()
    ray.get([
        engines[i].collective_rpc.remote(
            "init_inter_engine_group", args=(master_address, master_port, i, args.num_engines)
        )
        for i in range(args.num_engines)
    ])

    def cleanup():
        for llm in engines:
            try:
                ray.kill(llm)
            except Exception:
                pass
        ray.shutdown()

    def sig_handler(sig, frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # Main training loop
    for i in range(args.num_iterations):
        print(f"\n=== Iteration {i} ===")
        total_iter_start = time.time()

        # Random seeds for population
        seeds = [random.randint(0, 1_000_000) for _ in range(args.population_size)]
        seeds_perf = {}

        # Round-robin scheduling
        seed_iter = iter(seeds)
        inflight = {}
        all_rewards_samples = []

        # Sample prompts for this iteration
        bs = min(args.batch_size, len(train_pairs))
        rng = random.Random(args.global_seed * 100000 + i)
        subset_indices = rng.sample(range(len(train_pairs)), bs) if bs < len(train_pairs) else list(range(len(train_pairs)))
        curr_prompts = [train_pairs[idx][0] for idx in subset_indices]
        curr_answers = [train_pairs[idx][1] for idx in subset_indices]

        # Kick off evaluation on each engine
        for eng_idx, llm in enumerate(engines):
            try:
                seed = next(seed_iter)
            except StopIteration:
                break
            ray.get(llm.collective_rpc.remote("perturb_self_weights", args=(seed, args.sigma, False)))
            handle, start_ts = evaluate_alignment_handle(llm, curr_prompts, args.max_new_tokens)
            inflight[handle] = {
                "engine": llm,
                "engine_idx": eng_idx,
                "seed": seed,
                "start_ts": start_ts,
            }

        # Process results as they complete
        while inflight:
            done, _ = ray.wait(list(inflight.keys()), num_returns=1)
            h = done[0]
            meta = inflight.pop(h)

            outputs = ray.get(h)
            responses = build_responses(outputs)
            
            metrics = compute_rewards_math(responses, curr_prompts, curr_answers, batch_size=1024)
            elapsed = time.time() - meta["start_ts"]

            rewards_np = np.asarray(metrics["per_sample_reward"], dtype=np.float32)
            all_rewards_samples.extend(rewards_np.tolist())
            r_eff = rewards_np
            r_mean = float(r_eff.mean())

            if args.standardize_within_batch:
                r_std = float(r_eff.std()) + 1e-8
                r_mean /= r_std

            seeds_perf[meta["seed"]] = {
                **metrics,
                "mean_reward": r_mean,
            }

            llm = meta["engine"]
            ray.get(llm.collective_rpc.remote("restore_self_weights", args=(meta["seed"], args.sigma)))

            # Schedule next seed
            try:
                next_seed = next(seed_iter)
            except StopIteration:
                continue

            ray.get(llm.collective_rpc.remote("perturb_self_weights", args=(next_seed, args.sigma, False)))
            handle, start_ts = evaluate_alignment_handle(llm, curr_prompts, args.max_new_tokens)
            inflight[handle] = {
                "engine": llm,
                "engine_idx": meta["engine_idx"],
                "seed": next_seed,
                "start_ts": start_ts,
            }

        # Compute statistics
        reward_means = [v["mean_reward"] for v in seeds_perf.values()]
        reward_mean = float(np.mean(reward_means)) if reward_means else 0.0
        reward_std = float(np.std(reward_means)) if reward_means else 0.0
        reward_min = float(np.min(reward_means)) if reward_means else 0.0
        reward_max = float(np.max(reward_means)) if reward_means else 0.0

        print(f"Reward: mean={reward_mean:.4f}, std={reward_std:.4f}, min={reward_min:.4f}, max={reward_max:.4f}")

        # Normalize rewards for ES update
        for k in seeds_perf:
            seeds_perf[k]["norm_reward"] = (seeds_perf[k]["mean_reward"] - reward_mean) / (reward_std + 1e-8)

        # Log to tensorboard and wandb
        writer.add_scalar("train/reward_mean", reward_mean, i)
        writer.add_scalar("train/reward_std", reward_std, i)
        writer.add_scalar("train/accuracy", reward_mean, i)
        
        if args.wandb_project:
            wandb.log({
                'iteration': i,
                'train/reward_mean': reward_mean,
                'train/reward_std': reward_std,
                'train/reward_min': reward_min,
                'train/reward_max': reward_max,
                'train/accuracy': reward_mean,
            }, step=i)

        # ES weight update on engine 0
        per_seed_coeffs = [
            (seed, (args.alpha / args.population_size) * float(seeds_perf[seed]["norm_reward"]))
            for seed in seeds
        ]

        handles = []
        for seed, coeff in per_seed_coeffs:
            handles.append(engines[0].collective_rpc.remote("perturb_self_weights", args=(seed, coeff, False)))
        ray.get(handles)

        # Log to replay log (before broadcast, captures the update)
        if replay_log is not None:
            replay_log.log_iteration(
                iteration=i,
                seeds=seeds,
                update_coeffs=[coeff for _, coeff in per_seed_coeffs],
            )

        # Broadcast to all engines
        ray.get([e.collective_rpc.remote("broadcast_all_weights", args=(0,)) for e in engines])

        total_iter_time = time.time() - total_iter_start
        writer.add_scalar("time/iteration", total_iter_time, i)
        print(f"Iteration {i} completed in {total_iter_time:.1f}s")

        # Save checkpoint periodically
        if args.save_every > 0 and ((i + 1) % args.save_every == 0):
            try:
                ckpt_path = save_checkpoint(engines[0], i + 1)
                # Mark in replay log for faster reconstruction
                if replay_log is not None:
                    replay_log.save_full_checkpoint_marker(i + 1, ckpt_path)
                if args.wandb_project:
                    wandb.log({'checkpoint/iteration': i + 1}, step=i)
            except Exception as e:
                print(f"Warning: failed to save checkpoint at iter {i+1}: {e}")

    # Save final checkpoint
    print("\n" + "=" * 80)
    print("Training complete. Saving final checkpoint...")
    print("=" * 80)
    
    final_ckpt = save_checkpoint(engines[0], args.num_iterations, "final")
    
    # Mark final checkpoint in replay log
    if replay_log is not None:
        replay_log.save_full_checkpoint_marker(args.num_iterations, final_ckpt)
    
    # Save training config
    config_path = f"{logging_dir}/training_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "policy_model_path": args.policy_model_path,
            "num_iterations": args.num_iterations,
            "sigma": args.sigma,
            "alpha": args.alpha,
            "population_size": args.population_size,
            "batch_size": args.batch_size,
            "train_dataset": args.train_dataset,
            "train_sample_count": train_total,
            "global_seed": args.global_seed,
            "final_checkpoint": final_ckpt,
            "replay_log_enabled": args.save_replay_log,
        }, f, indent=2)
    
    print(f"\n✅ Training finished!")
    print(f"   Checkpoints: {model_saves_dir}/")
    print(f"   Final model: {final_ckpt}")
    
    # Print replay log storage stats if enabled
    if replay_log is not None:
        stats = replay_log.get_storage_stats()
        print(f"\n📊 Replay Log Storage Stats:")
        print(f"   Iterations logged: {stats['num_iterations']}")
        print(f"   Log size: {stats['log_size_kb']:.2f} KB ({stats['log_size_mb']:.4f} MB)")
        print(f"   Replay dir: {stats['replay_dir']}")
        # Estimate savings (assuming ~3GB model weights)
        model_size_mb = 3000  # Approximate for 1.5B model
        estimated_savings = model_size_mb / max(stats['log_size_mb'], 0.001)
        print(f"   Estimated storage savings: ~{estimated_savings:.0f}x vs full checkpoint")
        print(f"\n   Reconstruct any checkpoint with:")
        print(f"   python reconstruct_checkpoint.py --replay_log_dir {stats['replay_dir']} --target_iteration <N>")
    
    print(f"\n   Run evaluation with:")
    print(f"   python ../math_eval.py --model_path {final_ckpt}")

    cleanup()


if __name__ == "__main__":
    args = parse_args()
    main(args)
