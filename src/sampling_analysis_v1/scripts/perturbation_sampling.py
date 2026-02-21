"""Weight perturbation sampling for Countdown task.

For each noise scale sigma, apply N random weight perturbations to the base
model, generate one greedy response per perturbation per prompt, and compute
pass@k curves. Tests whether ES-style weight-space exploration alone (no
iterative updates) can find correct solutions.

Usage:
    python src/sampling_analysis_v1/scripts/perturbation_sampling.py \
        configs/sampling_analysis_v1/perturbation_sampling.yaml [--overwrite] [--debug]
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import ray
import yaml
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, SamplingParams

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from countdown.countdown_task import reward_function
from src.sampling_analysis_v1.utils import (
    RESPONSE_PROMPT,
    build_chat_prompts,
    compute_pass_at_k_dataset,
    load_countdown_data,
)

K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


class _RayLLM(LLM):
    """Thin LLM subclass that clears env vars before init (required for Ray)."""

    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)


def _launch_engine(model_name, dtype="float16"):
    """Launch a single Ray-managed vLLM engine with our WorkerExtension."""
    pg = placement_group([{"GPU": 1, "CPU": 0}], lifetime="detached")
    ray.get(pg.ready())

    strategy = PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_capture_child_tasks=True,
        placement_group_bundle_index=0,
    )
    engine = ray.remote(
        num_cpus=0, num_gpus=0, scheduling_strategy=strategy
    )(_RayLLM).remote(
        model=model_name,
        tensor_parallel_size=1,
        distributed_executor_backend="ray",
        worker_extension_cls="src.sampling_analysis_v1.worker_extension.SamplingWorkerExtension",
        dtype=dtype,
        enable_prefix_caching=False,
        enforce_eager=False,
        gpu_memory_utilization=0.65,
    )
    return engine, pg


def main(config_path, overwrite=False, debug=False):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if "output_dir" not in config:
        raise ValueError("FATAL: 'output_dir' required in config")

    output_dir = Path(config["output_dir"])
    if output_dir.exists() and not overwrite:
        raise ValueError(f"Output dir {output_dir} exists. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")

    # Load data
    data_cfg = config["data"]
    task_datas = load_countdown_data(
        data_cfg["path"], data_cfg["offset"], data_cfg["num_samples"]
    )
    print(f"Loaded {len(task_datas)} samples (offset={data_cfg['offset']})")

    if debug:
        task_datas = task_datas[:32]
        config["sigmas"] = config["sigmas"][:1]
        config["generation"]["num_perturbations"] = 32
        print(f"DEBUG: 32 problems, 1 sigma ({config['sigmas'][0]}), 32 perturbations")

    # Set PYTHONPATH so Ray workers can import our extension
    existing_pypath = os.environ.get("PYTHONPATH", "")
    if _REPO_ROOT not in existing_pypath.split(os.pathsep):
        os.environ["PYTHONPATH"] = (
            _REPO_ROOT + (os.pathsep + existing_pypath if existing_pypath else "")
        )

    # Clean stale Ray env vars
    for var in ("RAY_ADDRESS", "RAY_HEAD_IP", "RAY_GCS_SERVER_ADDRESS"):
        os.environ.pop(var, None)

    ray.init(
        address="local",
        include_dashboard=False,
        ignore_reinit_error=True,
        _temp_dir=tempfile.mkdtemp(prefix=f"ray_sampling_{int(time.time())}_"),
        dashboard_port=None,
    )

    # Build prompts
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    prompts = build_chat_prompts(task_datas, tokenizer)

    # Launch engine
    print(f"Launching vLLM engine: {config['model_id']} ({config['dtype']})")
    engine, pg = _launch_engine(config["model_id"], config["dtype"])

    # Store base weights in CPU memory (zero-drift perturbation strategy)
    print("Storing base weights in CPU memory...")
    ray.get(engine.collective_rpc.remote("store_base_weights"))
    print("Base weights stored.")

    gen_cfg = config["generation"]
    num_perturbations = gen_cfg["num_perturbations"]
    base_seed = config.get("seed", 42)
    greedy_params = SamplingParams(
        temperature=0.0, max_tokens=gen_cfg["max_new_tokens"]
    )

    all_results = {}

    # --- Greedy baseline (no perturbation) ---
    print("\n=== Greedy baseline (no perturbation) ===")
    outputs = ray.get(
        engine.generate.remote(prompts, greedy_params, use_tqdm=True)
    )

    greedy_correct = 0
    for out, data in zip(outputs, task_datas):
        response = RESPONSE_PROMPT + out.outputs[0].text
        r = reward_function(response, data["numbers"], data["target"])
        if r["reward_info"]["answer_reward"] > 0:
            greedy_correct += 1

    greedy_acc = greedy_correct / len(task_datas) * 100
    print(f"Base accuracy: {greedy_correct}/{len(task_datas)} ({greedy_acc:.2f}%)")
    all_results["greedy_baseline"] = {
        "accuracy": greedy_acc,
        "correct": greedy_correct,
        "total": len(task_datas),
    }

    # --- Weight perturbation sampling ---
    for sigma in config["sigmas"]:
        print(
            f"\n=== Sigma {sigma} ({num_perturbations} perturbations) ==="
        )
        correct_counts = [0] * len(task_datas)

        # Deterministic seeds for this sigma
        rng = np.random.default_rng(base_seed)
        seeds = rng.integers(0, 2**30, size=num_perturbations).tolist()

        t_start = time.time()
        for p_idx, seed in enumerate(seeds):
            # Reset to base + apply noise
            ray.get(
                engine.collective_rpc.remote(
                    "apply_perturbation", args=(int(seed), float(sigma))
                )
            )

            # Generate greedy for all prompts
            outputs = ray.get(
                engine.generate.remote(prompts, greedy_params, use_tqdm=False)
            )

            # Score
            for i, (out, data) in enumerate(zip(outputs, task_datas)):
                response = RESPONSE_PROMPT + out.outputs[0].text
                r = reward_function(response, data["numbers"], data["target"])
                if r["reward_info"]["answer_reward"] > 0:
                    correct_counts[i] += 1

            if (p_idx + 1) % 50 == 0 or p_idx == 0:
                elapsed = time.time() - t_start
                cum_correct = sum(correct_counts)
                total_evals = (p_idx + 1) * len(task_datas)
                print(
                    f"  Perturbation {p_idx + 1}/{num_perturbations} "
                    f"[{elapsed:.1f}s] cumulative correct: {cum_correct}/{total_evals}"
                )

        # Restore base weights before next sigma
        ray.get(engine.collective_rpc.remote("restore_base_weights"))

        # Compute pass@k
        k_values = [k for k in K_VALUES if k <= num_perturbations]
        pass_at_k_results = compute_pass_at_k_dataset(
            correct_counts, num_perturbations, k_values
        )

        prompts_any_correct = sum(c > 0 for c in correct_counts)
        pct_any_correct = prompts_any_correct / len(task_datas) * 100
        avg_correct = float(np.mean(correct_counts))
        elapsed = time.time() - t_start

        print(f"  Done in {elapsed:.1f}s")
        print(
            f"  Prompts with >=1 correct: "
            f"{prompts_any_correct}/{len(task_datas)} ({pct_any_correct:.1f}%)"
        )
        print(f"  Avg correct per prompt: {avg_correct:.2f}/{num_perturbations}")
        print(f"  pass@k: {pass_at_k_results}")

        all_results[f"sigma_{sigma}"] = {
            "sigma": sigma,
            "num_perturbations": num_perturbations,
            "pass_at_k": {str(k): v for k, v in pass_at_k_results.items()},
            "prompts_with_any_correct": prompts_any_correct,
            "avg_correct_per_prompt": avg_correct,
            "elapsed_seconds": elapsed,
            "per_prompt_correct_counts": correct_counts,
        }

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({"config": config, "results": all_results}, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Cleanup
    try:
        ray.kill(engine)
        remove_placement_group(pg)
    except Exception:
        pass
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to config YAML")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output dir")
    parser.add_argument("--debug", action="store_true", help="Debug mode (20 samples)")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
