"""Temperature sampling for Countdown task.

For each temperature, generate N rollouts per prompt from the base model
and compute pass@k curves. Establishes a baseline: how much can pure
token-level diversity (temperature) recover?

Usage:
    python src/sampling_analysis_v1/scripts/temperature_sampling.py \
        configs/sampling_analysis_v1/temperature_sampling.yaml [--overwrite] [--debug]
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml
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
        config["temperatures"] = config["temperatures"][:1]
        gen_cfg = config["generation"]
        gen_cfg["samples_per_prompt"] = 32
        gen_cfg["samples_per_call"] = min(gen_cfg.get("samples_per_call", 64), 32)
        print(f"DEBUG: 32 problems, 1 temperature ({config['temperatures'][0]}), 32 samples/prompt")

    # Init vLLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    prompts = build_chat_prompts(task_datas, tokenizer)

    llm = LLM(
        model=config["model_id"],
        dtype=config["dtype"],
        gpu_memory_utilization=0.65,
        enforce_eager=False,
    )

    gen_cfg = config["generation"]
    total_samples = gen_cfg["samples_per_prompt"]
    samples_per_call = gen_cfg.get("samples_per_call", 64)
    if total_samples % samples_per_call != 0:
        raise ValueError(
            f"samples_per_prompt ({total_samples}) must be divisible by "
            f"samples_per_call ({samples_per_call})"
        )
    num_rounds = total_samples // samples_per_call
    base_seed = config.get("seed", 42)

    all_results = {}

    # --- Greedy baseline (T=0, single sample) ---
    print("\n=== Greedy baseline (T=0) ===")
    greedy_params = SamplingParams(temperature=0.0, max_tokens=gen_cfg["max_new_tokens"])
    outputs = llm.generate(prompts, greedy_params, use_tqdm=True)

    greedy_correct = 0
    for out, data in zip(outputs, task_datas):
        response = RESPONSE_PROMPT + out.outputs[0].text
        r = reward_function(response, data["numbers"], data["target"])
        if r["reward_info"]["answer_reward"] > 0:
            greedy_correct += 1

    greedy_acc = greedy_correct / len(task_datas) * 100
    print(f"Greedy accuracy: {greedy_correct}/{len(task_datas)} ({greedy_acc:.2f}%)")
    all_results["greedy"] = {
        "accuracy": greedy_acc,
        "correct": greedy_correct,
        "total": len(task_datas),
    }

    # --- Temperature sampling ---
    for temp in config["temperatures"]:
        print(f"\n=== Temperature {temp} ({total_samples} samples/prompt, {num_rounds} rounds of {samples_per_call}) ===")
        correct_counts = [0] * len(task_datas)
        t_start = time.time()

        for r_idx in range(num_rounds):
            sampling_params = SamplingParams(
                temperature=temp,
                top_p=gen_cfg["top_p"],
                top_k=gen_cfg.get("top_k", -1),
                max_tokens=gen_cfg["max_new_tokens"],
                n=samples_per_call,
                seed=base_seed + r_idx,
            )
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

            for i, (out, data) in enumerate(zip(outputs, task_datas)):
                for completion in out.outputs:
                    response = RESPONSE_PROMPT + completion.text
                    r = reward_function(response, data["numbers"], data["target"])
                    if r["reward_info"]["answer_reward"] > 0:
                        correct_counts[i] += 1

            elapsed = time.time() - t_start
            done = (r_idx + 1) * samples_per_call
            print(
                f"  Round {r_idx + 1}/{num_rounds} "
                f"({done}/{total_samples} samples) [{elapsed:.1f}s]"
            )

        # Compute pass@k
        k_values = [k for k in K_VALUES if k <= total_samples]
        pass_at_k_results = compute_pass_at_k_dataset(
            correct_counts, total_samples, k_values
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
        print(f"  Avg correct per prompt: {avg_correct:.2f}/{total_samples}")
        print(f"  pass@k: {pass_at_k_results}")

        all_results[f"temp_{temp}"] = {
            "temperature": temp,
            "total_samples": total_samples,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to config YAML")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output dir")
    parser.add_argument("--debug", action="store_true", help="Debug mode (20 samples)")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
