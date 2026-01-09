#!/usr/bin/env python3
"""
Aggregate eval reports into a summary report.json file.
"""
import json
import glob
import os

def check_seed_hacked_reward(seed_data):
    """Check if a seed hacked the reward by producing empty string responses."""
    examples = seed_data.get("examples", [])
    for example in examples:
        if len(example) >= 2:
            response = example[1]  # model response is at index 1
            if response == "":
                return True
    return False

def main():
    logs_dir = os.path.dirname(os.path.abspath(__file__))
    pattern = os.path.join(logs_dir, "temp*_beta*.json")

    report_data = []

    # Find all matching eval files
    eval_files = glob.glob(pattern)

    for eval_file in sorted(eval_files):
        print(f"Processing {os.path.basename(eval_file)}")

        with open(eval_file, 'r') as f:
            data = json.load(f)

        # Extract parameters
        beta = data.get("beta")

        # Extract aggregate metrics
        aggregate = data.get("aggregate", {})
        normalized_reward_mean = aggregate.get("normalized_reward_mean")
        normalized_reward_std = aggregate.get("normalized_reward_std")
        mean_kl = aggregate.get("kl_mean")
        std_kl = aggregate.get("kl_std")

        # Check for reward hacking across all seeds
        seed_results = data.get("seed_results", {})
        hacked_seeds = 0

        for seed_id, seed_data in seed_results.items():
            if check_seed_hacked_reward(seed_data):
                hacked_seeds += 1

        # Create report entry
        entry = {
            "beta": beta,
            "aggregate_normalized_reward_mean": normalized_reward_mean,
            "aggregate_normalized_reward_std": normalized_reward_std,
            "num_seeds_hacked_reward": hacked_seeds,
            "mean_kl": mean_kl,
            "std_kl": std_kl
        }

        report_data.append(entry)

    # Write report.json
    report_path = os.path.join(logs_dir, "report.json")
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)

    print(f"Report written to {report_path}")

if __name__ == "__main__":
    main()
