#!/usr/bin/env python3
"""
Aggregate eval reports into summary report.json files for both ES and GRPO.
"""
import json
import glob
import os

def check_seed_hacked_reward(seed_data):
    """Check if a seed hacked the reward by producing empty string responses."""
    # Get all completions for this seed
    completions_list = seed_data.get("completions", [])
    
    if not completions_list:
        return False
    
    # Check if ALL completions are empty strings
    all_empty = True
    total_completions = 0
    
    for question_data in completions_list:
        question_completions = question_data.get("completions", [])
        for comp in question_completions:
            total_completions += 1
            completion_text = comp.get("completion", "")
            if completion_text != "":
                all_empty = False
                break
        if not all_empty:
            break
    
    # Only report as hacked if there are completions AND all are empty
    return total_completions > 0 and all_empty


def process_grpo_evals(evals_dir):
    """Process GRPO eval files and generate report."""
    pattern = os.path.join(evals_dir, "temp*_beta*.json")
    report_data = []
    
    eval_files = glob.glob(pattern)
    if not eval_files:
        print(f"No GRPO eval files found in {evals_dir}")
        return
    
    for eval_file in sorted(eval_files):
        print(f"Processing GRPO: {os.path.basename(eval_file)}")
        
        with open(eval_file, 'r') as f:
            data = json.load(f)
        
        # Extract parameters
        beta = data.get("beta")
        
        # Extract aggregate metrics
        aggregate = data.get("aggregate", {})
        reward_mean = aggregate.get("reward_mean")
        reward_std = aggregate.get("reward_std")
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
            "aggregate_reward_mean": reward_mean,
            "aggregate_reward_std": reward_std,
            "aggregate_normalized_reward_mean": normalized_reward_mean,
            "aggregate_normalized_reward_std": normalized_reward_std,
            "num_seeds_hacked_reward": hacked_seeds,
            "mean_kl": mean_kl,
            "std_kl": std_kl
        }
        
        report_data.append(entry)
    
    # Write report.json
    report_path = os.path.join(evals_dir, "report.json")
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"GRPO report written to {report_path}")


def process_es_evals(evals_dir):
    """Process ES eval files and generate report."""
    pattern = os.path.join(evals_dir, "alpha*_sigma*.json")
    report_data = []
    
    eval_files = glob.glob(pattern)
    if not eval_files:
        print(f"No ES eval files found in {evals_dir}")
        return
    
    for eval_file in sorted(eval_files):
        print(f"Processing ES: {os.path.basename(eval_file)}")
        
        with open(eval_file, 'r') as f:
            data = json.load(f)
        
        # Extract parameters
        alpha = data.get("alpha")
        sigma = data.get("sigma")
        
        # Extract aggregate metrics
        aggregate = data.get("aggregate", {})
        reward_mean = aggregate.get("reward_mean")
        reward_std = aggregate.get("reward_std")
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
            "alpha": alpha,
            "sigma": sigma,
            "aggregate_reward_mean": reward_mean,
            "aggregate_reward_std": reward_std,
            "aggregate_normalized_reward_mean": normalized_reward_mean,
            "aggregate_normalized_reward_std": normalized_reward_std,
            "num_seeds_hacked_reward": hacked_seeds,
            "mean_kl": mean_kl,
            "std_kl": std_kl
        }
        
        report_data.append(entry)
    
    # Write report.json
    report_path = os.path.join(evals_dir, "report.json")
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"ES report written to {report_path}")


def main():
    # Get the base directory (conciseness/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Process GRPO evals
    grpo_evals_dir = os.path.join(script_dir, "GRPO", "evals")
    if os.path.isdir(grpo_evals_dir):
        process_grpo_evals(grpo_evals_dir)
    else:
        print(f"GRPO evals directory not found: {grpo_evals_dir}")
    
    # Process ES evals
    es_evals_dir = os.path.join(script_dir, "ES", "evals")
    if os.path.isdir(es_evals_dir):
        process_es_evals(es_evals_dir)
    else:
        print(f"ES evals directory not found: {es_evals_dir}")


if __name__ == "__main__":
    main()
