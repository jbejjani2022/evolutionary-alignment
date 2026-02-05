#!/usr/bin/env python
"""
Math Model Evaluation Script.

Comprehensive evaluation for math reasoning models with:
- Pass@k curves up to k_max=1024
- Coverage metrics: P (Precision), E (Efficiency), S (Success), O (Overlap)
- Comparison metrics vs baseline: SRR, NDR, SDS, NSCR
- Multi-dataset support: MATH500, AIME2024, Minerva, OlympiadBench

Usage:
    # Evaluate a single checkpoint on all datasets
    python math_eval.py --model_path /path/to/checkpoint --datasets MATH500,AIME2024

    # Compare fine-tuned model against baseline
    python math_eval.py --model_path /path/to/checkpoint --baseline_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

    # Full evaluation with coverage metrics
    python math_eval.py --model_path /path/to/checkpoint --kmax 1024 --num_samples 1024
"""

import argparse
import gc
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import hashlib

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import importlib.util


# =============================================================================
# Dataset Configuration
# =============================================================================

DATASET_CONFIGS = {
    "MATH500": {
        "hf_name": "HuggingFaceH4/MATH-500",
        "split": "test",
        "question_key": "problem",
        "answer_key": "solution",
    },
    "AIME2024": {
        "hf_name": "AI-MO/aime-2024",
        "split": "train",  # AIME datasets often use 'train' split
        "question_key": "problem",
        "answer_key": "answer",
    },
    "Minerva": {
        "hf_name": "HuggingFaceH4/Minerva-MATH",
        "split": "test",
        "question_key": "problem",
        "answer_key": "solution",
    },
    "OlympiadBench": {
        "hf_name": "maxwell-jia/OlympiadBench_Dataset",
        "split": "test",
        "question_key": "question",
        "answer_key": "final_answer",
        "config": "OE_TO_maths_en",
    },
}

# Default k values for pass@k curve
DEFAULT_K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate math reasoning models with pass@k and coverage metrics"
    )
    
    # Model configuration
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint (or comma-separated list)")
    parser.add_argument("--baseline_model", type=str, default=None,
                        help="Baseline model for coverage comparison (default: same as model)")
    parser.add_argument("--model_dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype")
    
    # Dataset configuration
    parser.add_argument("--datasets", type=str, default="MATH500",
                        help="Comma-separated dataset names: MATH500,AIME2024,Minerva,OlympiadBench")
    parser.add_argument("--max_problems", type=int, default=None,
                        help="Max problems per dataset (None = all)")
    
    # Sampling configuration
    parser.add_argument("--num_samples", type=int, default=64,
                        help="Number of samples per problem (for pass@k)")
    parser.add_argument("--kmax", type=int, default=64,
                        help="Maximum k for pass@k curve")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p sampling")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Maximum tokens to generate")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Directory to save results")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Output JSON filename (auto-generated if not provided)")
    parser.add_argument("--save_generations", action="store_true",
                        help="Save all generated responses to JSON")
    
    # Hardware configuration
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size for vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization for vLLM")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()


# =============================================================================
# Math Verifier
# =============================================================================

def load_math_verifier_module():
    """Dynamically load math_verifier.py."""
    verifier_path = os.path.join(os.path.dirname(__file__), "math_verifier.py")
    spec = importlib.util.spec_from_file_location("math_verifier_local", verifier_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _extract_reward_list(obj) -> List[float]:
    """Normalize outputs from math_verifier.reward_func."""
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


def verify_responses(responses: List[str], prompts: List[str], answers: List[str]) -> List[float]:
    """Verify responses against ground truth answers."""
    mv = load_math_verifier_module()
    rewards = []
    batch_size = 256
    for i in range(0, len(responses), batch_size):
        reward_out = mv.reward_func(
            queries=responses[i:i+batch_size],
            prompts=prompts[i:i+batch_size],
            answers=answers[i:i+batch_size],
        )
        rewards.extend(_extract_reward_list(reward_out))
    return rewards


# =============================================================================
# Dataset Loading
# =============================================================================

def load_eval_dataset(dataset_name: str, max_problems: Optional[int] = None) -> List[Dict]:
    """Load evaluation dataset and return list of {question, answer} dicts."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset_name]
    
    try:
        if "config" in config:
            ds = load_dataset(config["hf_name"], config["config"], split=config["split"])
        else:
            ds = load_dataset(config["hf_name"], split=config["split"])
    except Exception as e:
        print(f"Warning: Failed to load {dataset_name}: {e}")
        return []
    
    q_key = config["question_key"]
    a_key = config["answer_key"]
    
    problems = []
    for record in ds:
        question = record.get(q_key)
        answer = record.get(a_key)
        
        # Handle nested structures
        if question is None and "sample" in record:
            question = record["sample"].get(q_key)
            answer = record["sample"].get(a_key)
        
        if question is not None and answer is not None:
            problems.append({
                "question": str(question),
                "answer": str(answer),
                "dataset": dataset_name,
            })
    
    if max_problems and len(problems) > max_problems:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(problems), size=max_problems, replace=False)
        problems = [problems[i] for i in indices]
    
    print(f"  Loaded {len(problems)} problems from {dataset_name}")
    return problems


# =============================================================================
# Pass@k Computation
# =============================================================================

def compute_pass_at_k_curve(
    correct_counts: List[int],
    num_samples: int,
    k_values: List[int]
) -> Dict[int, float]:
    """
    Compute pass@k for multiple k values using unbiased estimator.
    
    Args:
        correct_counts: List of number of correct samples per problem
        num_samples: Total samples generated per problem (n)
        k_values: List of k values to compute
    
    Returns:
        Dict mapping k -> pass@k score
    """
    results = {}
    n = num_samples
    
    for k in k_values:
        if k > n:
            # Can't compute pass@k when k > n
            results[k] = None
            continue
        
        # Unbiased estimator: pass@k = E[1 - C(n-c, k) / C(n, k)]
        # where c is number of correct samples
        pass_at_k_values = []
        for c in correct_counts:
            if c == 0:
                pass_at_k_values.append(0.0)
            elif c >= k:
                pass_at_k_values.append(1.0)
            else:
                # Use log for numerical stability
                # 1 - C(n-c, k) / C(n, k)
                num = sum(np.log(n - c - i) for i in range(k))
                den = sum(np.log(n - i) for i in range(k))
                pass_at_k_values.append(1.0 - np.exp(num - den))
        
        results[k] = float(np.mean(pass_at_k_values))
    
    return results


# =============================================================================
# Coverage Metrics
# =============================================================================

def compute_solution_hash(response: str) -> str:
    """Compute hash of normalized solution for clustering."""
    # Normalize: lowercase, remove whitespace, extract final answer
    normalized = response.lower().strip()
    # Try to extract boxed answer if present
    if "\\boxed{" in normalized:
        start = normalized.rfind("\\boxed{")
        end = normalized.find("}", start)
        if end > start:
            normalized = normalized[start:end+1]
    return hashlib.md5(normalized.encode()).hexdigest()[:16]


def compute_coverage_metrics(
    problems: List[Dict],
    responses_per_problem: List[List[str]],
    correct_per_problem: List[List[bool]],
) -> Dict[str, float]:
    """
    Compute coverage metrics.
    
    P (Precision): Fraction of unique correct solutions
    E (Efficiency): Total correct / total samples
    S (Success): Problems with at least one correct / total problems
    O (Overlap): Average pairwise similarity of correct responses per problem
    """
    total_samples = 0
    total_correct = 0
    problems_with_success = 0
    unique_solutions_correct = 0
    unique_solutions_total = 0
    overlap_scores = []
    
    for responses, correct in zip(responses_per_problem, correct_per_problem):
        n_correct = sum(correct)
        total_samples += len(responses)
        total_correct += n_correct
        
        if n_correct > 0:
            problems_with_success += 1
        
        # Compute unique solutions
        hashes = [compute_solution_hash(r) for r in responses]
        unique_hashes = set(hashes)
        unique_solutions_total += len(unique_hashes)
        
        correct_hashes = set(h for h, c in zip(hashes, correct) if c)
        unique_solutions_correct += len(correct_hashes)
        
        # Compute overlap (duplicate ratio among correct)
        if n_correct > 1:
            overlap = 1.0 - len(correct_hashes) / n_correct
            overlap_scores.append(overlap)
    
    n_problems = len(problems)
    
    return {
        "P_precision": unique_solutions_correct / unique_solutions_total if unique_solutions_total > 0 else 0.0,
        "E_efficiency": total_correct / total_samples if total_samples > 0 else 0.0,
        "S_success": problems_with_success / n_problems if n_problems > 0 else 0.0,
        "O_overlap": float(np.mean(overlap_scores)) if overlap_scores else 0.0,
    }


def compute_comparison_metrics(
    problems: List[Dict],
    model_responses: List[List[str]],
    model_correct: List[List[bool]],
    baseline_responses: List[List[str]],
    baseline_correct: List[List[bool]],
) -> Dict[str, float]:
    """
    Compute comparison metrics between model and baseline.
    
    SRR (Solution Recoverability Rate): Can model reproduce baseline's correct solutions?
    NDR (Novel Discovery Rate): New correct solutions not found by baseline
    SDS (Solution Diversity Score): Entropy of solution clusters
    NSCR (New Solution Coverage Rate): Coverage delta vs baseline
    """
    srr_scores = []  # Per problem recoverability
    ndr_scores = []  # Per problem novelty
    model_unique_correct = 0
    baseline_unique_correct = 0
    both_unique_correct = 0
    
    for m_responses, m_correct, b_responses, b_correct in zip(
        model_responses, model_correct, baseline_responses, baseline_correct
    ):
        # Get unique correct solution hashes
        m_correct_hashes = set(
            compute_solution_hash(r) for r, c in zip(m_responses, m_correct) if c
        )
        b_correct_hashes = set(
            compute_solution_hash(r) for r, c in zip(b_responses, b_correct) if c
        )
        
        # SRR: fraction of baseline solutions recovered by model
        if len(b_correct_hashes) > 0:
            recovered = len(m_correct_hashes & b_correct_hashes)
            srr_scores.append(recovered / len(b_correct_hashes))
        
        # NDR: fraction of model solutions that are novel
        if len(m_correct_hashes) > 0:
            novel = len(m_correct_hashes - b_correct_hashes)
            ndr_scores.append(novel / len(m_correct_hashes))
        
        # Track unique solution counts
        model_unique_correct += len(m_correct_hashes)
        baseline_unique_correct += len(b_correct_hashes)
        both_unique_correct += len(m_correct_hashes | b_correct_hashes)
    
    # Compute entropy-based diversity
    def solution_diversity_score(responses_list, correct_list):
        all_hashes = []
        for responses, correct in zip(responses_list, correct_list):
            for r, c in zip(responses, correct):
                if c:
                    all_hashes.append(compute_solution_hash(r))
        if not all_hashes:
            return 0.0
        # Compute entropy
        from collections import Counter
        counts = Counter(all_hashes)
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        # Normalize by max entropy
        max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    model_sds = solution_diversity_score(model_responses, model_correct)
    baseline_sds = solution_diversity_score(baseline_responses, baseline_correct)
    
    # NSCR: new solution coverage rate
    nscr = (model_unique_correct - baseline_unique_correct) / baseline_unique_correct if baseline_unique_correct > 0 else 0.0
    
    return {
        "SRR_recoverability": float(np.mean(srr_scores)) if srr_scores else 0.0,
        "NDR_novelty": float(np.mean(ndr_scores)) if ndr_scores else 0.0,
        "SDS_model_diversity": model_sds,
        "SDS_baseline_diversity": baseline_sds,
        "NSCR_coverage_delta": nscr,
    }


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate_model(
    model_path: str,
    problems: List[Dict],
    args,
    model_name: str = "model"
) -> Tuple[List[List[str]], List[List[bool]], Dict]:
    """
    Evaluate a model on given problems.
    
    Returns:
        responses_per_problem: List of response lists per problem
        correct_per_problem: List of correctness lists per problem
        metrics: Dictionary of metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Model: {model_path}")
    print(f"Problems: {len(problems)}")
    print(f"Samples per problem: {args.num_samples}")
    print(f"{'='*60}")
    
    # Load model with vLLM
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.model_dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        seed=args.seed,
    )
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        n=args.num_samples,
        seed=args.seed,
    )
    
    # Prepare prompts
    prompts = [p["question"] for p in problems]
    answers = [p["answer"] for p in problems]
    
    # Generate responses
    print("Generating responses...")
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    
    # Process outputs
    responses_per_problem = []
    correct_per_problem = []
    correct_counts = []
    
    print("Verifying responses...")
    for out, prompt, answer in zip(outputs, prompts, answers):
        responses = [o.text for o in out.outputs]
        responses_per_problem.append(responses)
        
        # Verify all responses
        correctness = verify_responses(
            responses,
            [prompt] * len(responses),
            [answer] * len(responses)
        )
        correct_bools = [c > 0.5 for c in correctness]
        correct_per_problem.append(correct_bools)
        correct_counts.append(sum(correct_bools))
    
    # Compute pass@k curve
    k_values = [k for k in DEFAULT_K_VALUES if k <= args.kmax]
    pass_at_k = compute_pass_at_k_curve(correct_counts, args.num_samples, k_values)
    
    # Compute coverage metrics
    coverage = compute_coverage_metrics(problems, responses_per_problem, correct_per_problem)
    
    # Aggregate metrics
    metrics = {
        "model_path": model_path,
        "num_problems": len(problems),
        "num_samples": args.num_samples,
        "pass_at_k": pass_at_k,
        **coverage,
        "total_correct": sum(correct_counts),
        "total_samples": len(problems) * args.num_samples,
    }
    
    # Cleanup
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return responses_per_problem, correct_per_problem, metrics


def main():
    args = parse_args()
    
    # Parse datasets
    dataset_names = [d.strip() for d in args.datasets.split(",")]
    
    # Parse model paths
    model_paths = [p.strip() for p in args.model_path.split(",")]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Results container
    all_results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "args": vars(args),
        "datasets": {},
    }
    
    # Load all datasets
    print("\n" + "=" * 60)
    print("Loading datasets...")
    print("=" * 60)
    
    datasets = {}
    for ds_name in dataset_names:
        problems = load_eval_dataset(ds_name, args.max_problems)
        if problems:
            datasets[ds_name] = problems
    
    if not datasets:
        print("ERROR: No datasets loaded!")
        sys.exit(1)
    
    # Evaluate each model on each dataset
    for model_path in model_paths:
        model_name = os.path.basename(model_path) or model_path
        
        for ds_name, problems in datasets.items():
            print(f"\n{'#'*60}")
            print(f"# Dataset: {ds_name} | Model: {model_name}")
            print(f"{'#'*60}")
            
            responses, correct, metrics = evaluate_model(
                model_path, problems, args, model_name
            )
            
            # Store results
            result_key = f"{model_name}_{ds_name}"
            all_results["datasets"][result_key] = {
                "model": model_path,
                "dataset": ds_name,
                "metrics": metrics,
            }
            
            if args.save_generations:
                all_results["datasets"][result_key]["generations"] = [
                    {
                        "question": p["question"],
                        "answer": p["answer"],
                        "responses": r,
                        "correct": c,
                    }
                    for p, r, c in zip(problems, responses, correct)
                ]
            
            # Print summary
            print(f"\n--- Results for {ds_name} ---")
            print(f"Pass@1:  {metrics['pass_at_k'].get(1, 'N/A'):.4f}" if metrics['pass_at_k'].get(1) else "Pass@1:  N/A")
            print(f"Pass@8:  {metrics['pass_at_k'].get(8, 'N/A'):.4f}" if metrics['pass_at_k'].get(8) else "Pass@8:  N/A")
            print(f"Pass@64: {metrics['pass_at_k'].get(64, 'N/A'):.4f}" if metrics['pass_at_k'].get(64) else "Pass@64: N/A")
            print(f"Success Rate (S): {metrics['S_success']:.4f}")
            print(f"Efficiency (E):   {metrics['E_efficiency']:.4f}")
            print(f"Precision (P):    {metrics['P_precision']:.4f}")
            
            # If baseline provided, compute comparison metrics
            if args.baseline_model and args.baseline_model != model_path:
                print(f"\nEvaluating baseline for comparison: {args.baseline_model}")
                
                baseline_responses, baseline_correct, baseline_metrics = evaluate_model(
                    args.baseline_model, problems, args, "baseline"
                )
                
                comparison = compute_comparison_metrics(
                    problems, responses, correct, baseline_responses, baseline_correct
                )
                
                all_results["datasets"][result_key]["baseline_metrics"] = baseline_metrics
                all_results["datasets"][result_key]["comparison_metrics"] = comparison
                
                print(f"\n--- Comparison Metrics ---")
                print(f"SRR (Recoverability): {comparison['SRR_recoverability']:.4f}")
                print(f"NDR (Novelty):        {comparison['NDR_novelty']:.4f}")
                print(f"SDS (Diversity):      {comparison['SDS_model_diversity']:.4f}")
                print(f"NSCR (Coverage Δ):    {comparison['NSCR_coverage_delta']:.4f}")
    
    # Save results
    output_filename = args.output_json or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    
    # Print final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for key, data in all_results["datasets"].items():
        metrics = data["metrics"]
        print(f"\n{key}:")
        print(f"  Pass@k: " + ", ".join(
            f"@{k}={v:.3f}" for k, v in sorted(metrics["pass_at_k"].items()) 
            if v is not None
        ))
        print(f"  P={metrics['P_precision']:.3f}, E={metrics['E_efficiency']:.3f}, "
              f"S={metrics['S_success']:.3f}, O={metrics['O_overlap']:.3f}")


if __name__ == "__main__":
    main()
