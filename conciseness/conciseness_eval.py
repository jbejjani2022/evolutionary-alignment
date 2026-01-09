#!/usr/bin/env python
"""
Unified evaluation script for ES and GRPO fine-tuned models on conciseness task.

Evaluates models by:
- Computing length-based rewards comparing generated vs. target answers
- Calculating KL divergence between fine-tuned and baseline models
- Aggregating results across random seeds

Supports both ES and GRPO checkpoints via --algorithm flag.

Usage:
    # Evaluate ES checkpoints
    python conciseness_eval.py --algorithm es --sigma 0.001 --alpha 0.0005 ...
    
    # Evaluate GRPO checkpoints  
    python conciseness_eval.py --algorithm grpo --beta 0.01 ...

Applies chat template and extracts completions correctly to match training.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import argparse
from datetime import datetime
import gc


logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate fine-tuned models (ES or GRPO) on conciseness task'
    )
    
    # Algorithm selection
    parser.add_argument('--algorithm', type=str, required=True, choices=['es', 'grpo'],
                        help='Algorithm used for fine-tuning: es or grpo')
    
    # ES-specific arguments
    parser.add_argument('--sigma', type=float, default=None,
                        help='Sigma parameter for ES fine-tuning (required if algorithm=es)')
    parser.add_argument('--alpha', type=float, default=None,
                        help='Alpha parameter for ES fine-tuning (required if algorithm=es)')
    
    # GRPO-specific arguments
    parser.add_argument('--beta', type=float, default=None,
                        help='Beta parameter for GRPO fine-tuning (required if algorithm=grpo)')
    
    # Common arguments
    parser.add_argument('--baseline_model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='Baseline model name for KL calculation')
    parser.add_argument('--hf_cache_dir', type=str, default='hf_cache',
                        help='HuggingFace cache directory')
    parser.add_argument('--precision', type=str, default='bf16', choices=['fp16', 'bf16', 'fp32'],
                        help='Model precision')
    parser.add_argument('--max_new_tokens', type=int, default=128,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--do_sample', action='store_true', default=True,
                        help='Enable sampling vs greedy decoding')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of responses to generate per prompt')
    parser.add_argument('--eval_data_path', type=str, 
                        default='/n/holylabs/LABS/sham_lab/Users/jbejjani/evolutionary-alignment/conciseness/data/eval.jsonl',
                        help='Path to evaluation data file')
    parser.add_argument('--print_examples', action='store_true', default=False,
                        help='Print example generations to stdout')
    parser.add_argument('--output_json', type=str, default=None,
                        help='Path to save results JSON')
    parser.add_argument('--eval_seed', type=int, default=44,
                        help='Seed for evaluation RNG (for reproducibility)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=1.0,
                        help='Top-p for sampling')
    parser.add_argument('--seeds', type=str, default='0,1,2,3',
                        help='Comma-separated list of training seeds to evaluate')
    
    # Model path configuration
    parser.add_argument('--model_path_template', type=str, default=None,
                        help='Custom template for model path. Use {seed}, {sigma}, {alpha}, {beta} placeholders.')
    parser.add_argument('--es_base_dir', type=str, 
                        default='/n/netscratch/sham_lab/Everyone/jbejjani/evolutionary-alignment/conciseness/ES',
                        help='Base directory for ES checkpoints')
    parser.add_argument('--grpo_base_dir', type=str,
                        default='/n/netscratch/sham_lab/Everyone/jbejjani/evolutionary-alignment/conciseness/GRPO',
                        help='Base directory for GRPO checkpoints')
    
    args = parser.parse_args()
    
    # Validate algorithm-specific arguments
    if args.algorithm == 'es':
        if args.sigma is None or args.alpha is None:
            parser.error("--sigma and --alpha are required when --algorithm=es")
    elif args.algorithm == 'grpo':
        if args.beta is None:
            parser.error("--beta is required when --algorithm=grpo")
    
    return args


# Global tokenizer reference for helper functions
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
    
    # Strip repeatedly until no more special tokens
    changed = True
    while changed:
        changed = False
        for token in [im_end_token, eos_token]:
            if token and text.endswith(token):
                text = text[:-len(token)].strip()
                changed = True
    return text


def compute_reward(generated_text: str, target_text: str) -> float:
    """
    Compute reward based on length difference.
    Generated text should already be cleaned (completion only, no special tokens).
    """
    gen_clean = clean_completion(generated_text)
    target_clean = target_text.strip()
    return -abs(len(gen_clean) - len(target_clean))


def compute_per_token_logps(model, input_ids, attention_mask):
    """Compute per-token log probabilities."""
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        log_probs = F.log_softmax(logits.float(), dim=-1)

    shift_log_probs = log_probs[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    per_token_logps = torch.gather(shift_log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    return per_token_logps


def force_memory_cleanup():
    """Force aggressive memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_model_path(args, seed: int) -> str:
    """
    Construct model path based on algorithm and parameters.
    """
    if args.model_path_template:
        # Use custom template
        return args.model_path_template.format(
            seed=seed,
            sigma=args.sigma,
            alpha=args.alpha,
            beta=args.beta,
        )
    
    if args.algorithm == 'es':
        # Default ES path template
        return os.path.join(
            args.es_base_dir,
            f"conciseness/Qwen/Qwen2.5-7B-Instruct/{seed}",
            f"es_random_seed{seed}_pop30_iter1000_sigma{args.sigma}_alpha{args.alpha}_bf16_threads1_question_num2_final"
        )
    else:  # grpo
        # Default GRPO path template
        return os.path.join(
            args.grpo_base_dir,
            f"beta{args.beta}_seed{seed}"
        )


def verify_setup(tokenizer, args):
    """Print verification info for debugging."""
    print("\n" + "=" * 60)
    print("EVAL SETUP VERIFICATION")
    print("=" * 60)
    
    print(f"\n[Algorithm]")
    print(f"  Type: {args.algorithm.upper()}")
    if args.algorithm == 'es':
        print(f"  Sigma: {args.sigma}")
        print(f"  Alpha: {args.alpha}")
    else:
        print(f"  Beta: {args.beta}")
    
    print(f"\n[Tokenizer]")
    print(f"  pad_token: {repr(tokenizer.pad_token)} (id={tokenizer.pad_token_id})")
    print(f"  eos_token: {repr(tokenizer.eos_token)} (id={tokenizer.eos_token_id})")
    print(f"  padding_side: {tokenizer.padding_side}")
    
    im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    print(f"  <|im_end|> token ids: {im_end_id}")
    
    # Test chat template
    test_question = "What is 2+2?"
    formatted = format_prompt_with_chat_template(test_question)
    print(f"\n[Sample Formatted Prompt]")
    print(f"  Raw: {repr(test_question)}")
    print(f"  Formatted: {repr(formatted[:150])}...")
    
    if "<|im_start|>" in formatted:
        print("  ✓ Chat template detected")
    else:
        print("  ⚠️  WARNING: Chat template may not be applied!")
    
    if "<|im_start|>assistant" in formatted:
        print("  ✓ Generation prompt added")
    else:
        print("  ⚠️  WARNING: Missing assistant generation prompt!")
    
    # Show example model path
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    example_path = get_model_path(args, seeds[0])
    print(f"\n[Model Paths]")
    print(f"  Example (seed={seeds[0]}): {example_path}")
    print(f"  Exists: {os.path.exists(example_path)}")
    
    print("\n" + "=" * 60 + "\n")


def evaluate_single_model(model_path, baseline_model, tokenizer, dataset, device, dtype, args):
    """Evaluate a single model checkpoint and return results."""
    print(f"\nEvaluating model: {model_path}")

    # Load fine-tuned model
    model_ft = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=args.hf_cache_dir,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model_ft.eval()

    # Storage for all results
    all_rewards = []
    all_answer_token_counts = []
    all_per_token_kls = []
    per_sample_kl_means = []
    
    # Store ALL completions for JSON output
    all_completions = []

    for question_idx, (question, target_answer) in enumerate(dataset):
        print(f"Processing question {question_idx + 1}/{len(dataset)}: {question[:50]}...")

        # Apply chat template to question
        formatted_prompt = format_prompt_with_chat_template(question)
        
        # Tokenize the formatted prompt
        tokenized_inputs = tokenizer(
            [formatted_prompt],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,  # Chat template already includes special tokens
        )
        input_ids = tokenized_inputs["input_ids"].to(device)
        attention_mask = tokenized_inputs["attention_mask"].to(device)
        prompt_length = input_ids.shape[1]

        question_completions = []

        for sample_idx in range(args.num_samples):
            if args.num_samples > 1 and (sample_idx + 1) % 5 == 0:
                print(f"  Sample {sample_idx + 1}/{args.num_samples}...")

            with torch.inference_mode():
                outputs = model_ft.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            full_ids = outputs[0]
            
            # Extract completion only (not the prompt)
            completion_ids = full_ids[prompt_length:]
            
            try:
                # Decode without skipping special tokens first, then clean
                generated_answer = tokenizer.decode(completion_ids, skip_special_tokens=False)
            except TypeError:
                tokens = tokenizer.convert_ids_to_tokens(completion_ids)
                filtered = [t for t in tokens if t is not None]
                generated_answer = tokenizer.convert_tokens_to_string(filtered)
            
            # Clean the completion (remove <|im_end|>, EOS, etc.)
            generated_answer = clean_completion(generated_answer)

            # Compute reward
            reward = compute_reward(generated_answer, target_answer)
            all_rewards.append(reward)

            # Track token counts
            num_answer_tokens = len(completion_ids)
            all_answer_token_counts.append(int(num_answer_tokens))

            # Store completion info
            completion_info = {
                "sample_idx": sample_idx,
                "completion": generated_answer,
                "completion_length": len(generated_answer),
                "target_length": len(target_answer),
                "reward": float(reward),
                "num_tokens": int(num_answer_tokens),
            }

            # Compute KL divergence
            full_input_ids = full_ids.unsqueeze(0).to(device)
            full_attention_mask = torch.ones_like(full_input_ids, device=device)

            ft_per_token_logps = compute_per_token_logps(model_ft, full_input_ids, full_attention_mask)
            ref_per_token_logps = compute_per_token_logps(baseline_model, full_input_ids, full_attention_mask)

            # KL only on generated tokens (after prompt)
            gen_start_idx = max(prompt_length - 1, 0)
            if gen_start_idx < ft_per_token_logps.shape[1]:
                ft_generated_logps = ft_per_token_logps[:, gen_start_idx:]
                ref_generated_logps = ref_per_token_logps[:, gen_start_idx:]

                generated_token_ids = full_input_ids[:, 1:][:, gen_start_idx:]

                # Truncate at EOS if present
                if tokenizer.eos_token_id is not None:
                    eos_positions = (generated_token_ids[0] == tokenizer.eos_token_id).nonzero(as_tuple=False)
                    if eos_positions.numel() > 0:
                        cutoff = eos_positions[0, 0].item() + 1
                        ft_generated_logps = ft_generated_logps[:, :cutoff]
                        ref_generated_logps = ref_generated_logps[:, :cutoff]

                if ft_generated_logps.shape[1] > 0:
                    # KL(ref || ft) using the formula: exp(log_ref - log_ft) - (log_ref - log_ft) - 1
                    logp_diff = ref_generated_logps - ft_generated_logps
                    per_token_kl = torch.exp(logp_diff) - logp_diff - 1
                    per_token_kl = per_token_kl.squeeze(0)
                    per_token_kl_list = per_token_kl.detach().cpu().tolist()

                    if per_token_kl_list:
                        all_per_token_kls.extend(per_token_kl_list)
                        mean_kl_value = float(np.mean(per_token_kl_list))
                        per_sample_kl_means.append(mean_kl_value)
                        completion_info["mean_per_token_kl"] = mean_kl_value

            question_completions.append(completion_info)

            # Cleanup per-sample
            del full_input_ids, full_attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Store all completions for this question
        all_completions.append({
            "question_idx": question_idx,
            "question": question,
            "target_answer": target_answer,
            "completions": question_completions,
        })

        # Print example if requested
        if args.print_examples and len(question_completions) > 0:
            print(f"\n    [Example] Question: {question}")
            print(f"    [Example] Target: {target_answer} (len={len(target_answer)})")
            print(f"    [Example] Generated: {question_completions[0]['completion']} (len={question_completions[0]['completion_length']})")
            print(f"    [Example] Reward: {question_completions[0]['reward']:.2f}")

        # Cleanup per-question
        del input_ids, attention_mask
        force_memory_cleanup()

    # Calculate metrics for this model
    if len(all_rewards) == 0:
        raise RuntimeError("No rewards computed. Check dataset and generation settings.")

    mean_reward = float(np.mean(all_rewards))
    std_reward = float(np.std(all_rewards))
    min_reward = float(np.min(all_rewards))
    max_reward = float(np.max(all_rewards))

    # Normalize reward to [0, 1] range (assuming max possible diff is ~2000 chars)
    normalized_mean_reward = (mean_reward + 2000) / 2001
    normalized_std_reward = std_reward / 2001

    total_kl_tokens = len(all_per_token_kls)
    mean_per_token_kl = float(np.mean(all_per_token_kls)) if total_kl_tokens > 0 else float('nan')
    std_per_token_kl = float(np.std(all_per_token_kls)) if total_kl_tokens > 0 else float('nan')

    total_answer_tokens = int(np.sum(all_answer_token_counts))
    mean_answer_tokens = float(total_answer_tokens / len(all_answer_token_counts)) if all_answer_token_counts else float('nan')

    # Clean up model
    del model_ft
    force_memory_cleanup()

    return {
        "reward": {
            "mean": mean_reward,
            "std": std_reward,
            "min": min_reward,
            "max": max_reward,
            "normalized": {
                "mean": normalized_mean_reward,
                "std": normalized_std_reward,
            },
        },
        "kl": {
            "total_tokens": total_kl_tokens,
            "mean_per_token": mean_per_token_kl,
            "std_per_token": std_per_token_kl,
            "per_sample_means": per_sample_kl_means,
        },
        "answer_tokens": {
            "total": total_answer_tokens,
            "mean_per_sample": mean_answer_tokens,
        },
        "completions": all_completions,
        "num_questions": len(dataset),
        "num_samples_per_question": args.num_samples,
        "total_samples": len(all_rewards),
    }


def main():
    global _tokenizer
    
    args = parse_args()
    
    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    
    # Print header
    print("=" * 80)
    print("Conciseness Evaluation")
    print("=" * 80)
    print(f"Algorithm: {args.algorithm.upper()}")
    if args.algorithm == 'es':
        print(f"  Sigma: {args.sigma}")
        print(f"  Alpha: {args.alpha}")
    else:
        print(f"  Beta: {args.beta}")
    print(f"Baseline model: {args.baseline_model_name}")
    print(f"Eval data path: {args.eval_data_path}")
    print(f"Seeds to evaluate: {seeds}")
    print(f"Precision: {args.precision}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Do sample: {args.do_sample}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Num samples per prompt: {args.num_samples}")
    print("=" * 80)

    # Optional seeding for reproducibility
    if args.eval_seed is not None:
        np.random.seed(args.eval_seed)
        torch.manual_seed(args.eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.eval_seed)

    # Determine device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.precision == 'fp16':
        dtype = torch.float16
    elif args.precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Load tokenizer first (needed for chat template)
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.baseline_model_name,
        use_fast=False,
        cache_dir=args.hf_cache_dir
    )
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set global tokenizer for helper functions
    _tokenizer = tokenizer

    # Verify setup
    verify_setup(tokenizer, args)

    # Load reference model (shared across all evaluations)
    print("Loading reference model...")
    model_ref = AutoModelForCausalLM.from_pretrained(
        args.baseline_model_name,
        cache_dir=args.hf_cache_dir,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model_ref.eval()
    print("Reference model loaded successfully")

    # Load evaluation dataset
    print("\nLoading evaluation dataset...")
    data_path = args.eval_data_path
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    dataset = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                question = item['question']
                answer = item['answer']
                dataset.append((question, answer))

    print(f"Loaded {len(dataset)} evaluation samples")

    # Evaluate all seeds
    seed_results = {}

    for seed in seeds:
        model_path = get_model_path(args, seed)

        if not os.path.exists(model_path):
            print(f"\n⚠️  Model path not found for seed {seed}: {model_path}")
            seed_results[str(seed)] = {"error": f"Model path not found: {model_path}"}
            continue

        try:
            seed_result = evaluate_single_model(
                model_path, model_ref, tokenizer, dataset, device, dtype, args
            )
            seed_results[str(seed)] = seed_result

            # Print individual seed results
            print(f"\n{'='*60}")
            print(f"RESULTS FOR SEED {seed}")
            print(f"{'='*60}")
            print(f"Reward (Length-based):")
            print(f"  Mean: {seed_result['reward']['mean']:.4f}")
            print(f"  Std:  {seed_result['reward']['std']:.4f}")
            print(f"  Min:  {seed_result['reward']['min']:.4f}")
            print(f"  Max:  {seed_result['reward']['max']:.4f}")
            print(f"  Normalized mean: {seed_result['reward']['normalized']['mean']:.4f}")
            print(f"\nKL Divergence:")
            if seed_result['kl']['total_tokens'] > 0:
                print(f"  Mean per-token KL: {seed_result['kl']['mean_per_token']:.6f}")
                print(f"  Std per-token KL: {seed_result['kl']['std_per_token']:.6f}")
                print(f"  Total KL tokens: {seed_result['kl']['total_tokens']}")
            else:
                print("  No generated tokens; KL statistics unavailable.")
            print(f"\nSamples:")
            print(f"  Total samples evaluated: {seed_result['total_samples']}")
            print(f"  Total answer tokens: {seed_result['answer_tokens']['total']}")
            print(f"  Mean answer tokens per sample: {seed_result['answer_tokens']['mean_per_sample']:.2f}")

        except Exception as e:
            import traceback
            print(f"\n❌ Error evaluating seed {seed}: {e}")
            traceback.print_exc()
            seed_results[str(seed)] = {"error": str(e)}

    # Calculate aggregate statistics
    print(f"\n{'='*80}")
    print("AGGREGATE STATISTICS ACROSS SEEDS")
    print(f"{'='*80}")

    # Collect metrics from successful evaluations
    reward_means = []
    normalized_reward_means = []
    kl_means = []
    
    for seed, result in seed_results.items():
        if "error" not in result:
            reward_means.append(result["reward"]["mean"])
            normalized_reward_means.append(result["reward"]["normalized"]["mean"])
            if not np.isnan(result["kl"]["mean_per_token"]):
                kl_means.append(result["kl"]["mean_per_token"])

    if reward_means:
        aggregate_reward_mean = float(np.mean(reward_means))
        aggregate_reward_std = float(np.std(reward_means)) if len(reward_means) > 1 else 0.0
        normalized_reward_mean = float(np.mean(normalized_reward_means))
        normalized_reward_std = float(np.std(normalized_reward_means)) if len(normalized_reward_means) > 1 else 0.0
        print(f"Reward across seeds:")
        print(f"  Mean of means: {aggregate_reward_mean:.4f}")
        print(f"  Std of means: {aggregate_reward_std:.4f}")
        print(f"  Mean of normalized means: {normalized_reward_mean:.4f}")
        print(f"  Std of normalized means: {normalized_reward_std:.4f}")
        print(f"  Successful seeds: {len(reward_means)}")
    else:
        print("No successful reward evaluations to aggregate")
        aggregate_reward_mean = float('nan')
        aggregate_reward_std = float('nan')
        normalized_reward_mean = float('nan')
        normalized_reward_std = float('nan')

    if kl_means:
        aggregate_kl_mean = float(np.mean(kl_means))
        aggregate_kl_std = float(np.std(kl_means)) if len(kl_means) > 1 else 0.0
        print(f"\nKL across seeds:")
        print(f"  Mean of means: {aggregate_kl_mean:.6f}")
        print(f"  Std of means: {aggregate_kl_std:.6f}")
    else:
        print("\nNo successful KL evaluations to aggregate")
        aggregate_kl_mean = float('nan')
        aggregate_kl_std = float('nan')

    print("=" * 80)

    # Prepare results payload
    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "algorithm": args.algorithm,
        "config": {
            "baseline_model_name": args.baseline_model_name,
            "eval_data_path": data_path,
            "precision": args.precision,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_samples": args.num_samples,
            "seeds_requested": seeds,
        },
        "seed_results": seed_results,
        "aggregate": {
            "reward_mean": aggregate_reward_mean,
            "reward_std": aggregate_reward_std,
            "normalized_reward_mean": normalized_reward_mean,
            "normalized_reward_std": normalized_reward_std,
            "kl_mean": aggregate_kl_mean,
            "kl_std": aggregate_kl_std,
            "successful_seeds": len(reward_means),
        },
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    }
    
    # Add algorithm-specific config
    if args.algorithm == 'es':
        results["config"]["sigma"] = args.sigma
        results["config"]["alpha"] = args.alpha
        results["sigma"] = args.sigma
        results["alpha"] = args.alpha
    else:
        results["config"]["beta"] = args.beta
        results["beta"] = args.beta

    # Determine output path
    if args.output_json:
        output_path = args.output_json
    else:
        if args.algorithm == 'es':
            output_path = f'logs/conciseness_eval_es_sigma{args.sigma}_alpha{args.alpha}.json'
        else:
            output_path = f'logs/conciseness_eval_grpo_beta{args.beta}.json'
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved results to {output_path}")
    except Exception as e:
        print(f"\n❌ Failed to save results to {output_path}: {e}")


if __name__ == "__main__":
    main()
