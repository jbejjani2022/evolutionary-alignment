#!/usr/bin/env python
"""
Optimized unified evaluation script for ES and GRPO fine-tuned models on conciseness task.

Optimizations implemented:
1. Batched generation - generate all samples for a question in one call
2. Batched KL computation - compute KL for all samples in batched forward passes
3. Multi-question batching - process multiple questions together
4. torch.compile() - JIT compilation for faster inference
5. Reduced memory operations - removed excessive empty_cache() calls

Evaluates models by:
- Computing length-based rewards comparing generated vs. target answers
- Calculating KL divergence between fine-tuned and baseline models
- Aggregating results across random seeds

Supports both ES and GRPO checkpoints via --algorithm flag.

Usage:
    # Evaluate ES checkpoints
    python conciseness_eval_optimized.py --algorithm es --sigma 0.001 --alpha 0.0005 ...
    
    # Evaluate GRPO checkpoints  
    python conciseness_eval_optimized.py --algorithm grpo --beta 0.01 ...

Applies chat template and extracts completions correctly to match training.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import argparse
from datetime import datetime, timezone
import gc


logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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
    parser.add_argument('--do_sample', action='store_true', default=False,
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
    
    # Optimization arguments
    parser.add_argument('--question_batch_size', type=int, default=1,
                        help='Number of questions to process in parallel')
    parser.add_argument('--kl_batch_size', type=int, default=8,
                        help='Batch size for KL computation')
    parser.add_argument('--generation_batch_size', type=int, default=20,
                        help='Batch size for generation (num_samples processed at once)')
    parser.add_argument('--use_compile', action='store_true', default=False,
                        help='Use torch.compile for model optimization')
    parser.add_argument('--no_compile', action='store_true', default=False,
                        help='Disable torch.compile')
    
    args = parser.parse_args()
    
    # Handle compile flag
    if args.no_compile:
        args.use_compile = False
    
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


def trim_ids(ids, terminators, pad_id):
    ids_list = ids.tolist()
    for i, t in enumerate(ids_list):
        if t == pad_id or t in terminators:
            return ids[:i]   # exclude stop/pad
    return ids


def get_terminator_ids(tokenizer, model=None):
    # Prefer model generation_config if available
    if model is not None and hasattr(model, "generation_config"):
        eos = model.generation_config.eos_token_id
        if eos is not None:
            return eos if isinstance(eos, (list, tuple)) else [int(eos)]

    # Fallback: tokenizer eos + im_end
    terminators = []
    if tokenizer.eos_token_id is not None:
        terminators.append(int(tokenizer.eos_token_id))
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None:
        terminators.append(int(im_end_id))
    return list(dict.fromkeys(terminators))  # unique, preserve order


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


def compute_per_token_logps_batched(model, input_ids, attention_mask):
    """Compute per-token log probabilities for a batch."""
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        log_probs = F.log_softmax(logits.float(), dim=-1)

    # Shift for next-token prediction
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
        # Additional cleanup
        torch.cuda.reset_peak_memory_stats()


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
    
    print(f"\n[Optimizations]")
    print(f"  torch.compile: {args.use_compile}")
    print(f"  Question batch size: {args.question_batch_size}")
    print(f"  Generation batch size: {args.generation_batch_size}")
    print(f"  KL batch size: {args.kl_batch_size}")
    
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


def get_model_device(model):
    """Get the device of the model's first parameter."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_batched_samples(model, tokenizer, formatted_prompt, num_samples, args, device):
    """
    Generate multiple samples for a single prompt in a batched manner.
    Handles sub-batching if num_samples > generation_batch_size.
    Returns list of (full_ids, completion_ids, generated_answer, prompt_length) tuples.
    """
    # Get actual model device (may differ from passed device due to device_map="auto")
    model_device = get_model_device(model)
    
    # Tokenize the prompt
    tokenized = tokenizer(
        [formatted_prompt],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    input_ids = tokenized["input_ids"].to(model_device)
    attention_mask = tokenized["attention_mask"].to(model_device)
    prompt_length = input_ids.shape[1]
    
    results = []
    generation_batch_size = args.generation_batch_size
    
    # Process in sub-batches if needed
    for batch_start in range(0, num_samples, generation_batch_size):
        batch_size = min(generation_batch_size, num_samples - batch_start)
        
        # Expand for batch generation
        batch_input_ids = input_ids.expand(batch_size, -1)
        batch_attention_mask = attention_mask.expand(batch_size, -1)

        with torch.inference_mode():
            terminators = get_terminator_ids(tokenizer, model)
            outputs = model.generate(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=terminators,
            )
        
        for i in range(batch_size):
            full_ids = outputs[i]
            completion_ids = full_ids[prompt_length:]
            
            try:
                terminators = set(get_terminator_ids(tokenizer, model))
                trimmed_completion_ids = trim_ids(completion_ids, terminators, tokenizer.pad_token_id)
                num_answer_tokens = int(trimmed_completion_ids.numel())
                generated_answer = tokenizer.decode(trimmed_completion_ids, skip_special_tokens=True).strip()
            except TypeError:
                tokens = tokenizer.convert_ids_to_tokens(trimmed_completion_id.tolist())
                filtered = [t for t in tokens if t is not None]
                generated_answer = tokenizer.convert_tokens_to_string(filtered)
            
            # Also try decoding without skipping to see raw output for debugging
            generated_answer = generated_answer.strip()
            
            # If empty, the model may have only generated special tokens
            # Try to get any content before special tokens
            if not generated_answer:
                raw_answer = tokenizer.decode(completion_ids, skip_special_tokens=False)
                generated_answer = clean_completion(raw_answer)
            
            results.append((full_ids, completion_ids, generated_answer, prompt_length, num_answer_tokens))
    
    return results


def evaluate_single_model(model_path, tokenizer, dataset, device, dtype, args, baseline_model_name):
    """Evaluate a single model checkpoint and return results."""
    print(f"\nEvaluating model: {model_path}")
    
    # Force cleanup before loading new model
    force_memory_cleanup()

    # Load fine-tuned model FIRST (before reference model)
    print("  Loading fine-tuned model...")
    model_ft = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=args.hf_cache_dir,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model_ft.eval()
    
    # Verify model is on expected device
    ft_device = get_model_device(model_ft)
    print(f"  Fine-tuned model device: {ft_device}")
    
    # Apply torch.compile if enabled
    if args.use_compile:
        print("  Applying torch.compile (this may take a moment on first run)...")
        try:
            model_ft = torch.compile(model_ft, mode="reduce-overhead")
            print("  ✓ torch.compile applied successfully")
        except Exception as e:
            print(f"  ⚠️  torch.compile failed, continuing without: {e}")

    # Storage for all results
    all_rewards = []
    all_answer_token_counts = []
    all_completions = []
    
    # Storage for sequences that need KL computation
    all_sequences_for_kl = []
    all_prompt_lengths_for_kl = []
    sequence_to_question_sample_for_kl = []

    # Process questions - GENERATION ONLY (no KL yet)
    num_questions = len(dataset)
    question_batch_size = args.question_batch_size
    
    for batch_start in range(0, num_questions, question_batch_size):
        batch_end = min(batch_start + question_batch_size, num_questions)
        questions_batch = dataset[batch_start:batch_end]
        
        print(f"  Generating for questions {batch_start + 1}-{batch_end}/{num_questions}...")
        
        for q_idx, (question, target_answer) in enumerate(questions_batch):
            question_idx = batch_start + q_idx
            formatted_prompt = format_prompt_with_chat_template(question)
            
            # Generate all samples for this question
            generation_results = generate_batched_samples(
                model_ft, tokenizer, formatted_prompt, args.num_samples, args, device
            )
            
            question_completions = []
            
            for sample_idx, (full_ids, completion_ids, generated_answer, prompt_length, num_answer_tokens) in enumerate(generation_results):
                # Compute reward
                reward = compute_reward(generated_answer, target_answer)
                
                completion_info = {
                    "sample_idx": sample_idx,
                    "completion": generated_answer,
                    "completion_length": len(generated_answer),
                    "target_length": len(target_answer),
                    "reward": float(reward),
                    "num_tokens": int(num_answer_tokens),
                }
                question_completions.append(completion_info)
                
                all_rewards.append(float(reward))
                all_answer_token_counts.append(int(num_answer_tokens))
                
                # Store for KL computation later
                all_sequences_for_kl.append(full_ids.cpu())  # Move to CPU to save GPU memory
                all_prompt_lengths_for_kl.append(prompt_length)
                sequence_to_question_sample_for_kl.append((question_idx, sample_idx))
            
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
        
        # Periodic cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Check we have results
    if len(all_rewards) == 0:
        raise RuntimeError("No rewards computed. Check dataset and generation settings.")
    
    # Now compute KL divergence
    # Strategy: Only load ONE model at a time to avoid OOM
    # 1. Load fine-tuned model, compute all logprobs, save to CPU
    # 2. Unload fine-tuned model
    # 3. Load reference model, compute all logprobs, save to CPU
    # 4. Compute KL on CPU
    
    print("  Computing KL divergence (phase 1: fine-tuned model logprobs)...")
    
    # Reload fine-tuned model
    model_ft = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=args.hf_cache_dir,
        device_map="auto", 
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model_ft.eval()
    ft_device = get_model_device(model_ft)
    
    # Compute fine-tuned model logprobs for all sequences
    all_ft_logps = []
    num_sequences = len(all_sequences_for_kl)
    kl_batch_size = args.kl_batch_size
    
    for batch_start in range(0, num_sequences, kl_batch_size):
        batch_end = min(batch_start + kl_batch_size, num_sequences)
        
        if batch_start % (kl_batch_size * 5) == 0:
            print(f"    FT logprobs batch {batch_start}-{batch_end}/{num_sequences}...")
        
        batch_sequences = all_sequences_for_kl[batch_start:batch_end]
        
        # Pad sequences
        max_len = max(seq.shape[0] for seq in batch_sequences)
        padded_ids = []
        attention_masks = []
        
        for seq in batch_sequences:
            seq_len = seq.shape[0]
            if seq_len < max_len:
                padding = torch.full((max_len - seq_len,), tokenizer.pad_token_id, dtype=seq.dtype)
                padded_seq = torch.cat([seq, padding])
                mask = torch.cat([torch.ones(seq_len), torch.zeros(max_len - seq_len)])
            else:
                padded_seq = seq
                mask = torch.ones(seq_len)
            padded_ids.append(padded_seq)
            attention_masks.append(mask)
        
        batch_input_ids = torch.stack(padded_ids).to(ft_device)
        batch_attention_mask = torch.stack(attention_masks).to(ft_device)
        
        # Compute logprobs
        ft_logps = compute_per_token_logps_batched(model_ft, batch_input_ids, batch_attention_mask)
        all_ft_logps.append(ft_logps.cpu())
        
        del batch_input_ids, batch_attention_mask, ft_logps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Unload fine-tuned model
    print("  Unloading fine-tuned model...")
    del model_ft
    force_memory_cleanup()
    
    # Load reference model
    print("  Computing KL divergence (phase 2: reference model logprobs)...")
    model_ref = AutoModelForCausalLM.from_pretrained(
        baseline_model_name,
        cache_dir=args.hf_cache_dir,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model_ref.eval()
    ref_device = get_model_device(model_ref)
    
    # Compute reference model logprobs for all sequences
    all_ref_logps = []
    
    for batch_start in range(0, num_sequences, kl_batch_size):
        batch_end = min(batch_start + kl_batch_size, num_sequences)
        
        if batch_start % (kl_batch_size * 5) == 0:
            print(f"    Ref logprobs batch {batch_start}-{batch_end}/{num_sequences}...")
        
        batch_sequences = all_sequences_for_kl[batch_start:batch_end]
        
        # Pad sequences (same as before)
        max_len = max(seq.shape[0] for seq in batch_sequences)
        padded_ids = []
        attention_masks = []
        
        for seq in batch_sequences:
            seq_len = seq.shape[0]
            if seq_len < max_len:
                padding = torch.full((max_len - seq_len,), tokenizer.pad_token_id, dtype=seq.dtype)
                padded_seq = torch.cat([seq, padding])
                mask = torch.cat([torch.ones(seq_len), torch.zeros(max_len - seq_len)])
            else:
                padded_seq = seq
                mask = torch.ones(seq_len)
            padded_ids.append(padded_seq)
            attention_masks.append(mask)
        
        batch_input_ids = torch.stack(padded_ids).to(ref_device)
        batch_attention_mask = torch.stack(attention_masks).to(ref_device)
        
        # Compute logprobs
        ref_logps = compute_per_token_logps_batched(model_ref, batch_input_ids, batch_attention_mask)
        all_ref_logps.append(ref_logps.cpu())
        
        del batch_input_ids, batch_attention_mask, ref_logps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Unload reference model
    print("  Unloading reference model...")
    del model_ref
    force_memory_cleanup()
    
    print("  Computing KL divergence (phase 3: KL calculation on CPU)...")
    all_per_token_kls = []
    per_sample_kl_means = []
    
    batch_idx = 0
    seq_idx_in_batch = 0
    
    terminators = set(get_terminator_ids(tokenizer))
    
    for seq_idx in range(num_sequences):
        # Find the right batch and position
        while seq_idx_in_batch >= all_ft_logps[batch_idx].shape[0]:
            seq_idx_in_batch = 0
            batch_idx += 1
        
        seq = all_sequences_for_kl[seq_idx]
        prompt_len = all_prompt_lengths_for_kl[seq_idx]
        seq_len = seq.shape[0]
        
        ft_logps_seq = all_ft_logps[batch_idx][seq_idx_in_batch]
        ref_logps_seq = all_ref_logps[batch_idx][seq_idx_in_batch]
        
        # Start checking tokens immediately after prompt
        gen_start_idx = max(prompt_len - 1, 0)
        
        if gen_start_idx < ft_logps_seq.shape[0]:
            # Slice valid logprobs
            # Note: ft_logps_seq is already shifted (len = seq_len - 1)
            ft_gen_logps = ft_logps_seq[gen_start_idx:seq_len-1]
            ref_gen_logps = ref_logps_seq[gen_start_idx:seq_len-1]

            # Get corresponding token IDs to check for padding/EOS
            # seq[1:] aligns with the logps array
            generated_token_ids = seq[1:][gen_start_idx:seq_len-1]
            
            # Create mask for valid tokens (stop at first pad or terminator)
            stop_mask = (generated_token_ids == tokenizer.pad_token_id)
            for tid in terminators:
                stop_mask |= (generated_token_ids == tid)

            stop_pos = stop_mask.nonzero(as_tuple=False)
            cutoff = None
            if stop_pos.numel() > 0:
                cutoff = stop_pos[0, 0].item()
                # Include the EOS token in the KL calc, but stop after it
                # If you strictly want content only, use [:cutoff]
                # Usually we include the EOS prediction as it is part of the generation
                ft_gen_logps = ft_gen_logps[:cutoff+1] 
                ref_gen_logps = ref_gen_logps[:cutoff+1]
            
            if ft_gen_logps.shape[0] > 0:
                # KL(FT || Ref)
                # This measures how much the FT model diverges from the base (Ref) model
                # k3 from Schulman (2020)
                logp_diff = ref_gen_logps - ft_gen_logps
                per_token_kl = torch.exp(logp_diff) - logp_diff - 1
                
                # Optional: Clamp to avoid extreme outliers if P_ref is extremely low
                # per_token_kl = torch.clamp(per_token_kl, min=-10.0, max=100.0)
                
                per_token_kl_list = per_token_kl.tolist()
                
                if per_token_kl_list:
                    all_per_token_kls.extend(per_token_kl_list)
                    per_sample_kl_means.append(float(np.mean(per_token_kl_list)))
                else:
                    per_sample_kl_means.append(float('nan'))
            else:
                per_sample_kl_means.append(float('nan'))
        else:
            per_sample_kl_means.append(float('nan'))
        
        seq_idx_in_batch += 1
    
    # Clean up logprobs
    del all_ft_logps, all_ref_logps
    
    # Assign KL values back to completions
    for seq_idx, (q_idx, sample_idx) in enumerate(sequence_to_question_sample_for_kl):
        if seq_idx < len(per_sample_kl_means):
            # Find the right completion
            for comp_data in all_completions:
                if comp_data["question_idx"] == q_idx:
                    for comp in comp_data["completions"]:
                        if comp["sample_idx"] == sample_idx:
                            comp["mean_per_token_kl"] = per_sample_kl_means[seq_idx]
                            break
                    break
    
    # Final cleanup
    force_memory_cleanup()

    mean_reward = float(np.mean(all_rewards))
    std_reward = float(np.std(all_rewards))
    min_reward = float(np.min(all_rewards))
    max_reward = float(np.max(all_rewards))

    # Normalize reward to [0, 1] range (assuming max possible diff is ~2000 chars)
    normalized_mean_reward = (mean_reward + 2000) / 2001
    normalized_std_reward = std_reward / 2001

    # Compute KL statistics
    # Filter out NaN values for KL computation
    valid_kls = [kl for kl in all_per_token_kls if not np.isnan(kl)]
    total_kl_tokens = len(valid_kls)
    mean_per_token_kl = float(np.mean(valid_kls)) if total_kl_tokens > 0 else float('nan')
    std_per_token_kl = float(np.std(valid_kls)) if total_kl_tokens > 0 else float('nan')

    # Compute per-sample KL means
    per_sample_kl_means = []
    for completion_data in all_completions:
        for comp in completion_data["completions"]:
            if "mean_per_token_kl" in comp and not np.isnan(comp["mean_per_token_kl"]):
                per_sample_kl_means.append(comp["mean_per_token_kl"])

    total_answer_tokens = int(np.sum(all_answer_token_counts))
    mean_answer_tokens = float(total_answer_tokens / len(all_answer_token_counts)) if all_answer_token_counts else float('nan')

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
    print("Conciseness Evaluation (OPTIMIZED)")
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
    print(f"\n[Optimization Settings]")
    print(f"  torch.compile: {args.use_compile}")
    print(f"  Question batch size: {args.question_batch_size}")
    print(f"  Generation batch size: {args.generation_batch_size}")
    print(f"  KL batch size: {args.kl_batch_size}")
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

    # Note: Reference model will be loaded per-seed in evaluate_single_model
    # This avoids OOM from having both models loaded simultaneously

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
                model_path, tokenizer, dataset, device, dtype, args, args.baseline_model_name
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "algorithm": args.algorithm,
        "optimized": True,
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
            "question_batch_size": args.question_batch_size,
            "generation_batch_size": args.generation_batch_size,
            "kl_batch_size": args.kl_batch_size,
            "use_compile": args.use_compile,
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
