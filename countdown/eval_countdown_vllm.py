# Eval script for accelerated countdown checkpoints (ES fine-tuning).
#
# NOTE: This script applies the model's chat template via _process_context()
# to match training-time prompt formatting exactly. The raw "context" field
# in the JSON data file is ignored — do NOT pass it directly to the model.
#
# Example usage:
# python eval_countdown_vllm.py \
#     --model_id "Qwen/Qwen2.5-3B-Instruct" \
#     --trained_model_path <path to your .pth file saved by es_accl_static.py> \
#     --eval_data_path "countdown/data/countdown.json" \
#     --eval_samples 2000 \
#     --eval_offset -2000 \
#     --max_new_tokens 1024 \
#     --batch_size 1024 \
#     --dtype float16 \
#     --seed 42 \
#     --save_responses \
#     --show_examples 5

import os
import sys
import json
import time
import uuid
import argparse
import tempfile
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
from vllm import LLM, SamplingParams, TokensPrompt
from transformers import AutoTokenizer

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import placement_group

# Must match training script (es_accl_static.py) exactly
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


def _process_context(task_data, tokenizer):
    """Build a properly chat-templated TokensPrompt. Copied from training script."""
    numbers = task_data["numbers"]
    target = task_data["target"]
    user_content = USER_TEMPLATE.format(numbers=numbers, target=target)
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content}
    ]

    messages = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    ) + RESPONSE_PROMPT

    prompts = tokenizer(
        messages,
        add_special_tokens=False
    )

    return TokensPrompt(prompt_token_ids=prompts['input_ids'])

def parse_args():
    parser = argparse.ArgumentParser(description='vLLM evaluation for ES models (Qwen/Llama/etc)')
    parser.add_argument('--model_id', type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help='HF model name for vLLM')
    parser.add_argument('--trained_model_path', type=str, default=None,
                    help='Path to trained .pth weights (omit to evaluate base model)')

    # Data args
    parser.add_argument('--eval_data_path', type=str,
                        default='/n/holylabs/LABS/sham_lab/Users/jbejjani/evolutionary-alignment/countdown/data/countdown.json',
                        help='Path to evaluation data JSON file')
    parser.add_argument('--eval_samples', type=int, default=100,
                        help='Number of evaluation samples to evaluate')
    parser.add_argument('--eval_offset', type=int, default=-100,
                        help='Offset for evaluation data (negative means from end)')

    # Generation args
    parser.add_argument('--max_new_tokens', type=int, default=1024,
                        help='Maximum number of new tokens to generate')
    parser.add_argument('--do_sample', action='store_true',
                        help='Whether to use sampling instead of greedy decoding')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p for nucleus sampling')

    # Batch/engine args
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for prompts per vLLM.generate call (default: min(32, dataset_size))')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                        help='Tensor parallelism for vLLM')
    parser.add_argument('--dtype', type=str, default='float16',
                        choices=['float16', 'bfloat16', 'float32'],
                        help='Model dtype for vLLM')

    # Output/verbosity
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save inference results (default: based on model name)')
    parser.add_argument('--save_responses', action='store_true',
                        help='Save individual responses to file')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output')
    parser.add_argument('--show_examples', type=int, default=5,
                        help='Number of examples to show in detail')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible generation')
    parser.add_argument('--hf_cache_dir', type=str, default=None,
                        help='Custom HuggingFace cache directory')
    return parser.parse_args()


class ESNcclLLM(LLM):
    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)

def launch_engines(num_engines, model_name, dtype="float16"):
    pgs = [placement_group([{"GPU": 1, "CPU": 0}], lifetime="detached") for _ in range(num_engines)]
    ray.get([pg.ready() for pg in pgs])

    strategies = [
        PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )
        for pg in pgs
    ]

    engines = [
        ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(ESNcclLLM).remote(
            model=model_name,
            tensor_parallel_size=1,
            distributed_executor_backend="ray",
            worker_extension_cls="utils.worker_extn.WorkerExtension",
            dtype=dtype,
            enable_prefix_caching=False,
            enforce_eager=False,
        )
        for strategy in strategies
    ]
    return engines, pgs

def load_data(data_path: str, num_samples: int = None, offset: int = 0) -> List[Dict[str, Any]]:
    """Load task data as full dicts (with 'numbers', 'target', etc.)."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    with open(data_path, 'r') as f:
        dataset = json.load(f)

    if offset < 0:
        start_idx, end_idx = len(dataset) + offset, len(dataset)
    else:
        start_idx, end_idx = offset, len(dataset)
    dataset = dataset[start_idx:end_idx]

    if num_samples is not None and num_samples < len(dataset):
        dataset = dataset[:num_samples]
    return dataset


def evaluate_batch_vllm(llm, batch_data: List[Dict[str, Any]], args, verbose: bool = False) -> List[Dict[str, Any]]:
    """Evaluate a batch of task dicts. Prompts must already have 'context' key (TokensPrompt)."""
    if verbose:
        print(f"Batch evaluating {len(batch_data)} samples...")

    temperature = args.temperature if args.do_sample else 0.0
    top_p = args.top_p if args.do_sample else 1.0
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=args.max_new_tokens,
        seed=getattr(args, 'seed', None),
    )

    prompts = [d["context"] for d in batch_data]
    outputs = ray.get(llm.generate.remote(prompts, sampling_params=sampling_params, use_tqdm=False))

    all_results = []
    for out, data in zip(outputs, batch_data):
        completion_text = out.outputs[0].text if out.outputs else ""
        # Prepend RESPONSE_PROMPT to match training-time reward computation
        response = RESPONSE_PROMPT + completion_text

        numbers = data["numbers"]
        target = data["target"]

        reward_result = reward_function(response, numbers, target)
        reward = reward_result["reward"]
        reward_info = reward_result["reward_info"]

        all_results.append({
            'numbers': numbers,
            'target': target,
            'completion': completion_text,
            'full_response': response,
            'reward': reward,
            'reward_info': reward_info,
        })
    return all_results


def evaluate_dataset_vllm(llm, dataset: List[Dict[str, Any]], args, dataset_name: str, batch_size: int = None) -> Dict[str, Any]:
    print(f"\n=== Evaluating on {dataset_name} dataset ({len(dataset)} samples) ===")
    if batch_size is None:
        batch_size = min(1024, len(dataset))
    print(f"Using batch size: {batch_size}")

    all_results = []
    total_reward = 0.0
    total_format_reward = 0.0
    total_answer_reward = 0.0

    start_time = time.time()
    for batch_start in range(0, len(dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch_data = dataset[batch_start:batch_end]

        if args.verbose:
            print(f"Processing batch {batch_start//batch_size + 1}/{(len(dataset)-1)//batch_size + 1} (samples {batch_start+1}-{batch_end})...")

        batch_results = evaluate_batch_vllm(llm, batch_data, args, verbose=args.verbose)
        all_results.extend(batch_results)

        for result in batch_results:
            total_reward += result['reward']
            total_format_reward += result['reward_info']['format_reward']
            total_answer_reward += result['reward_info']['answer_reward']

        if batch_start == 0:
            for i, result in enumerate(batch_results[:args.show_examples]):
                print(f"\n--- Example {i+1} ---")
                print(f"Numbers: {result['numbers']}, Target: {result['target']}")
                print(f"Response: {result['full_response']}")
                print(f"Reward: {result['reward']:.4f} (Format: {result['reward_info']['format_reward']:.4f}, Answer: {result['reward_info']['answer_reward']:.4f})")

    eval_time = time.time() - start_time

    avg_reward = total_reward / len(dataset)
    avg_format_reward = total_format_reward / len(dataset)
    avg_answer_reward = total_answer_reward / len(dataset)

    rewards = [r['reward'] for r in all_results]
    std_reward = np.std(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)

    high_reward_count = sum(1 for r in rewards if r >= 1.0)
    high_reward_percentage = high_reward_count / len(dataset) * 100

    answer_rewards = [r['reward_info']['answer_reward'] for r in all_results]
    correct_count = sum(1 for r in answer_rewards if r > 0)
    accuracy = correct_count / len(dataset) * 100

    stats = {
        'dataset_name': dataset_name,
        'num_samples': len(dataset),
        'avg_reward': avg_reward,
        'avg_format_reward': avg_format_reward,
        'avg_answer_reward': avg_answer_reward,
        'std_reward': std_reward,
        'min_reward': min_reward,
        'max_reward': max_reward,
        'high_reward_count': high_reward_count,
        'high_reward_percentage': high_reward_percentage,
        'correct_count': correct_count,
        'accuracy': accuracy,
        'eval_time': eval_time,
        'all_results': all_results
    }

    print(f"\n=== {dataset_name} Results Summary ===")
    print(f"Number of samples: {len(dataset)}")
    print(f"Average reward: {avg_reward:.4f} ± {std_reward:.4f}")
    print(f"  - Format reward: {avg_format_reward:.4f}")
    print(f"  - Answer reward: {avg_answer_reward:.4f}")
    print(f"Accuracy (answer_reward > 0): {correct_count}/{len(dataset)} ({accuracy:.1f}%)")
    print(f"High reward samples (≥1.0): {high_reward_count}/{len(dataset)} ({high_reward_percentage:.1f}%)")
    print(f"Reward range: [{min_reward:.4f}, {max_reward:.4f}]")
    print(f"Evaluation time: {eval_time:.2f}s ({eval_time/len(dataset):.3f}s per sample)")

    return stats


def save_results(results: Dict[str, Any], output_dir: str, args):
    # Create a timestamped subdirectory with a short unique ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    run_dir = os.path.join(output_dir, f"run_{timestamp}_{short_id}")
    os.makedirs(run_dir, exist_ok=True)

    summary = {
        'run_id': f"{timestamp}_{short_id}",
        'model_id': args.model_id,
        'trained_model_path': args.trained_model_path,
        'eval_stats': {k: v for k, v in results['eval_stats'].items() if k != 'all_results'},
        'generation_config': {
            'max_new_tokens': args.max_new_tokens,
            'do_sample': args.do_sample,
            'temperature': args.temperature if args.do_sample else None,
            'top_p': args.top_p if args.do_sample else None,
            'seed': args.seed,
            'batch_size': args.batch_size,
            'tensor_parallel_size': args.tensor_parallel_size,
            'dtype': args.dtype,
        }
    }

    summary_path = os.path.join(run_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    if args.save_responses:
        eval_details_path = os.path.join(run_dir, 'eval_detailed_results.json')
        with open(eval_details_path, 'w') as f:
            json.dump(results['eval_stats']['all_results'], f, indent=2)
        print(f"Eval detailed results saved to: {eval_details_path}")

    print(f"All results for this run saved under: {run_dir}")
    return run_dir


def main():
    args = parse_args()

    if args.hf_cache_dir is not None:
        os.environ["HF_HOME"] = args.hf_cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = args.hf_cache_dir

    os.environ.pop("RAY_ADDRESS", None)
    os.environ.pop("RAY_HEAD_IP", None)
    os.environ.pop("RAY_GCS_SERVER_ADDRESS", None)

    # Ensure the repo root is on PYTHONPATH so Ray workers can import `utils`
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    existing = os.environ.get("PYTHONPATH", "")
    if repo_root not in existing.split(os.pathsep):
        os.environ["PYTHONPATH"] = repo_root + (os.pathsep + existing if existing else "")

    unique_dir = tempfile.mkdtemp(prefix=f"ray_temp_session_{int(time.time())}_")

    ray.init(
        address="local",
        include_dashboard=False,
        ignore_reinit_error=True,
        _temp_dir=unique_dir, 
        dashboard_port=None 
    )

    global reward_function
    from countdown_task import reward_function

    if args.output_dir is None:
        model_name = os.path.basename(args.model_id.rstrip('/'))
        batch_suffix = f"_batch{args.batch_size}" if args.batch_size else ""
        args.output_dir = f"./inference_results_vllm_{model_name}{batch_suffix}"

    print("=== vLLM ES Model Inference Script ===")
    print(f"Model: {args.model_id}")
    print(f"Trained weights: {args.trained_model_path}")
    print(f"Eval data: {args.eval_data_path} (samples: {args.eval_samples}, offset: {args.eval_offset})")
    print(f"Output directory: {args.output_dir}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"Dtype: {args.dtype}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")

    # Load tokenizer for chat template application
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Initialize vLLM engine with correct dtype
    llm, _ = launch_engines(
        num_engines=1,
        model_name=args.model_id,
        dtype=args.dtype,
    )
    llm = llm[0]
    if args.trained_model_path is not None:
        ray.get(llm.collective_rpc.remote("load_weights_from_disk", args=(args.trained_model_path,)))
        print(f"Loaded trained weights from {args.trained_model_path}")
    else:
        print(f"Evaluating base model: {args.model_id}")

    # Load dataset as full dicts
    eval_dataset = load_data(
        args.eval_data_path,
        num_samples=args.eval_samples,
        offset=args.eval_offset
    )
    print(f"Loaded {len(eval_dataset)} evaluation samples")

    # Apply chat template to all samples (matches training script exactly)
    for d in eval_dataset:
        d["context"] = _process_context(d, tokenizer)

    # Evaluate
    eval_stats = evaluate_dataset_vllm(llm, eval_dataset, args, "Eval", batch_size=args.batch_size)
    results = {'eval_stats': eval_stats}
    save_results(results, args.output_dir, args)


if __name__ == "__main__":
    main()
