"""Shared utilities for sampling_analysis_v1 track."""

import json
from typing import Any, Dict, List

import numpy as np
from transformers import AutoTokenizer
from vllm import TokensPrompt

# Must match training/eval scripts exactly
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


def load_countdown_data(
    data_path: str, offset: int = 0, num_samples: int = None
) -> List[Dict[str, Any]]:
    """Load countdown task data from JSON file."""
    with open(data_path, "r") as f:
        dataset = json.load(f)
    dataset = dataset[offset:]
    if num_samples is not None:
        dataset = dataset[:num_samples]
    if len(dataset) == 0:
        raise ValueError(f"No data loaded from {data_path} with offset={offset}")
    return dataset


def build_chat_prompts(
    task_datas: List[Dict], tokenizer: AutoTokenizer
) -> List[TokensPrompt]:
    """Apply chat template to build TokensPrompts for vLLM."""
    prompts = []
    for d in task_datas:
        user_content = USER_TEMPLATE.format(numbers=d["numbers"], target=d["target"])
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_content},
        ]
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        ) + RESPONSE_PROMPT
        tokens = tokenizer(text, add_special_tokens=False)
        prompts.append(TokensPrompt(prompt_token_ids=tokens["input_ids"]))
    return prompts


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., Codex 2021).

    n: total samples, c: number correct, k: k value.
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def compute_pass_at_k_dataset(
    correct_counts: List[int], n_total: int, k_values: List[int]
) -> Dict[int, float]:
    """Compute average pass@k across all prompts."""
    results = {}
    for k in k_values:
        if k > n_total:
            continue
        scores = [pass_at_k(n_total, c, k) for c in correct_counts]
        results[k] = float(np.mean(scores))
    return results
