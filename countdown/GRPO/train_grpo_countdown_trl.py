import os, sys, json, random, argparse
import numpy as np
from typing import List, Dict, Any

# Ensure repo root is on sys.path so `countdown.*` imports work regardless of cwd
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ── Prompt constants — MUST match the ES script for a fair comparison ──
SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)
USER_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. "
    "And return the final answer in <answer> </answer> tags, "
    "for example <answer> (1 + 2) / 3 </answer>."
)
RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"


def load_yaml(path: str) -> dict:
    text = open(path, "r").read()
    try:
        return json.loads(text)
    except Exception:
        try:
            import yaml  # type: ignore
            return yaml.safe_load(text)
        except Exception:
            raise RuntimeError(f"Please install pyyaml or provide valid JSON at {path}.")


def read_json_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)


def build_datasets(cfg: dict, tokenizer, use_chat_template: bool = True):
    """
    Return (train_dataset, eval_dataset_or_None) for TRL GRPO.

    When *use_chat_template* is True (default), prompts are constructed from
    the raw ``numbers`` / ``target`` fields using the same chat template +
    RESPONSE_PROMPT as the ES script.  When False, the pre-baked ``context``
    field in the JSON is used directly (no chat template special tokens).
    """
    from datasets import Dataset

    rows = read_json_rows(cfg["data_json"])  # full list

    # Default split: first 200 for train, rest for eval (matches ES script)
    split_train = int(cfg.get("train_samples", 200))
    train_rows = rows[:split_train]
    eval_rows = rows[split_train:]

    nkey = cfg.get("numbers_key", "numbers")
    tkey = cfg.get("target_key", "target")

    def _to_float(value):
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value))
        except Exception:
            return None

    def _build_prompt(row):
        """Apply the chat template exactly as ES's _process_context() does."""
        numbers = row[nkey]
        target = row[tkey]
        user_content = USER_TEMPLATE.format(numbers=numbers, target=target)
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_content},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        ) + RESPONSE_PROMPT
        return prompt

    def _build_prompt_raw(row):
        """Return the pre-baked 'context' field directly (no chat template)."""
        return row["context"]

    prompt_fn = _build_prompt if use_chat_template else _build_prompt_raw

    train_prompts = [
        {
            "prompt": prompt_fn(r),
            "numbers": list(r.get(nkey, [])),
            "target": _to_float(r.get(tkey)),
        }
        for r in train_rows
    ]
    eval_prompts = [
        {
            "prompt": prompt_fn(r),
            "numbers": list(r.get(nkey, [])),
            "target": _to_float(r.get(tkey)),
        }
        for r in eval_rows
    ]

    d_train = Dataset.from_list(train_prompts)
    d_eval = Dataset.from_list(eval_prompts) if eval_prompts else None
    print(f"Chat template: {'ON' if use_chat_template else 'OFF (raw context)'}")
    print(f"Countdown sample prompt (first 300 chars):\n{train_prompts[0]['prompt'][:300]}...")

    return d_train, d_eval


def make_countdown_reward_fn(cfg: dict):
    """
    Training-time reward using the paper's Countdown reward.
    TRL passes dataset columns via **kwargs: we expect numbers, target.
    """

    def reward_fn(completions, **kwargs):
        from countdown.countdown_task import reward_function

        numbers_list = kwargs.get("numbers", [])
        targets_list = kwargs.get("target", [])

        rewards = []
        fmt_acc = 0
        ans_acc = 0

        def _to_float(value):
            if isinstance(value, (int, float)):
                return float(value)
            try:
                return float(str(value))
            except Exception:
                return None

        for i, completion in enumerate(completions):
            text = completion if isinstance(completion, str) else str(completion)

            numbers = numbers_list[i] if i < len(numbers_list) else []
            target = _to_float(targets_list[i]) if i < len(targets_list) else None

            r = reward_function(text, numbers=numbers, target=target)
            rewards.append(float(r["reward"]))
            fmt_acc += 1 if float(r["reward_info"]["format_reward"]) == 1.0 else 0
            ans_acc += 1 if float(r["reward_info"]["answer_reward"]) == 1.0 else 0

        # Lightweight wandb logging on rank 0
        try:
            import wandb
            if os.environ.get("RANK", "0") == "0" and len(rewards) > 0:
                wandb.log({
                    "train/format_accuracy": fmt_acc / max(1, len(rewards)),
                    "train/answer_accuracy": ans_acc / max(1, len(rewards)),
                    "train/reward_raw/mean": float(sum(rewards)) / len(rewards),
                }, commit=False)
        except Exception:
            pass

        return rewards

    return reward_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=None, help="Override LR from config")
    parser.add_argument("--chat_template", dest="chat_template", action="store_true", default=True,
                        help="Apply chat template to prompts (default)")
    parser.add_argument("--no_chat_template", dest="chat_template", action="store_false",
                        help="Pass raw context strings to the model")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg.setdefault("project", "es_countdown")
    cfg.setdefault("entity", None)

    # ── HF cache dir (must be set before any from_pretrained calls) ──
    hf_cache_dir = cfg.get("hf_cache_dir")
    if hf_cache_dir is not None:
        os.environ["HF_HOME"] = hf_cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir

    import torch
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    # ── Reproducibility (before any model / data loading) ──
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    # Tokenizer (loaded early — needed for chat-template prompt construction)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Dataset (prompt construction matches ES scripts)
    train_ds, eval_ds = build_datasets(cfg, tokenizer, use_chat_template=args.chat_template)

    # GRPO args
    use_vllm = bool(int(os.environ.get("USE_VLLM", "0")))
    lr = float(args.learning_rate) if args.learning_rate is not None else float(cfg["learning_rate"])

    # Evaluation strategy: enable periodic eval if eval_steps is set in config
    eval_steps = cfg.get("eval_steps")
    eval_strategy = cfg.get("eval_strategy", "steps" if eval_steps else "no")

    # Reporting: configurable via YAML (default: ["wandb"])
    report_to = cfg.get("report_to", ["wandb"])
    if isinstance(report_to, str):
        report_to = [report_to]

    grpo_args = GRPOConfig(
        output_dir=os.path.join(cfg["output_dir"], f"beta{args.beta}_lr{lr}_seed{args.seed}"),
        seed=args.seed,
        data_seed=args.seed,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        learning_rate=lr,
        lr_scheduler_type=cfg.get("lr_scheduler_type", "constant"),
        warmup_steps=cfg.get("warmup_steps", 0),
        num_train_epochs=cfg.get("num_train_epochs", 1),
        max_steps=cfg.get("max_steps", 1000),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 200),
        report_to=report_to,
        run_name=f"{cfg['model_name'].split('/')[-1]}_grpo_countdown_beta{args.beta}_lr{lr}_seed{args.seed}",
        bf16=True,
        remove_unused_columns=False,

        # evaluation
        do_eval=eval_ds is not None,
        eval_strategy=eval_strategy,
        eval_steps=int(eval_steps) if eval_steps is not None else None,
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 1)),
        num_generations_eval=int(cfg.get("num_generations_eval", 4)),

        # generation settings for policy sampling
        max_prompt_length=cfg["max_prompt_length"],
        max_completion_length=cfg["max_completion_length"],
        temperature=cfg.get("temperature", 0.7),
        top_p=cfg.get("top_p", 1.0),
        num_generations=cfg["num_generations"],

        # GRPO specifics
        beta=float(args.beta),
        loss_type=cfg.get("loss_type", "grpo"),

        # vLLM off by default
        use_vllm=use_vllm,

        # deepspeed config path if any
        deepspeed=cfg.get("deepspeed_config") or None,
    )

    # Reward
    reward_fn = make_countdown_reward_fn(cfg)

    # W&B
    if cfg.get("entity"):
        os.environ["WANDB_ENTITY"] = str(cfg["entity"])
    os.environ["WANDB_PROJECT"] = str(cfg["project"])

    # Trainer
    trainer = GRPOTrainer(
        model=cfg["model_name"],
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    # Train
    trainer.train()
    trainer.save_model()
    print("Training complete. Saved to:", grpo_args.output_dir)


if __name__ == "__main__":
    main()
