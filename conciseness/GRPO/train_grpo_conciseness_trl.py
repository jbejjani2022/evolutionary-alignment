#!/usr/bin/env python
"""
GRPO training for Conciseness task.
TRL 0.26.2 compatible.
"""
import os
import sys
import json
import argparse


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


def read_jsonl_rows(path: str):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def format_train_eval_datasets(cfg: dict, tokenizer):
    """
    Return (train_dataset, eval_dataset_or_None) with prompts formatted for TRL GRPO.
    Applies chat template for instruction-tuned models.
    """
    from datasets import Dataset
    pkey = cfg.get("prompt_key", "prompt")
    skey = cfg.get("solution_key", "answer")

    train_rows = list(read_jsonl_rows(cfg["train_jsonl"]))
    eval_rows = list(read_jsonl_rows(cfg["eval_jsonl"])) if (
        cfg.get("eval_jsonl") and os.path.exists(cfg["eval_jsonl"])
    ) else []

    def format_prompt(question):
        """Apply chat template to raw question."""
        messages = [{"role": "user", "content": question}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted

    train_prompts = [
        {"prompt": format_prompt(str(r[pkey]).strip()), "answer": str(r[skey]).strip()}
        for r in train_rows
    ]
    eval_prompts = [
        {"prompt": format_prompt(str(r[pkey]).strip()), "answer": str(r[skey]).strip()}
        for r in eval_rows
    ]

    d_train = Dataset.from_list(train_prompts)
    d_eval = Dataset.from_list(eval_prompts) if eval_prompts else None

    return d_train, d_eval


def make_reward_func_from_map(cfg: dict, tokenizer):
    """
    Paper reward (training-time):
      R = -|len(y) - len(s_k)|   (length in characters), s_k is the gold answer.
    
    TRL passes completions as strings (assistant response only, no prompt).
    We strip any trailing <|im_end|> or EOS artifacts before measuring length.
    """
    # Get special tokens to strip from completions
    eos_token = tokenizer.eos_token or ""
    # Qwen2.5 uses <|im_end|> as the chat end token
    im_end_token = "<|im_end|>"
    
    def clean_completion(text: str) -> str:
        """Remove EOS/chat-end tokens and strip whitespace."""
        text = text.strip()
        # Remove trailing special tokens (may appear if generation stopped naturally)
        for token in [im_end_token, eos_token]:
            if token and text.endswith(token):
                text = text[:-len(token)].strip()
        return text

    def reward_fn(completions, **kwargs):
        # TRL passes dataset columns via **kwargs; answers arrive under "answer".
        answers = kwargs.get("answer", None)
        
        if answers is None:
            raise ValueError("No 'answer' column found in kwargs. Check dataset columns.")

        rewards = []
        decoded_len = []
        
        for completion, answer in zip(completions, answers):
            text = completion if isinstance(completion, str) else str(completion)
            
            # Clean the completion (remove special tokens)
            y = clean_completion(text)
            a = (answer or "").strip()
            
            # Reward: negative absolute difference in character length
            r = -abs(len(y) - len(a))
            
            rewards.append(float(r))
            decoded_len.append(len(y))

        # W&B logging on rank 0
        try:
            import wandb
            if os.environ.get("RANK", "0") == "0" and len(rewards) > 0:
                wandb.log({
                    "train/decoded_length/mean": sum(decoded_len) / len(decoded_len),
                    "train/decoded_length/min": min(decoded_len),
                    "train/decoded_length/max": max(decoded_len),
                    "train/reward_raw/mean": sum(rewards) / len(rewards),
                }, commit=False)
        except Exception:
            pass
        
        return rewards

    return reward_fn


def verify_setup(tokenizer, train_ds, cfg, args):
    """Print verification info for debugging. Call on rank 0 only."""
    print("\n" + "=" * 60)
    print("SETUP VERIFICATION")
    print("=" * 60)
    
    # Tokenizer checks
    print(f"\n[Tokenizer]")
    print(f"  pad_token: {repr(tokenizer.pad_token)} (id={tokenizer.pad_token_id})")
    print(f"  eos_token: {repr(tokenizer.eos_token)} (id={tokenizer.eos_token_id})")
    print(f"  padding_side: {tokenizer.padding_side}")
    
    # Check for Qwen chat tokens
    im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    print(f"  <|im_end|> token ids: {im_end_id}")
    
    # Sample prompt check
    print(f"\n[Sample Formatted Prompt]")
    sample = train_ds[0]
    print(f"  Prompt:\n    {repr(sample['prompt'][:200])}...")
    print(f"  Answer: {repr(sample['answer'])}")
    
    # Verify chat template applied
    if "<|im_start|>" not in sample['prompt']:
        print("  ⚠️  WARNING: Prompt missing <|im_start|> - chat template may not be applied!")
    else:
        print("  ✓ Chat template detected")
    
    if "<|im_start|>assistant" in sample['prompt']:
        print("  ✓ Generation prompt added (ends with assistant turn)")
    else:
        print("  ⚠️  WARNING: Missing assistant generation prompt!")
    
    # Config summary
    print(f"\n[Training Config]")
    print(f"  beta (KL coef): {args.beta}")
    print(f"  num_generations: {cfg['num_generations']}")
    print(f"  max_completion_length: {cfg['max_completion_length']}")
    print(f"  learning_rate: {cfg['learning_rate']}")
    print(f"  max_steps: {cfg.get('max_steps', 1000)}")
    print(f"  seed: {args.seed}")
    
    print("\n" + "=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg.setdefault("project", "es_conciseness")
    cfg.setdefault("entity", None)

    import torch
    import random
    import numpy as np

    # Repro
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    # W&B setup
    if cfg.get("entity"):
        os.environ["WANDB_ENTITY"] = str(cfg["entity"])
    os.environ["WANDB_PROJECT"] = str(cfg["project"])

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"],
        use_fast=False,
        cache_dir=cfg.get("hf_cache_dir", "hf_cache")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Dataset
    train_ds, eval_ds = format_train_eval_datasets(cfg, tokenizer)

    # Verification (rank 0 only)
    if os.environ.get("RANK", "0") == "0":
        verify_setup(tokenizer, train_ds, cfg, args)

    # GRPO Config
    use_vllm = bool(int(os.environ.get("USE_VLLM", "0")))
    
    grpo_args = GRPOConfig(
        output_dir=os.path.join(cfg["output_dir"], f"beta{args.beta}_seed{args.seed}"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg.get("lr_scheduler_type", "constant"),
        warmup_steps=cfg.get("warmup_steps", 0),
        num_train_epochs=cfg.get("num_train_epochs", 1),
        max_steps=cfg.get("max_steps", 1000),
        logging_steps=cfg.get("logging_steps", 5),
        save_steps=cfg.get("save_steps", 200),
        report_to=["wandb"],
        run_name=f"qwen2.5-7b_grpo_conciseness_beta{args.beta}_seed{args.seed}",
        bf16=True,
        remove_unused_columns=False,

        # Generation settings
        max_prompt_length=cfg["max_prompt_length"],
        max_completion_length=cfg["max_completion_length"],
        temperature=cfg.get("temperature", 1.0),
        top_p=cfg.get("top_p", 1.0),
        num_generations=cfg["num_generations"],

        # GRPO specifics
        beta=float(args.beta),
        loss_type=cfg.get("loss_type", "grpo"),

        # vLLM
        use_vllm=use_vllm,

        # Let accelerate handle deepspeed
        deepspeed=None,

        # Reproducibility
        seed=args.seed,
        data_seed=args.seed,

        # Evaluation
        do_eval=eval_ds is not None,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=int(cfg.get("eval_steps", 50)) if eval_ds is not None else None,
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 1)),
        num_generations_eval=int(cfg.get("num_generations_eval", 4)),
    )

    # Reward function
    reward_fn = make_reward_func_from_map(cfg, tokenizer)

    # Trainer
    trainer = GRPOTrainer(
        model=cfg["model_name"],
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    # Verify KL setup (rank 0)
    if os.environ.get("RANK", "0") == "0":
        print("[KL Regularization Check]")
        if args.beta > 0:
            if hasattr(trainer, 'ref_model') and trainer.ref_model is not None:
                print(f"  ✓ Reference model loaded (beta={args.beta})")
            else:
                # In some TRL versions, ref_model might be created lazily or named differently
                print(f"    Reference model attribute not found directly.")
                print(f"    This may be normal - TRL 0.26.2 handles ref model internally.")
                print(f"    Monitor 'kl' in training logs to verify KL is computed.")
        else:
            print(f"    beta=0.0, KL regularization disabled")

    # Completion logging callback
    if eval_ds is not None and len(eval_ds) > 0:
        from trl import LogCompletionsCallback
        from transformers import GenerationConfig

        completion_log_steps = int(cfg.get("completion_log_steps", cfg.get("eval_steps", 50)))
        completion_log_num_prompts = int(cfg.get("completion_log_num_prompts", min(8, len(eval_ds))))

        gen_cfg = GenerationConfig(
            max_new_tokens=int(cfg["max_completion_length"]),
            do_sample=True,
            temperature=float(cfg.get("temperature", 1.0)),
            top_p=float(cfg.get("top_p", 1.0)),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        trainer.add_callback(
            LogCompletionsCallback(
                trainer=trainer,
                generation_config=gen_cfg,
                num_prompts=completion_log_num_prompts,
                freq=completion_log_steps,
            )
        )
        if os.environ.get("RANK", "0") == "0":
            print(f"[Callbacks]")
            print(f"  ✓ LogCompletionsCallback added (every {completion_log_steps} steps, {completion_log_num_prompts} prompts)")

    # Train
    if os.environ.get("RANK", "0") == "0":
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60 + "\n")

    trainer.train()
    trainer.save_model()
    
    if os.environ.get("RANK", "0") == "0":
        print("\n" + "=" * 60)
        print(f"Training complete. Saved to: {grpo_args.output_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
