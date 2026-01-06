#!/usr/bin/env python
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
            line=line.strip()
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

    train_rows = [r for r in read_jsonl_rows(cfg["train_jsonl"])]
    eval_rows  = [r for r in read_jsonl_rows(cfg["eval_jsonl"])] if (cfg.get("eval_jsonl") and os.path.exists(cfg["eval_jsonl"])) else []

    def format_prompt(question):
        """Apply chat template to raw question."""
        messages = [{"role": "user", "content": question}]
        formatted = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return formatted

    train_prompts = [{"prompt": format_prompt(str(r[pkey]).strip()), 
                      "answer": str(r[skey]).strip()} for r in train_rows]
    eval_prompts  = [{"prompt": format_prompt(str(r[pkey]).strip()), 
                      "answer": str(r[skey]).strip()} for r in eval_rows]

    # Debug: show sample
    print(f"=== Sample formatted prompt ===\n{train_prompts[0]['prompt']}\n=== End ===")
    
    d_train = Dataset.from_list(train_prompts)
    d_eval  = Dataset.from_list(eval_prompts) if eval_prompts else None

    return d_train, d_eval


def make_reward_func_from_map(cfg: dict):
    """
    Paper reward (training-time):
      R = -|len(y) - len(s_k)|   (length in characters), s_k is the gold answer.
    """

    G = int(cfg["num_generations"])

    def reward_fn(completions, **kwargs):
        # TRL passes dataset columns via **kwargs; answers arrive under "answer".
        answers = kwargs.get("answer", None)

        rewards = []
        decoded_len = []
        for completion, answer in zip(completions, answers):
            # print("completion:", completion)
            text = completion if isinstance(completion, str) else str(completion)

            y = text.strip()
            a = (answer or "").strip()
            r = -abs(len(y) - len(a))
            # print("y:", y, "answer:", answer, "r:", r)
            rewards.append(float(r))
            decoded_len.append(len(y))
        # optional wandb lightweight logging on rank 0
        try:
            import wandb, os
            if os.environ.get("RANK", "0") == "0" and len(rewards) > 0:
                wandb.log({
                    "train/decoded_length/mean": float(sum(decoded_len)) / len(decoded_len),
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
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg.setdefault("project", "es_conciseness")
    cfg.setdefault("entity", None)

    import torch
    import random
    import numpy as np
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    # Dataset
    train_ds, eval_ds = format_train_eval_datasets(cfg, tokenizer)
    
    # Tokenizer - match ES settings that work
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # GRPO args
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

        # generation settings for policy sampling
        max_prompt_length=cfg["max_prompt_length"],
        max_completion_length=cfg["max_completion_length"],
        temperature=cfg.get("temperature", 1.0),
        top_p=cfg.get("top_p", 1.0),
        num_generations=cfg["num_generations"],

        # GRPO specifics
        beta=float(args.beta),
        loss_type=cfg.get("loss_type", "grpo"),

        # vLLM off by default
        use_vllm=use_vllm,

        # Let accelerate handle deepspeed config via --deepspeed_config_file
        deepspeed=None,
        
        # Repro
        seed=args.seed,
        data_seed=args.seed,
        
        # Eval during training
        do_eval=eval_ds is not None,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=int(cfg.get("eval_steps", 50)) if eval_ds is not None else None,
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 1)),

        # Use fewer generations for eval to keep it cheap (must be compatible with eval batch sizing)
        num_generations_eval=int(cfg.get("num_generations_eval", 4))
    )

    # Reward (training-time)
    reward_fn = make_reward_func_from_map(cfg)
    
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
    
    # Log some eval completions for sanity checking
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

    # Train
    trainer.train()
    trainer.save_model()
    print("Training complete. Saved to:", grpo_args.output_dir)

if __name__ == "__main__":
    main()
