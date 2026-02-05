#!/usr/bin/env python3
"""
Checkpoint Reconstruction from ES Replay Logs

This script reconstructs model checkpoints from compact replay logs instead of 
storing full model weights. It uses the deterministic nature of the ES algorithm:
given the same seeds and update coefficients, we can reproduce identical updates.

Usage:
    python reconstruct_checkpoint.py \
        --replay_log_dir /path/to/replay_logs \
        --target_iteration 100 \
        --output_path /path/to/reconstructed_model.pth

Verification:
    To verify correctness, reconstruct a checkpoint that was also saved as a full
    HF checkpoint, then compare the weights:
    
    python reconstruct_checkpoint.py --replay_log_dir ... --target_iteration 200 --output_path reconstructed.pth
    # Then compare with the full checkpoint saved at iteration 200
"""

import argparse
import json
import os
import time
from typing import Optional, Tuple, List

import torch
from transformers import AutoModelForCausalLM


def load_replay_metadata(replay_dir: str) -> dict:
    """Load the one-time metadata from replay log."""
    meta_path = os.path.join(replay_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    
    with open(meta_path, "r") as f:
        return json.load(f)


def load_iteration_logs(replay_dir: str, start_iter: int = 0, end_iter: Optional[int] = None) -> List[dict]:
    """Load iteration logs from the JSONL file within the specified range."""
    log_path = os.path.join(replay_dir, "iteration_logs.jsonl")
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Iteration logs file not found: {log_path}")
    
    logs = []
    with open(log_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            iter_num = entry["iteration"]
            if iter_num >= start_iter and (end_iter is None or iter_num < end_iter):
                logs.append(entry)
    
    return logs


def load_full_checkpoint_markers(replay_dir: str) -> List[dict]:
    """Load the list of full checkpoint markers."""
    markers_path = os.path.join(replay_dir, "full_checkpoints.json")
    if not os.path.exists(markers_path):
        return []
    
    with open(markers_path, "r") as f:
        return json.load(f)


def find_nearest_checkpoint(markers: List[dict], target_iteration: int) -> Optional[Tuple[int, str]]:
    """Find the nearest full checkpoint at or before target_iteration."""
    valid_markers = [m for m in markers if m["iteration"] <= target_iteration]
    if not valid_markers:
        return None
    
    valid_markers.sort(key=lambda x: x["iteration"], reverse=True)
    nearest = valid_markers[0]
    return nearest["iteration"], nearest["checkpoint_path"]


def apply_perturbation(model: torch.nn.Module, seed: int, scale: float, device: str = "cpu"):
    """Apply a single perturbation to model parameters using the seed."""
    for name, param in model.named_parameters():
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        noise = torch.randn(param.shape, dtype=param.dtype, device=device, generator=gen)
        param.data.add_(scale * noise)
        del noise


def replay_iteration(model: torch.nn.Module, seeds: List[int], update_coeffs: List[float], 
                     device: str = "cpu"):
    """Replay a single iteration's updates."""
    for seed, coeff in zip(seeds, update_coeffs):
        apply_perturbation(model, seed, coeff, device)


def reconstruct_checkpoint(
    replay_dir: str,
    target_iteration: int,
    output_path: Optional[str] = None,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[torch.nn.Module, dict]:
    """
    Reconstruct a checkpoint from replay logs.
    
    Args:
        replay_dir: Path to the replay_logs directory
        target_iteration: Iteration number to reconstruct to
        output_path: Optional path to save the reconstructed weights
        device: Device to use for computation
        verbose: Whether to print progress
    
    Returns:
        Tuple of (reconstructed model, reconstruction stats)
    """
    start_time = time.time()
    
    # Load metadata
    if verbose:
        print(f"Loading replay log from: {replay_dir}")
    metadata = load_replay_metadata(replay_dir)
    base_model_path = metadata["base_model_path"]
    
    if verbose:
        print(f"   Base model: {base_model_path}")
        print(f"   Target iteration: {target_iteration}")
    
    # Check for full checkpoints to start from
    markers = load_full_checkpoint_markers(replay_dir)
    start_from_iter = 0
    checkpoint_path = None
    
    if markers:
        result = find_nearest_checkpoint(markers, target_iteration)
        if result:
            start_from_iter, checkpoint_path = result
            if verbose:
                print(f"   Found full checkpoint at iteration {start_from_iter}")
    
    # Load base model or checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path) and start_from_iter > 0:
        if verbose:
            print(f"Loading checkpoint from: {checkpoint_path}")
        
        # Load from HF checkpoint directory
        if os.path.isdir(checkpoint_path):
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path, 
                torch_dtype=torch.float16
            ).to(device)
        else:
            # Load as .pth file
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path, 
                torch_dtype=torch.float16
            ).to(device)
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict, strict=True)
            del state_dict
        
        if verbose:
            print(f"   Will replay from iteration {start_from_iter} to {target_iteration}")
    else:
        if verbose:
            print(f"Loading base model from: {base_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path, 
            torch_dtype=torch.float16
        ).to(device)
        start_from_iter = 0
        if verbose:
            print(f"   Will replay from iteration 0 to {target_iteration}")
    
    # Load iteration logs for the range we need to replay
    if verbose:
        print(f"Loading iteration logs [{start_from_iter} -> {target_iteration}]...")
    
    logs = load_iteration_logs(replay_dir, start_from_iter, target_iteration)
    
    if verbose:
        print(f"   Found {len(logs)} iterations to replay")
    
    # Replay iterations
    if verbose:
        print(f"Replaying updates...")
    
    replayed_count = 0
    for log_entry in logs:
        seeds = log_entry["seeds"]
        update_coeffs = log_entry["update_coeffs"]
        
        replay_iteration(model, seeds, update_coeffs, device=device)
        replayed_count += 1
        
        if verbose and (replayed_count % 10 == 0 or replayed_count == len(logs)):
            print(f"   Replayed {replayed_count}/{len(logs)} iterations")
    
    elapsed = time.time() - start_time
    
    stats = {
        "target_iteration": target_iteration,
        "started_from_iteration": start_from_iter,
        "iterations_replayed": replayed_count,
        "elapsed_seconds": elapsed,
        "base_model_path": base_model_path,
        "checkpoint_used": checkpoint_path,
    }
    
    if verbose:
        print(f"Reconstruction complete! Replayed {replayed_count} iterations in {elapsed:.2f}s")
    
    # Save if output path provided
    if output_path:
        if verbose:
            print(f"Saving reconstructed weights to: {output_path}")
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(state_dict, output_path)
        
        if verbose:
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"   Saved {size_mb:.2f} MB")
    
    return model, stats


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct ES checkpoint from compact replay logs"
    )
    parser.add_argument(
        "--replay_log_dir", 
        type=str, 
        required=True,
        help="Path to the replay_logs directory"
    )
    parser.add_argument(
        "--target_iteration", 
        type=int, 
        required=True,
        help="Target iteration to reconstruct"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default=None,
        help="Path to save reconstructed model weights (.pth)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu",
        help="Device to use (cpu or cuda:X)"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    model, stats = reconstruct_checkpoint(
        replay_dir=args.replay_log_dir,
        target_iteration=args.target_iteration,
        output_path=args.output_path,
        device=args.device,
        verbose=not args.quiet,
    )
    
    print(f"\nReconstruction Stats:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
