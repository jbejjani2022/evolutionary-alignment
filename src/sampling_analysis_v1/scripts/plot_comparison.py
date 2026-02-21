"""Downstream plots comparing temperature sampling vs perturbation sampling."""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: Path) -> dict:
    results_path = path / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")
    with open(results_path) as f:
        return json.load(f)


def merge_temperature_results(result_list: list[dict]) -> dict:
    """Merge results from multiple temperature sampling runs into one dict."""
    merged = {"results": {}}
    for results in result_list:
        for key, val in results["results"].items():
            if key.startswith("temp_"):
                merged["results"][key] = val
            elif key == "greedy" and "greedy" not in merged["results"]:
                merged["results"]["greedy"] = val
    return merged


def plot_pass_at_k_comparison(temp_results: dict, perturb_results: dict, output_dir: Path):
    """pass@k curves: temperature (solid) vs perturbation (dashed) on same axes."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Temperature curves
    temps = sorted(
        [k for k in temp_results["results"] if k.startswith("temp_")],
        key=lambda x: float(x.split("_")[1]),
    )
    cmap_temp = plt.cm.Oranges
    temp_colors = [cmap_temp(0.3 + 0.6 * i / max(len(temps) - 1, 1)) for i in range(len(temps))]

    for color, key in zip(temp_colors, temps):
        r = temp_results["results"][key]
        pak = r["pass_at_k"]
        ks = sorted(int(k) for k in pak)
        vals = [pak[str(k)] for k in ks]
        ax.plot(ks, vals, "o-", color=color, label=f"T={r['temperature']}", markersize=4)

    # Perturbation curves
    sigmas = sorted(
        [k for k in perturb_results["results"] if k.startswith("sigma_")],
        key=lambda x: float(x.split("_", 1)[1]),
    )
    cmap_perturb = plt.cm.Blues
    perturb_colors = [cmap_perturb(0.4 + 0.5 * i / max(len(sigmas) - 1, 1)) for i in range(len(sigmas))]

    for color, key in zip(perturb_colors, sigmas):
        r = perturb_results["results"][key]
        pak = r["pass_at_k"]
        ks = sorted(int(k) for k in pak)
        vals = [pak[str(k)] for k in ks]
        ax.plot(ks, vals, "s--", color=color, label=f"σ={r['sigma']}", markersize=4)

    # Greedy baseline
    greedy_acc = temp_results["results"]["greedy"]["accuracy"]
    ax.axhline(greedy_acc, color="gray", linestyle=":", linewidth=1, label=f"Greedy ({greedy_acc:.1%})")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("k (number of samples / perturbations)")
    ax.set_ylabel("pass@k")
    ax.set_title("Temperature vs Perturbation Sampling — pass@k\n(Qwen-2.5-0.5B-Instruct, Countdown, 128 problems)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "pass_at_k_comparison.png", dpi=200)
    plt.close(fig)
    print("  Saved pass_at_k_comparison.png")


def plot_best_comparison(temp_results: dict, perturb_results: dict, output_dir: Path):
    """Bar chart comparing best temperature vs best perturbation at key k values."""
    # Find best temperature (by pass@512)
    temps = [k for k in temp_results["results"] if k.startswith("temp_")]
    best_temp_key = max(temps, key=lambda k: temp_results["results"][k]["pass_at_k"]["512"])
    best_temp = temp_results["results"][best_temp_key]

    # Find best sigma (by pass@512)
    sigmas = [k for k in perturb_results["results"] if k.startswith("sigma_")]
    best_sigma_key = max(sigmas, key=lambda k: perturb_results["results"][k]["pass_at_k"]["512"])
    best_sigma = perturb_results["results"][best_sigma_key]

    ks = ["1", "4", "16", "64", "256", "512"]
    temp_vals = [best_temp["pass_at_k"][k] for k in ks]
    sigma_vals = [best_sigma["pass_at_k"][k] for k in ks]

    x = np.arange(len(ks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, temp_vals, width, label=f"Temperature (T={best_temp['temperature']})", color="tab:orange")
    ax.bar(x + width / 2, sigma_vals, width, label=f"Perturbation (σ={best_sigma['sigma']})", color="tab:blue")

    ax.set_xticks(x)
    ax.set_xticklabels([f"pass@{k}" for k in ks])
    ax.set_ylabel("Rate")
    ax.set_title("Best Temperature vs Best Perturbation")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "best_comparison.png", dpi=200)
    plt.close(fig)
    print("  Saved best_comparison.png")


def plot_coverage_vs_depth(temp_results: dict, perturb_results: dict, output_dir: Path):
    """Scatter: coverage (fraction of prompts solved) vs depth (avg correct rate among solved prompts)."""
    fig, ax = plt.subplots(figsize=(6, 5))

    def _get_coverage_depth(r):
        counts = r["per_prompt_correct_counts"]
        n_prompts = len(counts)
        n_samples = int(max(r["pass_at_k"].keys(), key=int))
        solved_counts = [c for c in counts if c > 0]
        coverage = len(solved_counts) / n_prompts
        depth = np.mean(solved_counts) / n_samples if solved_counts else 0.0
        return coverage, depth

    # Temperature points
    for key in sorted(temp_results["results"]):
        if not key.startswith("temp_"):
            continue
        r = temp_results["results"][key]
        coverage, depth = _get_coverage_depth(r)
        ax.scatter(coverage, depth, marker="o", s=80, color="tab:orange", zorder=3)
        ax.annotate(f"T={r['temperature']}", (coverage, depth),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    # Perturbation points
    for key in sorted(perturb_results["results"]):
        if not key.startswith("sigma_"):
            continue
        r = perturb_results["results"][key]
        coverage, depth = _get_coverage_depth(r)
        ax.scatter(coverage, depth, marker="s", s=80, color="tab:blue", zorder=3)
        ax.annotate(f"σ={r['sigma']}", (coverage, depth),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    # Legend entries
    ax.scatter([], [], marker="o", color="tab:orange", s=80, label="Temperature")
    ax.scatter([], [], marker="s", color="tab:blue", s=80, label="Perturbation")

    ax.set_xlabel("Coverage (fraction of prompts with any correct)")
    ax.set_ylabel("Depth (avg correct rate among solved prompts)")
    ax.set_title("Coverage (breadth) vs Depth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "coverage_vs_depth.png", dpi=200)
    plt.close(fig)
    print("  Saved coverage_vs_depth.png")


def plot_solve_overlap(temp_results: dict, perturb_results: dict, output_dir: Path):
    """Stacked bar showing per-prompt solve overlap between best temp and best perturbation."""
    # Best temperature
    temps = [k for k in temp_results["results"] if k.startswith("temp_")]
    best_temp_key = max(temps, key=lambda k: temp_results["results"][k]["pass_at_k"]["512"])
    best_temp = temp_results["results"][best_temp_key]
    temp_label = f"T={best_temp['temperature']}"

    # Best sigma
    sigmas = [k for k in perturb_results["results"] if k.startswith("sigma_")]
    best_sigma_key = max(sigmas, key=lambda k: perturb_results["results"][k]["pass_at_k"]["512"])
    best_sigma = perturb_results["results"][best_sigma_key]
    sigma_label = f"σ={best_sigma['sigma']}"

    temp_solved = np.array(best_temp["per_prompt_correct_counts"]) > 0
    sigma_solved = np.array(best_sigma["per_prompt_correct_counts"]) > 0
    n = len(temp_solved)

    both = int(np.sum(temp_solved & sigma_solved))
    temp_only = int(np.sum(temp_solved & ~sigma_solved))
    sigma_only = int(np.sum(~temp_solved & sigma_solved))
    neither = int(np.sum(~temp_solved & ~sigma_solved))

    # --- Horizontal stacked bar ---
    fig, ax = plt.subplots(figsize=(8, 3))

    categories = [
        (both, f"Both ({both})", "#7B68EE"),
        (temp_only, f"Temp only ({temp_only})", "tab:orange"),
        (sigma_only, f"Perturb only ({sigma_only})", "tab:blue"),
        (neither, f"Neither ({neither})", "#D3D3D3"),
    ]

    left = 0
    for width, label, color in categories:
        ax.barh(0, width, left=left, height=0.5, color=color, edgecolor="white", linewidth=1, label=label)
        if width > 3:
            ax.text(left + width / 2, 0, str(width), ha="center", va="center", fontweight="bold", fontsize=12)
        left += width

    ax.set_xlim(0, n)
    ax.set_yticks([])
    ax.set_xlabel(f"Prompts (n={n})")
    ax.set_title(f"Solve Overlap @ 512 samples — {temp_label} vs {sigma_label}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "solve_overlap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved solve_overlap.png")

    # --- Per-prompt detail strip ---
    # Sort prompts: both > temp_only > sigma_only > neither, then by temp correct count
    temp_counts = np.array(best_temp["per_prompt_correct_counts"])
    sigma_counts = np.array(best_sigma["per_prompt_correct_counts"])

    category = np.where(
        temp_solved & sigma_solved, 0,
        np.where(temp_solved & ~sigma_solved, 1,
                 np.where(~temp_solved & sigma_solved, 2, 3))
    )
    sort_idx = np.lexsort((-temp_counts, -sigma_counts, category))

    fig, axes = plt.subplots(3, 1, figsize=(10, 3.5), gridspec_kw={"height_ratios": [1, 1, 0.4]}, sharex=True)

    # Temp correct counts
    axes[0].bar(range(n), temp_counts[sort_idx], width=1, color="tab:orange", edgecolor="none")
    axes[0].set_ylabel(f"# correct\n({temp_label})")
    axes[0].set_yticks([0, max(temp_counts) // 2, max(temp_counts)])

    # Sigma correct counts
    axes[1].bar(range(n), sigma_counts[sort_idx], width=1, color="tab:blue", edgecolor="none")
    axes[1].set_ylabel(f"# correct\n({sigma_label})")
    axes[1].set_yticks([0, max(max(sigma_counts), 1) // 2, max(max(sigma_counts), 1)])

    # Category strip
    cat_colors = ["#7B68EE", "tab:orange", "tab:blue", "#D3D3D3"]
    cat_labels = ["Both", "Temp only", "Perturb only", "Neither"]
    strip_colors = [cat_colors[c] for c in category[sort_idx]]
    axes[2].bar(range(n), [1] * n, width=1, color=strip_colors, edgecolor="none")
    axes[2].set_yticks([])
    axes[2].set_xlabel(f"Prompts (sorted by category, n={n})")

    # Legend for strip
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=f"{l} ({int(np.sum(category == i))})") for i, (c, l) in enumerate(zip(cat_colors, cat_labels))]
    axes[2].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.6), ncol=4, fontsize=8)

    fig.suptitle(f"Per-Prompt Solve Detail @ 512 — {temp_label} vs {sigma_label}", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "solve_detail.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved solve_detail.png")


def main(config_path: str, overwrite: bool = False):
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if "temperature_upstream_dirs" not in config:
        raise ValueError("FATAL: 'temperature_upstream_dirs' (list) required")
    if "perturbation_upstream_dir" not in config:
        raise ValueError("FATAL: 'perturbation_upstream_dir' required")
    if "output_dir" not in config:
        raise ValueError("FATAL: 'output_dir' required")

    temp_dirs = [Path(d) for d in config["temperature_upstream_dirs"]]
    perturb_dir = Path(config["perturbation_upstream_dir"])
    output_dir = Path(config["output_dir"])

    for d in temp_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Temperature upstream not found: {d}")
    if not perturb_dir.exists():
        raise FileNotFoundError(f"Perturbation upstream not found: {perturb_dir}")

    if output_dir.exists() and not overwrite:
        raise ValueError(f"Output {output_dir} exists. Use --overwrite to replace.")

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")

    temp_results_list = [load_results(d) for d in temp_dirs]
    temp_results = merge_temperature_results(temp_results_list)
    perturb_results = load_results(perturb_dir)

    print(f"Merged {len(temp_dirs)} temperature runs: "
          f"{sum(1 for k in temp_results['results'] if k.startswith('temp_'))} temperatures total")

    print("Generating comparison plots...")
    plot_pass_at_k_comparison(temp_results, perturb_results, output_dir)
    plot_best_comparison(temp_results, perturb_results, output_dir)
    plot_coverage_vs_depth(temp_results, perturb_results, output_dir)
    plot_solve_overlap(temp_results, perturb_results, output_dir)
    print(f"All plots saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite)
