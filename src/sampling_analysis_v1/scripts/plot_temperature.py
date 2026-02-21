"""Downstream plots for temperature sampling results."""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(upstream_dir: Path) -> dict:
    results_path = upstream_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")
    with open(results_path) as f:
        return json.load(f)


def plot_pass_at_k(results: dict, output_dir: Path):
    """pass@k curves: one line per temperature."""
    fig, ax = plt.subplots(figsize=(6, 4))

    temps = sorted(
        [k for k in results["results"] if k.startswith("temp_")],
        key=lambda x: float(x.split("_")[1]),
    )

    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(temps) - 1, 1)) for i in range(len(temps))]

    for color, key in zip(colors, temps):
        temp = results["results"][key]["temperature"]
        pak = results["results"][key]["pass_at_k"]
        ks = sorted(int(k) for k in pak)
        vals = [pak[str(k)] for k in ks]
        ax.plot(ks, vals, "o-", color=color, label=f"T={temp}", markersize=4)

    # greedy baseline
    greedy_acc = results["results"]["greedy"]["accuracy"]
    ax.axhline(greedy_acc, color="gray", linestyle="--", linewidth=1, label=f"Greedy ({greedy_acc:.1%})")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("k (number of samples)")
    ax.set_ylabel("pass@k")
    ax.set_title("Temperature Sampling — pass@k on Countdown\n(Qwen-2.5-0.5B-Instruct, 128 problems)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "pass_at_k.png", dpi=200)
    plt.close(fig)
    print(f"  Saved pass_at_k.png")


def plot_correct_distribution(results: dict, output_dir: Path):
    """Histogram of per-prompt correct counts across temperatures."""
    temps = sorted(
        [k for k in results["results"] if k.startswith("temp_")],
        key=lambda x: float(x.split("_")[1]),
    )

    fig, axes = plt.subplots(1, len(temps), figsize=(3.5 * len(temps), 3.5), sharey=True)
    if len(temps) == 1:
        axes = [axes]

    for ax, key in zip(axes, temps):
        temp = results["results"][key]["temperature"]
        counts = results["results"][key]["per_prompt_correct_counts"]
        n_total = results["results"][key]["total_samples"]
        n_zero = sum(1 for c in counts if c == 0)
        n_prompts = len(counts)

        max_count = max(counts) if max(counts) > 0 else 1
        bins = np.arange(-0.5, max_count + 1.5, 1)
        ax.hist(counts, bins=bins, color="steelblue", edgecolor="white", linewidth=0.5)
        ax.set_xlabel("# correct (out of {})".format(n_total))
        ax.set_title(f"T={temp}\n{n_zero}/{n_prompts} unsolved")
        ax.axvline(np.mean(counts), color="red", linestyle="--", linewidth=1, label=f"mean={np.mean(counts):.1f}")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("# prompts")
    fig.suptitle("Per-Prompt Correct Count Distribution", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "correct_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved correct_distribution.png")


def plot_summary_bars(results: dict, output_dir: Path):
    """Bar chart comparing key metrics across temperatures."""
    temps = sorted(
        [k for k in results["results"] if k.startswith("temp_")],
        key=lambda x: float(x.split("_")[1]),
    )

    labels = []
    pass1 = []
    pass64 = []
    pass512 = []
    frac_solved = []

    for key in temps:
        r = results["results"][key]
        labels.append(f"T={r['temperature']}")
        pak = r["pass_at_k"]
        pass1.append(pak["1"])
        pass64.append(pak["64"])
        pass512.append(pak["512"])
        frac_solved.append(r["prompts_with_any_correct"] / len(r["per_prompt_correct_counts"]))

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 1.5 * width, pass1, width, label="pass@1")
    ax.bar(x - 0.5 * width, pass64, width, label="pass@64")
    ax.bar(x + 0.5 * width, pass512, width, label="pass@512")
    ax.bar(x + 1.5 * width, frac_solved, width, label="% prompts solved", color="gray", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate")
    ax.set_title("Temperature Sampling — Summary Metrics")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "summary_bars.png", dpi=200)
    plt.close(fig)
    print(f"  Saved summary_bars.png")


def main(config_path: str, overwrite: bool = False):
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if "upstream_dir" not in config:
        raise ValueError("FATAL: 'upstream_dir' required for downstream work")
    if "output_dir" not in config:
        raise ValueError("FATAL: 'output_dir' required")

    upstream_dir = Path(config["upstream_dir"])
    output_dir = Path(config["output_dir"])

    if not upstream_dir.exists():
        raise FileNotFoundError(f"Upstream dir not found: {upstream_dir}")

    if output_dir.exists() and not overwrite:
        raise ValueError(f"Output {output_dir} exists. Use --overwrite to replace.")

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")

    results = load_results(upstream_dir)
    print("Generating plots...")
    plot_pass_at_k(results, output_dir)
    plot_correct_distribution(results, output_dir)
    plot_summary_bars(results, output_dir)
    print(f"All plots saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite)
