"""Build interactive HTML explorer for solve overlap between temperature and perturbation sampling."""

import argparse
import html
import json
import shutil
from pathlib import Path

import numpy as np


def load_results(path: Path) -> dict:
    results_path = path / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")
    with open(results_path) as f:
        return json.load(f)


def merge_temperature_results(result_list: list[dict]) -> dict:
    merged = {"results": {}}
    for results in result_list:
        for key, val in results["results"].items():
            if key.startswith("temp_"):
                merged["results"][key] = val
            elif key == "greedy" and "greedy" not in merged["results"]:
                merged["results"]["greedy"] = val
    return merged


def merge_perturbation_results(result_list: list[dict]) -> dict:
    merged = {"results": {}}
    for results in result_list:
        for key, val in results["results"].items():
            if key.startswith("sigma_"):
                merged["results"][key] = val
            elif key == "greedy_baseline" and "greedy_baseline" not in merged["results"]:
                merged["results"]["greedy_baseline"] = val
    return merged


def build_html(task_datas, temp_results, perturb_results, output_path):
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

    temp_counts = best_temp["per_prompt_correct_counts"]
    sigma_counts = best_sigma["per_prompt_correct_counts"]
    n_samples = int(max(best_temp["pass_at_k"].keys(), key=int))

    # Categorize
    prompts = []
    for i, data in enumerate(task_datas):
        t_solved = temp_counts[i] > 0
        s_solved = sigma_counts[i] > 0
        if t_solved and s_solved:
            cat = "both"
        elif t_solved:
            cat = "temp_only"
        elif s_solved:
            cat = "perturb_only"
        else:
            cat = "neither"
        prompts.append({
            "idx": i,
            "id": data.get("id", i),
            "numbers": data["numbers"],
            "target": data["target"],
            "solution": data.get("solution", "N/A"),
            "temp_correct": temp_counts[i],
            "sigma_correct": sigma_counts[i],
            "category": cat,
        })

    counts = {
        "both": sum(1 for p in prompts if p["category"] == "both"),
        "temp_only": sum(1 for p in prompts if p["category"] == "temp_only"),
        "perturb_only": sum(1 for p in prompts if p["category"] == "perturb_only"),
        "neither": sum(1 for p in prompts if p["category"] == "neither"),
    }

    prompts_json = json.dumps(prompts)
    total = len(task_datas)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Solve Overlap Explorer — {temp_label} vs {sigma_label}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; color: #333; }}

.header {{
    background: #1a1a2e; color: white; padding: 14px 24px;
    display: flex; align-items: center; gap: 24px;
}}
.header h1 {{ font-size: 18px; font-weight: 600; }}
.header .stats {{ font-size: 13px; opacity: 0.7; }}

.container {{ display: flex; height: calc(100vh - 52px); }}

/* Left: category buttons stacked, height proportional to count */
.cat-buttons {{
    width: 180px; min-width: 180px;
    display: flex; flex-direction: column;
    border-right: 1px solid #ddd;
}}
.cat-btn {{
    cursor: pointer;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    border: none; outline: none;
    font-size: 15px; font-weight: 600;
    color: white;
    transition: filter 0.15s, opacity 0.15s;
    opacity: 0.75;
    gap: 2px;
}}
.cat-btn:hover {{ filter: brightness(1.1); opacity: 0.9; }}
.cat-btn.active {{ opacity: 1; filter: brightness(1); box-shadow: inset 0 0 0 3px rgba(255,255,255,0.4); }}
.cat-btn .cat-count {{ font-size: 28px; font-weight: 800; }}
.cat-btn .cat-label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9; }}
.cat-btn[data-cat="both"] {{ background: #7B68EE; }}
.cat-btn[data-cat="temp_only"] {{ background: #e67e22; }}
.cat-btn[data-cat="perturb_only"] {{ background: #3498db; }}
.cat-btn[data-cat="neither"] {{ background: #aaa; }}

/* Right: scrollable list */
.list-panel {{
    flex: 1; overflow-y: auto; padding: 16px 24px;
}}
.list-panel h2 {{
    font-size: 15px; color: #888; margin-bottom: 12px;
    text-transform: uppercase; letter-spacing: 0.5px;
}}

.prompt-card {{
    background: white; border-radius: 10px; padding: 14px 18px;
    margin-bottom: 10px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06);
    display: flex; align-items: center; gap: 18px;
}}
.prompt-card .problem {{
    flex: 1; min-width: 0;
}}
.prompt-card .problem .numbers {{
    font-size: 17px; font-weight: 700; color: #1a1a2e;
}}
.prompt-card .problem .target {{
    font-size: 15px; color: #7B68EE; font-weight: 600; margin-left: 6px;
}}
.prompt-card .problem .solution {{
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 13px; color: #666; margin-top: 3px;
}}
.prompt-card .counts {{
    display: flex; flex-direction: column; gap: 4px;
    min-width: 150px; font-size: 13px;
}}
.prompt-card .counts .bar-row {{
    display: flex; align-items: center; gap: 6px;
}}
.prompt-card .counts .bar-label {{
    width: 50px; text-align: right; font-weight: 600; font-size: 11px;
}}
.prompt-card .counts .bar-label.temp {{ color: #e67e22; }}
.prompt-card .counts .bar-label.sigma {{ color: #3498db; }}
.bar-track {{
    flex: 1; background: #eee; border-radius: 4px; height: 16px;
    overflow: hidden; position: relative;
}}
.bar-fill {{
    height: 100%; border-radius: 4px;
    transition: width 0.3s;
}}
.bar-fill.temp {{ background: #e67e22; }}
.bar-fill.sigma {{ background: #3498db; }}
.bar-num {{
    font-size: 11px; color: #666; width: 28px;
}}
</style>
</head>
<body>

<div class="header">
    <h1>Solve Overlap Explorer</h1>
    <div class="stats">{temp_label} vs {sigma_label} &middot; {n_samples} samples/perturbations &middot; {total} prompts</div>
</div>

<div class="container">
    <div class="cat-buttons">
        <div class="cat-btn active" data-cat="both" style="flex: {counts['both']}" onclick="filterCat('both')">
            <span class="cat-count">{counts['both']}</span>
            <span class="cat-label">Both</span>
        </div>
        <div class="cat-btn" data-cat="temp_only" style="flex: {counts['temp_only']}" onclick="filterCat('temp_only')">
            <span class="cat-count">{counts['temp_only']}</span>
            <span class="cat-label">Temp only</span>
        </div>
        <div class="cat-btn" data-cat="perturb_only" style="flex: {counts['perturb_only']}" onclick="filterCat('perturb_only')">
            <span class="cat-count">{counts['perturb_only']}</span>
            <span class="cat-label">Perturb only</span>
        </div>
        <div class="cat-btn" data-cat="neither" style="flex: {counts['neither']}" onclick="filterCat('neither')">
            <span class="cat-count">{counts['neither']}</span>
            <span class="cat-label">Neither</span>
        </div>
    </div>
    <div class="list-panel" id="list-panel"></div>
</div>

<script>
const prompts = {prompts_json};
const nSamples = {n_samples};
const tempLabel = "{temp_label}";
const sigmaLabel = "{sigma_label}";

const catTitles = {{
    both: "Solved by both temperature and perturbation",
    temp_only: "Solved by temperature only",
    perturb_only: "Solved by perturbation only",
    neither: "Not solved by either method"
}};

let currentCat = "both";

function filterCat(cat) {{
    currentCat = cat;
    document.querySelectorAll('.cat-btn').forEach(b => b.classList.toggle('active', b.dataset.cat === cat));

    const panel = document.getElementById('list-panel');
    const filtered = prompts.filter(p => p.category === cat);
    filtered.sort((a, b) => (b.temp_correct + b.sigma_correct) - (a.temp_correct + a.sigma_correct));

    const maxCorrect = Math.max(1, ...filtered.map(p => Math.max(p.temp_correct, p.sigma_correct)));

    panel.innerHTML = `<h2>${{catTitles[cat]}} (${{filtered.length}})</h2>` +
        filtered.map(p => {{
            const tPct = (p.temp_correct / maxCorrect * 100);
            const sPct = (p.sigma_correct / maxCorrect * 100);
            return `
            <div class="prompt-card">
                <div class="problem">
                    <span class="numbers">[${{p.numbers.join(', ')}}]</span>
                    <span class="target">= ${{p.target}}</span>
                    <div class="solution">${{p.solution}}</div>
                </div>
                <div class="counts">
                    <div class="bar-row">
                        <span class="bar-label temp">temp</span>
                        <div class="bar-track"><div class="bar-fill temp" style="width: ${{tPct}}%"></div></div>
                        <span class="bar-num">${{p.temp_correct}}</span>
                    </div>
                    <div class="bar-row">
                        <span class="bar-label sigma">perturb</span>
                        <div class="bar-track"><div class="bar-fill sigma" style="width: ${{sPct}}%"></div></div>
                        <span class="bar-num">${{p.sigma_correct}}</span>
                    </div>
                </div>
            </div>`;
        }}).join('');

    panel.scrollTop = 0;
}}

filterCat('both');
</script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html_content)
    print(f"  Saved {output_path}")


def main(config_path: str, overwrite: bool = False):
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if "temperature_upstream_dirs" not in config:
        raise ValueError("FATAL: 'temperature_upstream_dirs' (list) required")
    if "perturbation_upstream_dirs" not in config:
        raise ValueError("FATAL: 'perturbation_upstream_dirs' (list) required")
    if "data_path" not in config:
        raise ValueError("FATAL: 'data_path' required")
    if "output_dir" not in config:
        raise ValueError("FATAL: 'output_dir' required")

    output_dir = Path(config["output_dir"])
    if output_dir.exists() and not overwrite:
        raise ValueError(f"Output {output_dir} exists. Use --overwrite to replace.")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")

    # Load task data
    data_cfg = config["data_path"]
    offset = config.get("data_offset", 200)
    num_samples = config.get("data_num_samples", 128)
    with open(data_cfg) as f:
        dataset = json.load(f)
    task_datas = dataset[offset:offset + num_samples]

    # Load results
    temp_dirs = [Path(d) for d in config["temperature_upstream_dirs"]]
    perturb_dirs = [Path(d) for d in config["perturbation_upstream_dirs"]]

    for d in temp_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Temperature upstream not found: {d}")
    for d in perturb_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Perturbation upstream not found: {d}")

    temp_results_list = [load_results(d) for d in temp_dirs]
    temp_results = merge_temperature_results(temp_results_list)
    perturb_results_list = [load_results(d) for d in perturb_dirs]
    perturb_results = merge_perturbation_results(perturb_results_list)

    print("Building solve explorer HTML...")
    build_html(task_datas, temp_results, perturb_results, output_dir / "index.html")
    print(f"Done. Open {output_dir / 'index.html'} in a browser.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite)
