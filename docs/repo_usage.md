# Repository Usage Guide

## Core Development Principles

### 1. Research Integrity First - Fail Fast Philosophy
- **This is a research repo**: Hidden invisible fallbacks are *criminal* to research integrity
- **Failing is fine**: NEVER make code do something secretly without being obvious
- **Stay explicit**: Implicit behavior corrupts experiments and wastes time
- **No silent failures**: Crash immediately on missing configs or invalid states
- **No fallbacks**: Required parameters must be explicitly provided
- **Loud failures**: Better to crash in 1 second than run for 10 hours with wrong behavior
- **Why**: Silent failures waste compute hours and corrupt research results

### 2. Implementation vs Orchestration
- **Implementation (HOW)**: Lives in `src/<track>/` modules - reusable functions and core logic
- **Orchestration (WHAT/WHEN)**: Lives in `src/<track>/scripts/` - experiment flow and coordination at Python level
- **Bash scripts in `/scripts/<track>/`**: Minimal wrappers that just call Python scripts, e.g., `uv run python src/approach_a/scripts/run_experiment.py configs/approach_a/experiment.yaml`

### 3. Abstraction Guidelines
- **Just the right amount**: Do not over-abstract, aim for clarity over cleverness
- **Consult before major changes**: When introducing new structures/abstractions, ask first
- **Watch for spaghetti**: If code is getting tangled, stop and discuss restructuring

### 4. Script Legibility
- Scripts should read like a story - each line is a meaningful step
- Everything happens in `main()` function
- Clear, sequential flow from config loading to execution

## Research Tracks

### Why Tracks?

Research is fundamentally different from product development:

- **Product development**: Linear progression where v1 → v2 → v3, each replacing the last
- **Research**: Parallel exploration where you try approach A, gain an insight, then try completely different approach B, maybe C branches from A's insight but uses B's infrastructure...

Unlike git branches where you eventually merge or abandon, research tracks often:
- Run in parallel indefinitely
- Inform each other without merging
- Have completely different code, not just different parameters
- Need to remain runnable even when you're focused on another track

### What is a Track?

A **track** is a self-contained research direction with its own code, configs, scripts, and outputs:

```
src/<track_name>/           # Track-specific implementation
configs/<track_name>/       # Track-specific configs
scripts/<track_name>/       # Track-specific bash scripts
data/<track_name>/          # Track-specific outputs (gitignored)
docs/tracks/<track_name>/   # Track-specific documentation
```

**Example with two tracks:**
```
src/
├── utils.py                # Cross-track utilities (stable API)
├── approach_a/             # First research direction
│   ├── utils.py            # Track-specific utilities
│   ├── model.py
│   └── scripts/
│       └── run_experiment.py
└── approach_b/             # Second research direction
    ├── utils.py            # Track-specific utilities
    ├── pipeline.py
    └── scripts/
        └── run_analysis.py

configs/
├── approach_a/
│   └── experiment_v1.yaml
└── approach_b/
    └── analysis_config.yaml

scripts/
├── approach_a/
│   └── run_experiment.sh
└── approach_b/
    └── run_analysis.sh

data/
├── approach_a/
│   └── experiment_001/
└── approach_b/
    └── analysis_001/

docs/
├── repo_usage.md           # Cross-track documentation
├── research_context.md     # Overall research context
└── tracks/
    ├── approach_a/         # Track-specific docs
    │   ├── progress.md     # History and progress of the track (REQUIRED)
    │   ├── notes.md        # Tacit knowledge, decisions, insights
    │   └── results.md      # Key findings summary
    └── approach_b/
        ├── progress.md
        └── notes.md
```

### Track Documentation

Each track **must** have a `docs/tracks/<track_name>/` folder with at minimum a `progress.md` file:

- **`progress.md` (REQUIRED)**: The maintained history and progress of the track. This is the single most important file for a track — it tells you where the track has been, where it is now, and where it's going. Without it, returning to a track after time away means starting from scratch. Keep it updated as work progresses.
- **Notes**: Tacit knowledge, design decisions, why certain approaches were chosen/abandoned, observations, gotchas
- **Results summaries**: Key findings without digging through data/

This documentation is invaluable when:
- Returning to a track after working on another
- Explaining the track to collaborators
- Deciding whether to build on or abandon a track

### When to Create a New Track

**Create a new track when:**
- You're exploring a fundamentally different approach or question
- The code structure/abstractions would be significantly different
- You want to preserve a working approach while experimenting with another
- Insights from one direction lead to a completely different implementation

**Do NOT create a new track for:**
- Parameter variations (use different configs instead)
- Bug fixes
- Adding new models/providers to existing infrastructure
- Incremental improvements to existing approach

### Cross-Track Rules

1. **Each track is self-contained** - no imports between tracks
2. **Shared utilities only in `src/utils.py`** - and these must have stable APIs (see below)
3. **Tracks can read each other's data** - for comparison/analysis, but not modify
4. **No track is "primary"** - they're parallel, not hierarchical

## Script Organization

### Required Config Structure
- All configs **MUST** have `output_dir` field - this is non-negotiable
- All scripts should *only* modify the `output_dir` specified in config (except system-level cache/temp files)

### Recommended Script Interface
```bash
python src/scripts/<script_name>.py configs/<config_file>.yaml --overwrite --debug
```
- **Only two runtime flags are recommended:**
  - `--overwrite`: Whether to overwrite existing `output_dir`
  - `--debug`: Debug mode for temporary testing
- Everything else should be in the config file

### Bash Scripts
- All .sh scripts must be in `scripts/<track>/`
- Each track has its own scripts folder: `scripts/approach_a/`, `scripts/approach_b/`, etc.
- All .sh scripts should be as minimal as possible, mostly they simply keep track of what .py scripts should be ran to recreate the experiment
- **IMPORTANT: Always pass through arguments using `"$@"`** to allow flags like `--overwrite` and `--debug` to be passed from bash to Python
- Example:
  ```bash
  #!/bin/bash
  uv run python src/approach_a/scripts/run_experiment.py configs/approach_a/experiment.yaml "$@"
  ```
  This allows you to run: `bash scripts/approach_a/run_experiment.sh --overwrite --debug`

### Config Files
- All configs must be in `configs/<track>/`
- Each track has its own configs folder: `configs/approach_a/`, `configs/approach_b/`, etc.
- Within a track, you can further organize by experiment type if needed:
  - `configs/approach_a/sweeps/`
  - `configs/approach_a/baselines/`
- This organization makes it easy to find and manage configs per track

### Python Scripts
- Orchestration scripts (entry points) live in `src/<track>/scripts/`
- Implementation modules live in `src/<track>/` (but not in `scripts/` subdirectory)
- All orchestration scripts *must* read in *exactly* a `config_path` argument (required)
- Optional flags: `--overwrite` (default False) and `--debug` (default False)
- All orchestration scripts *must* write *all* results to the `output_dir` mentioned by config (error out if this is not given by the config)
- All orchestration scripts should validate the output directory and handle overwrite logic appropriately
- **All scripts should copy the config YAML file to the output directory immediately after creating it** (recommended to save as `config.yaml` for consistency) - this ensures reproducibility and keeps a record of exact parameters used for each run

## Practical Guidelines

### Config Validation
```python
# Always validate upfront - fail fast!
def validate_config(config):
    # Check for required output_dir FIRST
    if 'output_dir' not in config:
        raise ValueError("FATAL: 'output_dir' is required in config")

    # Check other required fields
    required_fields = ['experiment', 'model']  # Add your requirements
    for field in required_fields:
        if field not in config:
            raise ValueError(f"FATAL: '{field}' is required in config")

    # Validate nested fields explicitly
    if 'learning_rate' not in config.get('training', {}):
        raise ValueError("FATAL: 'training.learning_rate' is required")
```

### Error Handling
```python
# ❌ BAD: Silent fallback
if config is None:
    print("Warning: config missing")
    return default_value

# ✅ GOOD: Immediate failure
if config is None:
    raise ValueError("FATAL: config is required")

# ✅ GOOD: Exit immediately on critical failures
import sys
if not Path(data_path).exists():
    print(f"Error: Data path {data_path} does not exist!")
    sys.exit(1)
```

### Output Directory Management
```python
# Use the safe init_directory utility from src/utils.py
# IMPORTANT: Scripts in src/scripts/ need to add project root to path first
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import init_directory

# In your orchestration script
output_dir = init_directory(config['output_dir'], overwrite=args.overwrite)

# Create standard subdirectories
(output_dir / 'figures').mkdir(parents=True, exist_ok=True)
(output_dir / 'results').mkdir(parents=True, exist_ok=True)
(output_dir / 'logs').mkdir(parents=True, exist_ok=True)
```

### Code Organization

**Utilities hierarchy:**
- **`src/utils.py`** (cross-track): Stable, rarely-touched functions used across multiple tracks
  - API must remain stable - changing signatures breaks multiple tracks
  - Examples: `init_directory()`, `load_config()`, generic file I/O helpers
  - Think of these as "infrastructure" - boring but reliable
- **`src/<track>/utils.py`** (track-specific): Utilities specific to one track
  - Can evolve freely without breaking other tracks
  - Examples: track-specific data loaders, formatting functions, domain helpers

**Code placement:**
- **Orchestration scripts**: Put in `src/<track>/scripts/` - entry points that coordinate experiments
- **Implementation modules**: Put in `src/<track>/` - track-specific logic and classes
- Don't over-modularize - functions that do one complete operation are fine
- Trust framework features (e.g., use HuggingFace's TrainingArguments instead of custom schedulers)

### Resources Directory Usage
The `resources/` folder is for external reference materials — code, docs, and implementations you want to look at locally:

- **Git repositories**: Clone external repos you're studying or referencing
- **Documentation**: External docs, API specifications, or references
- **Implementations**: Reference implementations or code samples

For papers, use `literature/` instead (see below).

### Literature Directory (`literature/`)
Research papers converted to markdown for easy reference. Each paper lives in its own folder named `lastname_year_fewwordtitle`:

```
literature/
├── hendel_2023_functionvectors/
│   ├── paper.pdf
│   └── paper/
│       ├── paper.md
│       └── figures/
└── todd_2023_linearreps/
    ├── paper.pdf
    └── paper/
        ├── paper.md
        └── figures/
```

**Adding papers:** Use the `/add-paper` skill with an arXiv URL and folder name:
```
/add-paper https://arxiv.org/abs/2310.15916 hendel_2023_functionvectors
```

This downloads the PDF, converts it to markdown, and extracts figures with AI-generated descriptions.

**Sharing:** The directory uses DVC-style gitignore rules — `.dvc` manifests are committed to git while actual files are tracked via DVC. Use `dvc add literature/<paper_name>` to share papers with collaborators, then `dvc push`.

```bash
# Example structure
resources/
├── repos/
│   └── some-reference-repo/
└── docs/
    └── api-spec.md
```

This folder is gitignored - it's for local reference only, not committed to the repo.

### Memos (`docs/memos/`)
Project-specific tips, accumulated knowledge, and informal notes. Memos capture tacit knowledge that builds up over the course of a project — the kind of thing you'd tell a new collaborator over coffee.

- **Tacit knowledge**: Quirks of the codebase, non-obvious gotchas, "we tried X and it doesn't work because Y"
- **Project-specific tips**: Workflow shortcuts, environment setup tricks, debugging recipes
- **Todo lists**: Informal tracking of ideas, things to try, or deferred tasks
- **Decision rationale**: Why we chose approach A over B, even if it's just a few sentences

Memos are git-tracked and meant to persist. They don't need to be polished — a quick note is better than no note.

### Reports (`docs/reports/`)
Compilations of results written up as mini-narratives. A report is polished enough to share with a collaborator as an interesting, established piece of result — more than raw notes, less than a paper draft.

**IMPORTANT: Never place files directly in `docs/reports/`.** Each report lives in its own subfolder named `<date>_<reportname>`:

```
docs/reports/
├── 2026-02-15_scaling_analysis/
│   ├── report.md
│   └── figures/
└── 2026-02-18_baseline_comparison/
    ├── report.md
    └── figures/
```

- Summarize key findings with context and interpretation
- Include relevant figures and tables
- Should stand on their own — a collaborator can read one without digging through code or data

**Research output hierarchy:** experiments produce raw results in `data/` → interesting results become findings in track notes → established findings get compiled into reports → reports mature into paper sections.

### Papers (`papers/`)
Each subdirectory is an Overleaf-synced git repo for a paper draft:

```
papers/
├── my-icml-paper/       # git clone from Overleaf
└── my-workshop-paper/   # another Overleaf repo
```

The entire `papers/` directory is gitignored (except `.gitkeep`) to avoid dirty submodule issues — each paper has its own git history via Overleaf. Clone them locally as needed. See `docs/paper_writing.md` for writing conventions.

### Data Sharing with DVC

We use [DVC (Data Version Control)](https://dvc.org/) to share data across machines and collaborators. DVC tracks large files via small `.dvc` manifest files committed to git, while actual data lives in a remote storage backend (S3).

**Why DVC:**
- Decoupled from git — no hooks/filters, repo works fine without DVC installed
- Content-addressed storage — deduplication, only changed files re-uploaded
- Selective tracking — `dvc add` only what needs sharing, not all of `data/`
- Version-pinned — each git commit pins exact data versions via `.dvc` manifests

**Setup (one-time per repo):**
```bash
uv add dvc dvc-s3          # Install DVC with S3 backend
dvc init                    # Initialize DVC in repo
dvc remote add -d storage s3://your-bucket-name
```

**Tracking data:**
```bash
# Track a specific directory (creates data/my_experiment.dvc manifest)
dvc add data/my_experiment

# Commit the manifest to git (small text file with hashes)
git add data/my_experiment.dvc data/.gitignore
git commit -m "Track my_experiment data"

# Push actual data to S3
dvc push
```

**Pulling data on another machine:**
```bash
git pull                    # Get .dvc manifests
uv sync                    # Install DVC
dvc pull                    # Download tracked data from S3
```

**Key principles:**
- **Track at the right granularity** — `dvc add` entire experiment directories, not individual files. You can't partially pull a tracked directory.
- **Only track what needs sharing** — training data, checkpoints needed for downstream eval, final results. Don't track intermediate/debug outputs.
- **Never modify tracked data without re-adding** — after changing tracked files, run `dvc add` again to update the manifest.
- **Credentials** — for small teams, shared AWS credentials in `.dvc/config` is fine. For larger teams, use IAM roles.

**`.gitignore` pattern for DVC:**
```
data/**          # ignore all files recursively
!data/**/        # un-ignore directories (so git descends into them)
!data/**/*.dvc   # un-ignore .dvc manifests at any depth
!data/.gitkeep
```

The key trick: `data/**` ignores files AND directories, but `!data/**/` un-ignores just directories. This lets git descend into subdirectories to find `.dvc` files at any nesting depth.

### Scratch Directory Usage
**CRITICAL: NEVER place files directly in `scratch/`!**

- **Always create a subfolder** within `scratch/` for your temporary work: `scratch/{subfolder_name}/`
- **Why this matters**: Direct files in scratch/ create clutter and make it impossible to track what belongs to what task
- **Examples of proper usage:**
  - `scratch/test_visualization/` - for testing plotting code
  - `scratch/debug_model/` - for debugging model behavior
  - `scratch/quick_analysis/` - for one-off data checks
- **Everything in these subfolders is gitignored** (except .gitkeep in scratch/ itself)
- **Clean up subfolders when done** - delete entire subdirectories when the temporary work is complete

```bash
# ✅ CORRECT: Create subfolder for temporary work
mkdir scratch/my_test
touch scratch/my_test/temp_script.py
# ... do your work ...
rm -rf scratch/my_test  # Clean up when done

# ❌ WRONG: Never do this!
touch scratch/temp_script.py  # DO NOT put files directly in scratch/
```

### Running Scripts
Always execute from project root:
```bash
# Via bash script (recommended)
bash scripts/approach_a/run_experiment.sh
bash scripts/approach_a/run_experiment.sh --overwrite --debug

# Direct Python execution
uv run python src/approach_a/scripts/run_experiment.py configs/approach_a/experiment.yaml

# With flags
uv run python src/approach_a/scripts/run_experiment.py configs/approach_a/experiment.yaml --overwrite

# From activated environment
python src/approach_b/scripts/run_analysis.py configs/approach_b/analysis_config.yaml
```

## Quick Decision Guide

**Track decisions:**
- "Is this a fundamentally different approach?" → New track in `src/<new_track>/`
- "Is this a parameter/config variation?" → Same track, new config
- "Is this a bug fix or incremental improvement?" → Same track, same code

**Code placement:**
- "Could ALL tracks reuse this with a stable API?" → `src/utils.py`
- "Could experiments within THIS track reuse this?" → `src/<track>/utils.py`
- "Is this specific to one experiment?" → Orchestration script in `src/<track>/scripts/`
- "Is this HOW to do something?" → Implementation in `src/<track>/` modules
- "Is this WHEN/WHETHER to do something?" → Orchestration in `src/<track>/scripts/`

## Standard Script Template

For a script at `src/<track>/scripts/run_experiment.py`:

```python
import argparse
import yaml
from pathlib import Path

# Cross-track utilities (stable API)
from src.utils import init_directory

# Track-specific utilities (can import from parent track module)
# from src.<track>.utils import some_helper

def main(config_path, overwrite=False, debug=False):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate config - fail fast!
    if 'output_dir' not in config:
        raise ValueError("FATAL: 'output_dir' required in config")

    # Initialize output directory with safety checks
    output_dir = init_directory(config['output_dir'], overwrite=overwrite)

    # Create standard subdirectories
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
    (output_dir / 'results').mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)

    # Debug mode handling
    if debug:
        print(f"DEBUG MODE: Output will be written to {output_dir}")
        # Add any debug-specific behavior

    # Your experiment code here

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to config file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output directory')
    parser.add_argument('--debug', action='store_true', help='Debug mode for testing')
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)
```

**Corresponding bash script** at `scripts/<track>/run_experiment.sh`:
```bash
#!/bin/bash
uv run python src/<track>/scripts/run_experiment.py configs/<track>/experiment.yaml "$@"
```

## Downstream Scripts Pattern

### What is Downstream Work?

Downstream scripts operate on the output of a previous run. This includes:
- **Analysis**: plotting, statistics, summarization
- **Further processing**: transforming or refining outputs
- **Evaluation**: scoring, comparison, validation
- **Data processing**: extracting subsets, reformatting, distribution analysis

The key idea: run something expensive once, then do multiple cheaper downstream operations on it.

### Config Structure for Downstream Scripts

```yaml
upstream_dir: "data/my_experiment/run_001"  # Points to existing run
output_dir: "data/my_experiment/run_001/downstream/evaluation"  # Subdirectory within upstream
```

### Key Principles

- `upstream_dir`: Points to the **run directory** containing results to work on
- `output_dir`: Points to where **downstream outputs** should be saved (typically `upstream_dir/downstream/<name>/`)
- Downstream scripts **never modify** the upstream data
- `--overwrite` flag only affects the downstream subdirectory, not the upstream
- **Composable**: downstream of downstream is natural - just point `upstream_dir` to a previous downstream output

### Example Downstream Script

```python
def main(config_path, overwrite=False):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required fields
    if 'upstream_dir' not in config:
        raise ValueError("FATAL: 'upstream_dir' required for downstream work")
    if 'output_dir' not in config:
        raise ValueError("FATAL: 'output_dir' required")

    upstream_dir = Path(config['upstream_dir'])
    output_dir = Path(config['output_dir'])

    # Check upstream results exist
    if not upstream_dir.exists():
        raise FileNotFoundError(f"Upstream dir not found: {upstream_dir}")

    # Create downstream output directory (only overwrites this downstream, not upstream!)
    if output_dir.exists() and not overwrite:
        raise ValueError(f"Output {output_dir} exists. Use --overwrite to replace.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load upstream results and do downstream work
    # ... your code here ...

    # Save downstream outputs
    # ...
```

### Example Directory Structure

```
data/my_experiment/run_001/
├── config.yaml              # Original run config (untouched)
├── results/                 # Original results (untouched)
├── logs/                    # Original logs (untouched)
└── downstream/
    ├── evaluation/          # First downstream
    │   ├── config.yaml
    │   ├── results/
    │   └── downstream/      # Downstream of downstream!
    │       └── detailed_scores/
    │           └── ...
    ├── visualization/       # Another downstream of run_001
    │   └── ...
    └── data_distribution/   # Yet another
        └── ...
```

### Why This Pattern

- **Preserves integrity**: upstream data is never modified
- **Multiple downstream**: run many different operations on the same upstream
- **Composable**: downstream of downstream works naturally
- **Re-runnable**: `--overwrite` lets you iterate on downstream without re-running expensive upstream
- **Clear lineage**: directory structure shows what depends on what

## Package Management

We use `uv` for Python package management.

**Always use the uv virtual environment:**
```bash
# Option 1: Activate the environment
source .venv/bin/activate
python src/approach_a/scripts/run_experiment.py configs/approach_a/experiment.yaml

# Option 2: Use uv run (no activation needed)
uv run python src/approach_a/scripts/run_experiment.py configs/approach_a/experiment.yaml
```

**Always add packages with uv:**
```bash
uv add numpy pandas matplotlib
uv add scipy  # Never use pip install
```

## Code Conventions

### Imports
```python
# Good - using src module structure
from src.utils import init_directory
from src.approach_a.model import MyModel

# Bad - relative imports or files outside src/
from model import MyModel       # Wrong
from ../utils import helper     # Wrong
```

## Environment Setup

1. Clone repository
2. Copy `.env.example` to `.env` and configure settings
3. Install dependencies: `uv sync`
4. Run experiments: `bash scripts/<track>/run_experiment.sh`

## Git Conventions

- `/data/` is gitignored - never commit experiment outputs
- `.env` is gitignored - use `.env.example` as template
- `uv.lock` is committed for reproducibility
- Commit messages should be descriptive

## Testing

Before committing:
1. Ensure all scripts run without errors
2. Check that output appears in correct directories
3. Verify configs have required `output_dir` field
4. Run `uv sync` to ensure dependencies are locked

## Common Commands

```bash
# Run experiment
bash scripts/<track>/run_experiment.sh

# Add new package
uv add package_name

# Sync dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

## Troubleshooting

- **ModuleNotFoundError**: Ensure you're using `uv run` or activated `.venv`
- **Import errors**: Check that imports use `from src.module import ...`
- **Missing output**: Verify config has `output_dir` field
- **Permission denied**: Make scripts executable with `chmod +x`

## Project Information

### Research Focus
[Add your research questions and goals here]

### Key Components
[Add descriptions of your main modules and components here]

### Key Metrics
[Add the metrics you're tracking here]

### Expected Outcomes
[Add your expected results here]

---

## Key Takeaways
- **Implementation is HOW. Orchestration is WHAT/WHEN/WHETHER.**
- **Research integrity demands explicit behavior** - no hidden magic
- **Better to crash loudly than fail silently**
- **When in doubt, be explicit and ask before abstracting**

## IMPORTANT
- NEVER create file variations like _v2, _new, _final, _updated. Use git for versioning.
- Always follow the existing project structure as defined in this document
- Config files must specify `output_dir` - this is non-negotiable
- All data outputs go to the path specified in `output_dir` in the config
