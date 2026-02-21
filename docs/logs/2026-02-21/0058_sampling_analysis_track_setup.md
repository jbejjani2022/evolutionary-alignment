# Session Log: 2026-02-21 ~00:00-00:58

## Summary
Set up `sampling_analysis_v1` track to test whether ES's advantage comes from weight-space sampling (no training needed) vs token-space sampling (temperature). Also migrated repo to uv, rewrote research_context.md, and added template docs.

## Tasks Completed

### Repo Infrastructure
- Copied template files from research-template: `CLAUDE.md`, `docs/start.md`, `docs/closing_tasks.md`, `docs/repo_usage.md`
- Created `pyproject.toml` with all dependencies (migrating from `requirements.txt`)
- Set up uv environment (Python 3.10, pinned transformers<5.0.0 for vllm 0.11.0 compatibility)
- Created `src/__init__.py` for package structure
- Updated `.gitignore` to properly ignore `data/`, `paper/`, `resources/`

### Documentation
- Rewrote `docs/research_context.md` — comprehensive rewrite incorporating all findings from the paper (abstract, intro, all experiment sections, alignment results, weight space analysis)
- Created `docs/tracks/sampling_analysis_v1/progress.md`

### sampling_analysis_v1 Track
- Created full track structure: `src/`, `configs/`, `scripts/`, `docs/tracks/`
- **Core question**: Does ES need iterative training, or does just sampling weight perturbations and picking the best already work?
- Two experiments comparing equal budget (1024 rollouts per prompt):
  1. **Temperature sampling** (`temperature_sampling.py`): Base model, vary temperature [0.4, 0.6, 0.8, 1.0], top_p=1.0, compute pass@k
  2. **Perturbation sampling** (`perturbation_sampling.py`): Base model + Gaussian weight noise [σ=0.001, 0.003, 0.005, 0.01], greedy decode, compute pass@k
- Custom `SamplingWorkerExtension` — stores base weights in CPU memory, resets to base+noise each perturbation (zero numerical drift)
- Both scripts include greedy baseline and `--debug` mode (32 problems, 1 sweep value, 32 samples)
- Verified temperature sampling runs end-to-end in debug mode

## Files Created
- `CLAUDE.md`
- `docs/start.md`, `docs/closing_tasks.md`, `docs/repo_usage.md`
- `pyproject.toml`, `.python-version`
- `src/__init__.py`, `src/sampling_analysis_v1/__init__.py`
- `src/sampling_analysis_v1/utils.py` — shared data loading, prompt building, pass@k
- `src/sampling_analysis_v1/worker_extension.py` — base-weight-in-CPU perturbation strategy
- `src/sampling_analysis_v1/scripts/temperature_sampling.py`
- `src/sampling_analysis_v1/scripts/perturbation_sampling.py`
- `configs/sampling_analysis_v1/temperature_sampling.yaml`
- `configs/sampling_analysis_v1/perturbation_sampling.yaml`
- `scripts/sampling_analysis_v1/temperature_sampling.sh`
- `scripts/sampling_analysis_v1/perturbation_sampling.sh`
- `docs/tracks/sampling_analysis_v1/progress.md`

## Files Modified
- `docs/research_context.md` — comprehensive rewrite
- `.gitignore` — added data/, paper/, resources/

## Key Decisions
- Used `uv` (not pip/conda) per repo conventions
- Pinned `transformers<5.0.0` for vllm 0.11.0 compatibility
- Worker extension stores base weights in CPU memory rather than using perturb/restore (avoids any drift over 1024 perturbations)
- `gpu_memory_utilization=0.65` for this machine (44GB GPU with ~30GB free)

## Debug Run Results (Temperature Sampling)
- Greedy baseline: 0/32 correct (0%) — consistent with known ~0.1% base accuracy
- T=0.4, 32 samples/prompt: 1/32 prompts had >=1 correct (3.1%)

## Next Steps
- Run full temperature sampling (4 temperatures × 1024 samples × 2000 problems)
- Run full perturbation sampling (4 sigmas × 1024 perturbations × 2000 problems)
- Compare pass@k curves between temperature and weight perturbation diversity
- Downstream analysis: do weight perturbations find *different* correct solutions than temperature?
