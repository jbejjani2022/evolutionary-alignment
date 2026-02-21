# Sampling Script Cleanup

## Summary
Simplified temperature sampling script by removing unnecessary round-based batching, and scaled down both sampling configs to feasible sizes (128 prompts, 512 samples).

## Tasks Completed
- Removed 16-round loop from `temperature_sampling.py` — vLLM handles `n=512` in a single `generate()` call internally, no need to split into `samples_per_call` chunks
- Enabled tqdm for temperature sampling (was `use_tqdm=False`)
- Removed `samples_per_call` config parameter (no longer needed)
- Simplified debug mode accordingly
- Reduced both configs from 2000 prompts × 1024 samples (~58h estimated) to 128 prompts × 512 samples (~1.8h estimated):
  - `temperature_sampling.yaml`: num_samples 2000→128, samples_per_prompt 1024→512
  - `perturbation_sampling.yaml`: num_samples 2000→128, num_perturbations 1024→512
- Updated K_VALUES in both scripts: removed k=1024 (max is now 512)

## Files Modified
- `src/sampling_analysis_v1/scripts/temperature_sampling.py` — removed round loop, single generate call per temperature
- `src/sampling_analysis_v1/scripts/perturbation_sampling.py` — K_VALUES capped at 512
- `configs/sampling_analysis_v1/temperature_sampling.yaml` — 128 prompts, 512 samples
- `configs/sampling_analysis_v1/perturbation_sampling.yaml` — 128 prompts, 512 perturbations

## Key Decisions
- pass@k for lower k values is computed combinatorially from the same 512 samples — no separate sampling runs needed per k

## Next Steps
- Run both sampling scripts and check results
- If 128 prompts is too few for stable pass@k estimates, scale up
