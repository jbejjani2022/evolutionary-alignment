# Include σ=0.0005 in All Plots

## Summary
Updated all downstream plot scripts and configs to merge σ=0.0005 (from `perturbation_sampling_low`) into the perturbation results, matching how temperature already merges multiple runs.

## Tasks Completed
- Confirmed σ=0.0005 run had completed (pass@512 = 32.0%, 41/128 prompts solved)
- Confirmed σ=0.001 remains best perturbation setting (35.2% > 32.0%)
- Updated `plot_perturbation.py` to accept `upstream_dirs` (list) and merge multiple perturbation runs
- Updated `plot_comparison.py` to accept `perturbation_upstream_dirs` (list) with same merge pattern
- Updated `build_solve_explorer.py` with same multi-dir perturbation support
- Updated all three corresponding YAML configs to include both perturbation dirs
- Regenerated all plots (perturbation, comparison, solve explorer) — now show all 5 sigmas

## Files Modified
- `configs/sampling_analysis_v1/plot_perturbation.yaml` — `upstream_dir` → `upstream_dirs` (list)
- `configs/sampling_analysis_v1/plot_comparison.yaml` — `perturbation_upstream_dir` → `perturbation_upstream_dirs` (list)
- `configs/sampling_analysis_v1/solve_explorer.yaml` — same change
- `src/sampling_analysis_v1/scripts/plot_perturbation.py` — added `merge_perturbation_results()`, list-based loading
- `src/sampling_analysis_v1/scripts/plot_comparison.py` — added `merge_perturbation_results()`, list-based loading
- `src/sampling_analysis_v1/scripts/build_solve_explorer.py` — added `merge_perturbation_results()`, list-based loading

## Key Findings
- σ=0.0005: pass@512 = 32.0%, 41/128 solved — slightly worse than σ=0.001 (35.2%, 45/128)
- Overlap numbers unchanged (best perturb is still σ=0.001): 40 both, 16 temp-only, 5 perturb-only, 67 neither
- Temperature sampling dominates weight perturbation at every operating point — ES's advantage comes from iterative optimization, not single-shot weight-space exploration

## Discussion
The σ=0.0005 result narrows the viable perturbation band further: σ=0.001 is the sweet spot, with performance degrading in both directions. This contrasts with temperature which has a broader useful range (T=0.2–0.6 all reasonable).
