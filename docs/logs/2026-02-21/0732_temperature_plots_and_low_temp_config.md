# 2026-02-21 07:32 — Temperature Plots & Low-Temperature Config

## Summary

Ran the temperature sampling experiment (T=0.4–1.0), created downstream plotting scripts for the results, and added a new config for low-temperature sampling (T=0.05, 0.1, 0.2).

## Completed

- Reviewed temperature sampling results (T=0.4, 0.6, 0.8, 1.0 on Countdown, 128 prompts × 512 samples)
- Created downstream plotting script (`plot_temperature.py`) with three figures:
  - **pass@k curves**: Lower temperature strictly dominates at all k values
  - **Correct count distribution**: Heavy zero-inflation; most prompts unsolved at all temperatures
  - **Summary bars**: Comparing pass@1, pass@64, pass@512, and % prompts solved
- Created config and bash script for low-temperature sweep (T=0.05, 0.1, 0.2) — same setup, separate output dir
- Confirmed SyntaxWarning from Countdown eval is benign (model outputs like `(4)(3+2)` parsed by Python eval)

## Key Results

- Greedy: 0% accuracy
- T=0.4 pass@512: 43.8% (best), 56/128 prompts with ≥1 correct
- T=1.0 pass@512: 20.3% (worst), 26/128 prompts with ≥1 correct
- Lower temperature is strictly better — more concentrated sampling near the mode helps for this task
- Even at best temperature, pass@512 = 44% < ES post-training accuracy (~15.7% greedy)

## Files Created

- `src/sampling_analysis_v1/scripts/plot_temperature.py` — downstream plotting script
- `configs/sampling_analysis_v1/plot_temperature.yaml` — downstream config
- `scripts/sampling_analysis_v1/plot_temperature.sh` — bash wrapper
- `configs/sampling_analysis_v1/temperature_sampling_low.yaml` — low-temp config (T=0.05, 0.1, 0.2)
- `scripts/sampling_analysis_v1/temperature_sampling_low.sh` — bash wrapper

## Next Steps

- Run `bash scripts/sampling_analysis_v1/temperature_sampling_low.sh` for low-temperature results
- Run perturbation sampling experiment for weight-space comparison
- Combine all temperature results into unified plots once both runs complete
