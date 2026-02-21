# Evolutionary Alignment

Investigating Evolution Strategies (ES) as an alternative to Reinforcement Learning (GRPO) for LLM fine-tuning. ES uses weight perturbation rather than gradient-based optimization, and we find it explores qualitatively different solution spaces with less reward hacking on safety tasks.

## Setup

```bash
uv sync
```

## Running Experiments

```bash
# Temperature sampling (Countdown, base model)
bash scripts/sampling_analysis_v1/temperature_sampling.sh
bash scripts/sampling_analysis_v1/temperature_sampling_low.sh

# Weight perturbation sampling (Countdown, ES-style, no training)
bash scripts/sampling_analysis_v1/perturbation_sampling.sh

# Downstream plots and analysis
bash scripts/sampling_analysis_v1/plot_temperature.sh
bash scripts/sampling_analysis_v1/plot_perturbation.sh
bash scripts/sampling_analysis_v1/plot_comparison.sh
bash scripts/sampling_analysis_v1/solve_explorer.sh

# Debug mode (32 problems, 32 samples)
bash scripts/sampling_analysis_v1/temperature_sampling.sh --debug
```

## References

Our data and Evolution Strategies train scripts for Conciseness and Countdown tasks are based on the following work:
```bibtex
@misc{qiu2025evolutionstrategiesscalellm,
      title={Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning},
      author={Xin Qiu and Yulu Gan and Conor F. Hayes and Qiyao Liang and Elliot Meyerson and Babak Hodjat and Risto Miikkulainen},
      year={2025},
      eprint={2509.24372},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.24372},
}
```
