# Evolutionary Alignment
Aligning LLMs more reliably with Evolution Strategies.

## Setup

We use Python 3.12.11. Create an environment and install dependencies:
```bash
mamba create --name evolutionary-alignment python=3.12.11
mamba activate evolutionary-alignment
pip install -r requirements.txt
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
