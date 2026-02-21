evolutionary-alignment/
├── README.md
├── CLAUDE.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── requirements.txt                # Legacy (use pyproject.toml / uv sync)
├── LICENSE
│
├── conciseness/                    # Conciseness task experiments (legacy structure)
│   ├── data/
│   │   ├── train.jsonl
│   │   └── eval.jsonl
│   ├── ES/
│   ├── GRPO/
│   ├── conciseness_eval.py
│   └── conciseness_eval_seeds_sweep.py
│
├── countdown/                      # Countdown task experiments (legacy structure)
│   ├── data/
│   │   └── countdown.json
│   ├── ES/
│   ├── GRPO/
│   ├── countdown_task.py
│   └── eval_countdown_vllm.py
│
├── utils/                          # Shared utilities (legacy)
│   ├── __init__.py
│   └── worker_extn.py
│
├── src/                            # Track-based code (new structure)
│   ├── __init__.py
│   └── sampling_analysis_v1/
│       ├── __init__.py
│       ├── utils.py                # Data loading, prompt building, pass@k
│       ├── worker_extension.py     # vLLM extension for weight perturbation
│       └── scripts/
│           ├── temperature_sampling.py
│           └── perturbation_sampling.py
│
├── configs/
│   └── sampling_analysis_v1/
│       ├── temperature_sampling.yaml
│       └── perturbation_sampling.yaml
│
├── scripts/
│   └── sampling_analysis_v1/
│       ├── temperature_sampling.sh
│       └── perturbation_sampling.sh
│
├── data/                           # Experiment outputs (gitignored)
│   └── sampling_analysis_v1/
│
├── docs/
│   ├── repo_usage.md
│   ├── research_context.md
│   ├── structure.md
│   ├── start.md
│   ├── closing_tasks.md
│   ├── paper-workflow.md
│   ├── paper_writing.md
│   ├── conciseness_eval_comparison.md
│   ├── neurosymbolic_es.md
│   ├── logs/
│   │   ├── 2026-01-05/
│   │   └── 2026-02-21/
│   └── tracks/
│       └── sampling_analysis_v1/
│           └── progress.md
│
├── paper/                          # Overleaf paper (gitignored, separate git repo)
│
└── resources/                      # Reference materials (gitignored)
    ├── eggroll/
    └── es-fine-tuning-paper/
