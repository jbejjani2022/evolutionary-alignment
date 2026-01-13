evolutionary-alignment/
├── README.md                    # Project overview and setup
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
│
├── conciseness/                 # Main experiment code
│   ├── data/
│   │   ├── train.jsonl          # Training samples (2)
│   │   └── eval.jsonl           # Evaluation samples (8)
│   │
│   ├── ES/                      # Evolution Strategies
│   │   ├── es_fine-tuning_conciseness_iid.py  # Main ES training
│   │   ├── run_es.sh            # SLURM training script
│   │   └── eval_es.sh           # SLURM eval script
│   │
│   ├── GRPO/                    # Group Relative Policy Optimization
│   │   ├── train_grpo_conciseness_trl.py      # GRPO training
│   │   ├── run_grpo_sweep.sh    # SLURM sweep script
│   │   ├── conciseness_grpo_eval.sh           # SLURM eval script
│   │   ├── grpo_conciseness_trl.yaml          # Training config
│   │   ├── accelerate_trl_grpo.yaml           # Accelerate config
│   │   └── ds_zero2.json        # DeepSpeed ZeRO-2 config
│   │
│   ├── conciseness_eval.py      # Single model evaluation
│   └── conciseness_eval_seeds_sweep.py        # Multi-seed eval
│
├── docs/                        # Documentation
│   ├── research_context.md      # Research goals, findings, team assignments
│   ├── paper-workflow.md        # Overleaf sync instructions
│   ├── paper_writing.md         # AI content guidelines (\ai{} command)
│   ├── structure.md            # This file
│   └── logs/                    # Session logs
│       └── YYYY-MM-DD/          # Daily log directories
│           └── HHMM_topic.md    # Individual session logs
│
├── resources/                   # Reference materials
│   └── es-fine-tuning-paper/    # Old working repo (archived)
│
└── paper/                       # Overleaf paper (GITIGNORED - separate repo)
    ├── main.tex
    ├── sections/
    ├── appendix/
    ├── figures/
    └── ...
