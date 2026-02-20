# ES for Neurosymbolic Systems: Discussion Notes

## The EGGROLL Claim

The EGGROLL paper (arXiv:2511.16652) claims ES enables "large scale end-to-end neurosymbolic systems with nondifferentiable components." This appears only in the conclusion as future work — they haven't actually done neurosymbolic experiments.

## Why the Claim is Overstated (for standard tool use)

For standard LLM + tool use, the "non-differentiable" part is in the environment, not the policy:

```
LLM generates: "<tool>calc(2+2)</tool>"  ← differentiable (autoregressive sampling)
Tool executes: returns "4"               ← non-differentiable, but irrelevant
LLM continues with "4" in context        ← differentiable (grad-mask tool output)
Reward: is final answer correct?         ← scalar signal, policy gradient handles this
```

GRPO/policy gradient only needs gradients through the policy. Tool execution is just environment dynamics — same as not backpropping through Atari physics.

## Where ES Actually Has an Advantage

ES matters when optimizing **discrete parameters that are NOT model outputs** — i.e., system-level configuration.

### Example: Joint optimization of scaffold + LLM

Say you have 20 calculator APIs: `(usage_instruction, tool_name, format, output_format)`

**If API choice is model output** (model generates "use API #7"):
- GRPO handles this fine — just another token

**If API choice is system config** (which instructions go in prompt):
- GRPO: Train 20 models, pick best (grid search)
- ES: Can jointly optimize (weights, api_choice) in one loop

## ES vs Hybrid (REINFORCE-configs + GRPO-weights)

An alternative to ES: use REINFORCE to learn a distribution over configs, GRPO for weights.

| Aspect | Hybrid | ES |
|--------|--------|-----|
| Config updates | Episode-level reward | Episode-level reward |
| Weight updates | First-order (gradient) | Zeroth-order (perturbation) |
| Sample efficiency | Higher (uses gradients for weights) | Lower |
| Config space | Simple categorical works well | Better for large/combinatorial |
| Optimization | Two nested algorithms | Unified framework |

**Key insight**: Both use episode-level reward for config selection. The difference is that Hybrid uses gradient information for LLM weights, while ES uses only perturbation + fitness.

For "pick 1 of 20 APIs + train LLM," Hybrid is probably better — you get gradient-based learning for weights while still searching configs.

## When ES Genuinely Wins

ES is better when:
1. Config space is large/combinatorial/structured (not just "pick 1 of N")
2. Config is inherently system-level (not expressible as model output)
3. You want unified optimization capturing config-weight interactions
4. Inference is cheap, can run large populations

### Concrete Example: LLM + Symbolic Planner

```
Setup:
- LLM generates goal specification from natural language
- PDDL planner finds plan to achieve goal
- Plan executed, reward = task success

Planner configs (NOT LLM outputs):
- Heuristic: {FF, LM-cut, blind, landmark, ...}
- Search: {A*, greedy-best-first, weighted-A*}
- Axiom subset: which domain rules to include
```

Why ES wins:
- Config is system-level, not LLM output
- Config space is combinatorial (heuristic × search × axioms × ...)
- Interactions matter: different LLM styles pair better with different planner configs
- ES finds best (LLM, planner-config) pair jointly

## Summary

- **Weak claim** (overstated): "ES enables LLM + tool use" — GRPO handles this fine
- **Strong claim** (valid): "ES enables joint optimization of LLM weights + discrete system configs that aren't model outputs"
- **Practical advice**: For simple config spaces, Hybrid (REINFORCE-config + GRPO-weights) may be more sample efficient. ES shines for complex/combinatorial configs and unified optimization.
