# ARC Prize 2026 — ARC-AGI-2

Kaggle: [`arc-prize-2026-arc-agi-2`](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2) · deadline 2026-11-02

## Problem
ARC-AGI tasks are few-shot visual reasoning puzzles: each task gives 2–6 demonstration
pairs (input grid → output grid, colors 0–9, up to 30×30) and one or more test inputs.
You must produce the exact output grid — **exact match only**, partial credit none.
Two attempts allowed per test input (`attempt_1`, `attempt_2`); a task input scores if
either attempt is exactly right.

Dataset here: 1000 training tasks, 120 public-eval tasks (with solutions), 240 hidden
test tasks scored server-side. Submission is offline — notebooks run without internet
on Kaggle hardware, so any LLM must fit in the GPU budget (no API calls).

## Why this is the flagship
ARC-AGI-2 is deliberately resistant to memorization — it's the benchmark for abstract
reasoning and program synthesis, the open problem in the field. Even a modest score with
a clearly-explained novel approach is portfolio gold; the writeup matters as much as
the leaderboard.

## Approach ladder (each rung is a valid checkpoint)
1. **Pipeline rung** — submission machinery + local scorer over the 120 eval tasks.
   Trivial predictors (identity, most-common-output-shape heuristics) to verify the
   harness end-to-end. *Score ≈ 0, value = infrastructure.*
2. **Heuristic/DSL rung** — a small domain-specific language of grid transformations
   (symmetry, tiling, color mapping, object extraction, gravity…) + brute-force /
   guided search over short programs that explain all demo pairs. Classic approach,
   solves a real fraction of easier tasks, fully interpretable.
3. **Test-time training rung** — fine-tune a small transformer per-task on augmented
   demo pairs (the dominant open-source approach from ARC Prize 2024/25). Needs Kaggle
   GPU budget engineering.
4. **Hybrid rung** — DSL search + learned proposal/verification; ensemble attempts 1/2
   from different solvers (search answer + learned answer maximizes pass@2).

Strategy doc with the current-SOTA survey: `notebooks/strategy.md` (TODO).

## Layout
```
data/        # competition JSONs — gitignored, fetch via `make data`
src/         # arc_io.py (loading/scoring), solvers/
notebooks/   # strategy doc, experiment logs
```

## Reproduce
```bash
make data    # requires kaggle CLI + accepted rules
make score   # run current solver against the 120 public eval tasks
```

## Status
🚧 Rung 1: pipeline + local scorer.
