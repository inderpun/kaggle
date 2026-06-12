# Rung 3: Test-Time Training — Design & Learning Notes

*The approach family that actually cracked ARC. Extends dsl-design.md / objects-design.md.*

## 1. Where we stand, and why TTT

Our measured ladder so far:

| Solver | Training (v1-era) | ARC-AGI-2 eval | Hidden LB |
|---|---|---|---|
| whole-grid DSL (30 ops) | 51/1000 | 0/120 | — |
| + object ops (~110) | 56/1000 | 0/120 | **0.00** |

The symbolic ceiling is real and we've hit it. The fundamental limit: a hand-coded DSL
can only express transformations *we anticipated*. ARC-AGI-2 tasks compose novel
concepts — the op you'd need wasn't in anyone's vocabulary.

**Test-time training** flips the strategy. Instead of searching a fixed hypothesis
space, *adapt the model to each task at inference time*: take the task's 2–6 demo
pairs, expand them into hundreds of training examples via augmentation, and fine-tune
briefly before predicting. The model's weights — not a program — become the hypothesis.
This was the engine of the open-source breakthroughs on ARC-v1 (the 2024 winning
family reached ~53–55%), and it's the concrete form of Chollet's "skill-acquisition
efficiency": the system literally acquires the skill during the test.

## 2. The mechanics

### Grid serialization
LLMs eat tokens, so grids become text — row-wise digit strings with line breaks.
Detail that matters: digit-per-cell tokenization (avoid BPE merging "23" into one
token); newline as row delimiter teaches the geometry.

### Augmentation — the data multiplier
A task gives ~4 demo pairs. The augmentation group multiplies them:
- **D8 symmetries** (8 transforms: rotations + reflections) applied to input *and*
  output consistently — the rule survives, the surface form changes.
- **Color permutations** (sample from 9! relabelings of non-background colors).
- **Demo-pair permutation + leave-one-out**: hold out each demo as a pseudo-test,
  train on the rest — turns 4 demos into 4 supervised episodes.

4 demos × 8 symmetries × k color maps × LOO ≈ **hundreds of examples per task**, all
guaranteed consistent with the unknown rule.

### Per-task adaptation
LoRA-finetune a small base model (see §3) for a fixed budget (~1–2 min/task) on the
augmented set, predicting output grids token-by-token with cross-entropy.

### Augmented inference + voting
Predict under multiple symmetries, invert each transform back to canonical
orientation, and **vote**. Variance across augmentations is a confidence signal, and
the top-2 vote clusters map directly onto attempt_1/attempt_2 — the same
distinct-hypothesis principle our DSL pass@2 uses, now in neural form.

## 3. Making it fit Kaggle's budget

The constraint stack: ~240 hidden tasks, GPU kernel ≤ 12h, offline (weights from a
Kaggle Dataset/Model). Budget ≈ 2.5–3 min per task all-in. The standard structure:

1. **Phase A — shared base (offline, once):** start from a small open model
   (0.5–2B class), finetune on the 1000 public training tasks with the same
   serialization + augmentation. This bakes in the *priors* (objectness, symmetry,
   counting) so per-task adaptation starts warm. Done in a separate internet-ON
   training kernel; weights saved as a Kaggle Dataset.
2. **Phase B — per-task TTT (in the submission kernel):** short LoRA adaptation per
   task from the shared base (seconds to ~1 min), then augmented-voting inference.
   Skip adaptation for tasks the DSL already solved (free time reclaim).
3. **Phase C — ensemble for pass@2:** attempt_1 = TTT vote winner; attempt_2 = DSL
   program's answer when one exists, else TTT runner-up. Maximally *different*
   hypothesis families per attempt — the pass@2 ideal.

## 4. Honest expectations & checkpoints

ARC-AGI-2 is hard for TTT too — open results sit far below v1 numbers (verify the
current leaderboard; it moves). Our checkpoints, each shippable:

- **3.1** Serialization + augmentation library, validated by round-tripping (apply →
  invert → identity) and by training-set statistics. CPU-only, fully testable.
- **3.2** Phase-A shared finetune; measure *zero-adaptation* accuracy on our 120-task
  local eval (this alone may beat 0).
- **3.3** Per-task TTT on a 20-task slice; tune the time/quality knob (steps, rank, LR).
- **3.4** Full pipeline under budget on all 120 → submit.

Any nonzero is a win condition; single-digit % on v2 with a clean writeup is a
genuinely respectable open result.

## 5. Interview vocabulary from this rung

Test-time training/adaptation, LoRA-per-task, augmentation groups (D8), equivariance,
leave-one-out episode construction, grid serialization & tokenization pitfalls,
self-consistency voting, compute-budget engineering, ensemble diversity for pass@k.
