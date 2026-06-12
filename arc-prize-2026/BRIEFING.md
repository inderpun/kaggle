# Briefing: ARC Prize 2026 (ARC-AGI-2)

*What this competition really is, what it tests, and how the field attacks it.*

## What the competition is

ARC (Abstraction and Reasoning Corpus) is François Chollet's 2019 benchmark, built
around a pointed definition: **intelligence is skill-acquisition efficiency** — how
well you handle problems you've *never seen*, not how many skills you've memorized.
Each task shows 2–6 input→output grid pairs demonstrating some transformation rule
(colors 0–9, grids up to 30×30), then asks you to apply the rule to a new input.
Humans solve most tasks easily; they're designed so that *every task's rule is novel*.

ARC-AGI-2 (2025) is the hardened second edition — tasks were filtered against frontier
models specifically so that memorization-style solving fails. The prize pool is large
($700K on this track) because the organizers believe — correctly so far — that nobody
is close to the 85% grand-prize threshold.

Competition mechanics that shape everything:
- **Exact match only.** A 1-pixel error scores zero. No partial credit.
- **Two attempts** per test input (`pass@2`) — so *diverse* solvers beat one solver
  run twice. Ensembling different approaches into attempt_1/attempt_2 is free points.
- **Offline kernels with a fixed GPU budget, no internet.** You cannot call an API;
  whatever intelligence you bring must fit in the box. Efficiency is part of the score
  by design.

## What it's actually testing (the skills)

1. **Program synthesis** — the dominant framing: search for a *program* (a composition
   of grid transformations) that explains all demo pairs, then run it on the test
   input. This is the classic induction problem: many programs fit; prefer the
   shortest/simplest (Occam's razor, formally: minimum description length).
2. **Test-time adaptation** — the breakthrough idea from ARC Prize 2024: don't ship a
   fixed model, *train at inference time* on each task's own demo pairs (augmented
   with rotations/reflections/color permutations). The model literally learns the task
   while solving it. This is the closest thing the field has to "skill-acquisition
   efficiency" made concrete.
3. **Neural + symbolic hybrids** — pure neural nets generalize poorly here; pure
   symbolic search explodes combinatorially. Everything competitive blends them:
   LLMs *proposing* programs, search *verifying* them, or transformers guided by
   symbolic augmentation.
4. **Compute-budget engineering** — fitting model loading, per-task finetuning, and
   search for 240 tasks into the kernel's GPU hours. Real systems engineering.

## Why industry cares (interview framing)

- ARC is *the* reasoning benchmark frontier labs cite (OpenAI's o3 announcement led
  with its ARC score). Knowing why it's hard — and why benchmarks saturate except this
  one — signals you understand generalization vs memorization at a deep level.
- **Test-time compute** is the defining trend of current frontier models (reasoning
  models, long chains of thought, search at inference). ARC work gives you a concrete,
  defensible story about it.
- **Vocabulary to own**: program induction, DSL (domain-specific language), test-time
  training (TTT), augmentation groups (D8 symmetries, color permutations), pass@2,
  neuro-symbolic, skill-acquisition efficiency, MDL/Occam, o3-style test-time search.

## The standard approach ladder (what the field does)

| Rung | Approach | Character |
|---|---|---|
| 1 | Submission pipeline + local pass@2 scorer | infrastructure (← we are here) |
| 2 | **DSL + search**: hand-built grid-operation language, brute-force/guided search for programs consistent with all demos | interpretable, solves the "easy" fraction; Hodel's `arc-dsl` is the reference |
| 3 | **Test-time training**: small transformer (or finetuned LLM) trained per-task on augmented demos; the 2024 open-source winning family (~53–55% on ARC-v1) | the modern workhorse |
| 4 | **LLM program synthesis**: model writes candidate Python/DSL programs, executor verifies against demos, sample widely | the Greenblatt-style approach; needs an in-box code model |
| 5 | **Hybrid ensemble**: rung 2 answer as attempt_1, rung 3/4 answer as attempt_2; diversity maximizes pass@2 | how leaders actually submit |

Honest expectation-setting: ARC-AGI-2 top scores remain low (tens of percent at best
within compute budget — verify the current leaderboard, this moves). Scoring *anything*
nonzero with a clean, well-explained system puts you in respectable territory, and the
writeup is the portfolio piece regardless. This is a marathon track with a Nov 2
deadline; we bank checkpoints at every rung.

> ⚠️ Freshness note: SOTA numbers here move fast — exactly the staleness problem the
> Evergreen hub's Gardener agent is being built to solve. Treat numbers as
> last-verified 2026-06-11.

## Our plan against the ladder

1. Finish rung 1 this week: trivial solvers through the local scorer (verifies the
   harness; expect ~0%).
2. Rung 2 DSL with ~30 core grid operations + BFS over short programs; measure on the
   120 public eval tasks. Every solved task is interpretable — great writeup material.
3. Rung 3 TTT on Kaggle GPUs — this is where we learn serious GPU engineering.
4. Ensemble for pass@2.
