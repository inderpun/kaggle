# Briefing: LLM Classification Finetuning

*What this competition really is, what it tests, and how the field attacks it.*

## What the competition is

This is the Getting Started re-run of the **LMSYS Chatbot Arena Human Preference**
competition (mid-2024, one of the most-entered LLM comps ever). The data is real
Chatbot Arena traffic: a user typed a prompt, two anonymous LLMs answered side-by-side,
and the human clicked which answer was better (or "tie"). You get the prompt and both
responses; you predict the probability of each of the three outcomes.

The evaluation metric is **multiclass log loss** — you submit probabilities, not picks,
and you're punished hardest for being *confidently wrong*. That makes this a
**calibration** problem as much as an accuracy problem: a model that says 40/35/25 when
unsure beats a model that says 90/5/5 and is sometimes wrong.

## What it's actually testing (the skills)

1. **Preference modeling** — the core of RLHF. A model that predicts "which response
   does a human prefer?" *is* a reward model. The math behind Arena rankings is the
   **Bradley–Terry model** (pairwise comparisons → latent quality scores → Elo-style
   ratings). When you train on this data you are literally building the same artifact
   labs use to align their models.
2. **Bias-aware ML** — human judges have systematic, measurable biases, and our own EDA
   reproduced the famous ones:
   - **Verbosity bias**: the longer response wins ~1.6× more often (our numbers: 43.1%
     vs 26.3% when A is longer). Humans (and LLM judges!) over-reward length.
   - **Position bias**: response A wins slightly more (+0.72pp) just for being first.
   These matter because exploiting them gets you cheap points, and *correcting* for
   them is a real research topic in LLM evaluation.
3. **Long-text pair classification** — inputs are (prompt, response_a, response_b)
   triples, often long, sometimes multi-turn (13% of rows). Fitting these into a
   transformer's context window efficiently is the engineering crux.
4. **Calibration** — see above. Techniques: temperature scaling, label smoothing,
   never letting the softmax saturate.

## Why industry cares (interview framing)

- **Reward models / RLHF**: every aligned model (ChatGPT, Claude, Gemini) is trained
  against human-preference data exactly like this. "I built a preference model on
  Chatbot Arena data" is a sentence a hiring manager understands instantly.
- **LLM-as-judge**: automated evaluation (MT-Bench, Arena-Hard, every internal eval
  pipeline) uses an LLM to pick the better response. Judge biases — verbosity,
  position, self-preference — are exactly what this dataset lets you measure.
- **Vocabulary to own**: Bradley–Terry, Elo, reward model, RLHF/RLAIF, LLM-as-judge,
  position/verbosity bias, calibration, log loss vs accuracy, pairwise preference data.

## The standard approach ladder (what the field does)

| Rung | Approach | Expected log loss | What you learn |
|---|---|---|---|
| 0 | Class priors only | ~1.097 | the floor |
| 1 | **Structural features + LR** (← we are here) | **1.067 (our CV)** | the biases are real signal |
| 2 | TF-IDF / GBDT (LightGBM) on text features | ~1.03–1.05 | classic NLP still earns its keep |
| 3 | Finetuned encoder (DeBERTa-v3) on the triple | ~0.98–1.02 | transformer finetuning, GPU pipelines |
| 4 | Finetuned LLM (Gemma-2-9b / Llama-3-8b + **QLoRA**) with a classification head | ~0.88–0.93 | PEFT, quantization, fitting 9B models in 16GB |
| 5 | Winning recipes: ensembles of rung 4 + **position-swap TTA** (run A/B and B/A, average), pseudo-labeling extra Arena data, distillation | ~0.87–0.88 (2024 winners) | competition-grade engineering |

Position-swap test-time augmentation is the elegant trick worth understanding: by
evaluating both orderings and averaging, you *cancel the position bias by construction*.

## Our plan against the ladder

Rung 1 submitted (CV 1.0668). Next: rung 2 as a fast iteration exercise, then jump to
rung 4 on Kaggle GPUs (rung 3 is pedagogically useful but rung 4 is where the modern
skills are — QLoRA, quantization, context-length management). This is a **no-medal**
Getting Started comp, so we optimize for learning + a polished public notebook, then
take the skills to a medal-bearing comp.
