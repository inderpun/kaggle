# LLM Classification — Predicting Human Preference Between LLM Responses

Kaggle: [`llm-classification-finetuning`](https://www.kaggle.com/competitions/llm-classification-finetuning)

## Problem
Given a user prompt and two competing LLM responses (`response_a`, `response_b`), predict
which response a human judge preferred — or whether they tied. This is the modeling task
behind [Chatbot Arena](https://lmarena.ai/): turning pairwise human votes into a reward signal.

- **Input:** `prompt`, `response_a`, `response_b` (plus the anonymized model names in train)
- **Target:** 3-way probability over `winner_model_a`, `winner_model_b`, `winner_tie`
- **Metric:** multiclass **log loss** (calibration matters as much as accuracy)

## Why this is a real GenAI problem
Preference modeling is the core of RLHF reward models and LLM-as-judge evaluation. A strong
solution has to reason about *response quality* (helpfulness, correctness, formatting,
verbosity bias, position bias) — not just surface text features.

## Approach (planned, iterative)
1. **Baseline** — class-prior / length-heuristic submission to lock in the pipeline and a log-loss floor.
2. **Feature baseline** — TF-IDF + gradient-boosted trees over engineered features
   (length deltas, formatting, overlap with prompt) for a fast, interpretable reference.
3. **Transformer model** — fine-tune a pretrained encoder (e.g. DeBERTa-v3) on the
   `[prompt, response_a, response_b]` triple to predict the 3-way label.
4. **Calibration + de-biasing** — temperature scaling; correct known position/verbosity bias.

## Layout
```
data/        # competition data — gitignored, fetch via `make data`
src/         # reusable modules (data loading, features, models)
notebooks/   # EDA + experiment notebooks (the Kaggle-facing deliverable)
```

## Reproduce
```bash
# 1. Accept competition rules on Kaggle, then:
make data          # downloads + extracts into data/
python3 src/eda.py # sanity-check + class balance
```

## Status
🚧 Scaffolding. Baseline next once data is downloaded.
