# EDA Findings — LLM Classification Finetuning (Chatbot Arena)

Generated: 2026-06-11

## Dataset Overview

| Metric | Value |
|---|---|
| Training rows | 57,477 |
| Test rows | (see test.csv) |
| Columns | id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie |
| Missing values | None in any column |

### Column Encoding

The `prompt`, `response_a`, and `response_b` columns are **JSON-encoded lists of strings** representing multi-turn conversations. Each element in the list is one turn. Rows with a single-turn conversation have a 1-element list. Parsing is required before computing lengths or turn counts — raw `.str.len()` on the JSON string overestimates and mixes in JSON syntax characters.

---

## Class Balance

| Class | Count | Rate |
|---|---|---|
| winner_model_a | 20,063 | 34.91% |
| winner_model_b | 19,650 | 34.19% |
| winner_tie | 17,764 | 30.90% |

The distribution is roughly balanced across the three classes with model_a slightly ahead of model_b and ties being the least common outcome.

---

## Position Bias

- `winner_model_a` rate: **34.91%**
- `winner_model_b` rate: **34.19%**
- Difference (a − b): **+0.72 percentage points**

There is a **small but consistent position bias toward model_a** (the first-shown response). The 0.72pp gap is modest — annotators show a slight preference for whichever response appeared first. This is consistent with the well-known "primacy bias" in preference annotation. The effect is small enough that position alone is a weak signal, but models should not be built that assume symmetry.

---

## Response Length Statistics

Lengths are computed after JSON parsing (sum of all turn character lengths per row).

| Column | Mean | p50 | p95 | p99 | Max |
|---|---|---|---|---|---|
| response_a | 1,330 | 1,036 | 3,584 | 6,783 | 53,265 |
| response_b | 1,337 | 1,044 | 3,574 | 6,734 | 52,371|
| prompt | 352 | 91 | 1,419 | 4,698 | 32,829 |

Median response lengths (~1,040 chars) are roughly 3× the median prompt length. The distribution is right-skewed with long tails (p99 is ~6.5× the median).

---

## Verbosity Bias (length-vs-win relationship)

**When response_a is longer than response_b (49.7% of rows):**

| Outcome | Win rate |
|---|---|
| winner_model_a | **43.09%** |
| winner_model_b | 26.26% |
| tie | 30.65% |

**When response_b is longer than response_a (49.6% of rows):**

| Outcome | Win rate |
|---|---|
| winner_model_a | 27.01% |
| winner_model_b | **42.48%** |
| tie | 30.51% |

**Finding: Strong verbosity bias.** The longer response wins at ~43% vs ~26% for the shorter response — a 16+ pp lift. When one response is longer, the longer one is ~1.6× more likely to win than the shorter one. This makes `len_a > len_b` (and its inverse) one of the most predictive single features available without any model inference.

---

## Multi-Turn Distribution

| Turn count | Row count | % of total |
|---|---|---|
| 1 | 49,938 | 86.9% |
| 2 | 4,673 | 8.1% |
| 3 | 1,485 | 2.6% |
| 4 | 607 | 1.1% |
| 5 | 311 | 0.5% |
| 6+ | 463 | 0.8% |

- **Single-turn conversations**: 86.9% of training data
- **Multi-turn (>1 prompt turn)**: 13.1% (7,539 rows)
- **Maximum turns**: 36

The majority of comparisons are single-turn. Multi-turn data is worth preserving in features (turn count may correlate with task complexity and preference patterns), but single-turn patterns dominate.

---

## Model Win Rates (Top 10, min 200 appearances)

Appearances = rows where the model appeared as either model_a or model_b.

| Model | Total Appearances | Wins | Win Rate |
|---|---|---|---|
| gpt-4-1106-preview | 7,387 | 4,073 | **55.1%** |
| gpt-3.5-turbo-0314 | 1,302 | 711 | 54.6% |
| gpt-4-0125-preview | 1,160 | 596 | 51.4% |
| gpt-4-0314 | 4,122 | 1,993 | 48.4% |
| claude-1 | 3,978 | 1,747 | 43.9% |
| gpt-4-0613 | 6,165 | 2,450 | 39.7% |
| claude-instant-1 | 4,136 | 1,642 | 39.7% |
| qwen1.5-72b-chat | 551 | 215 | 39.0% |
| claude-2.0 | 2,456 | 956 | 38.9% |
| llama-2-70b-chat | 3,428 | 1,277 | 37.3% |

**Bottom 5 (min 200 appearances):**

| Model | Total Appearances | Wins | Win Rate |
|---|---|---|---|
| stablelm-tuned-alpha-7b | 771 | 132 | 17.1% |
| llama-13b | 547 | 88 | 16.1% |
| chatglm3-6b | 989 | 157 | 15.9% |
| dolly-v2-12b | 800 | 124 | 15.5% |
| chatglm2-6b | 564 | 73 | **12.9%** |

**Finding:** GPT-4 variants dominate. The best model (gpt-4-1106-preview) wins 55% of comparisons while the worst (chatglm2-6b) wins only 13%. A ~63 unique models exist in the dataset. Model identity is a very strong predictor — knowing which two models are competing provides strong prior information about the likely outcome.

---

## Key Takeaways for Modeling

1. **Verbosity bias is the strongest cheap feature**: when len_a > len_b, model_a wins ~43% of the time vs 26% for model_b — a 16pp gap.
2. **Position bias exists but is small** (~0.72pp advantage for model_a). Not worth exploiting directly.
3. **Model identity is highly predictive** — GPT-4 wins ~55% vs chatglm ~13%. Encoding model elo/win-rate priors as features should significantly improve predictions.
4. **Single-turn dominates** (87%), but turn count may still be a useful feature for multi-turn rows.
5. **Classes are roughly balanced** (~35/34/31 split), so log loss on random predictions ≈ log(3) ≈ 1.099 and a class-prior baseline gives ≈ 1.086.
