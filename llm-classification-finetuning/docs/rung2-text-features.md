# Rung 2: TF-IDF + Linear Models — Design & Learning Notes

*Why 1970s-era text representation still earns a place in every 2026 pipeline.*

## The idea

Rung 1 looked only at *shape* (lengths, formatting). Rung 2 finally reads the *words* —
but cheaply, with *sparse lexical features* instead of a neural encoder.

**TF-IDF** scores each term by how frequent it is in this document (TF) discounted by
how common it is everywhere (IDF): distinctive words light up, filler words fade.
Two views, both useful here:
- **Word n-grams (1–2)** capture content ("recipe", "sorry, I cannot").
- **Char n-grams (3–5)** capture style robustly — typos, markdown habits, emoji,
  code tokens — and are the classic trick for authorship/style problems. Judging
  "which answer feels better" is partly a style problem.

Why a **linear model** on top: TF-IDF produces ~10⁵–10⁶ sparse dimensions. Logistic
regression consumes sparse matrices natively and is naturally well-calibrated (it
optimizes log loss directly — exactly our metric). Tree ensembles need dense or
reduced input (SVD first), losing the rare-feature signal that sparse LR exploits.
Industry shorthand: **sparse + linear, dense + trees.**

## Architecture

```
response_a ─┐                       ┌─ TFIDF_word(A), TFIDF_char(A)
response_b ─┼─ parse JSON turns ────┼─ TFIDF_word(B), TFIDF_char(B)
prompt     ─┘                       └─ structural features (rung 1's 16)
                     hstack (sparse)  →  LogisticRegression (multinomial, saga)
```

Key detail — **shared vocabulary, separate encodings**: fit each vectorizer on
A-texts + B-texts together (one vocabulary), then transform A and B separately.
The model then sees *the same word* as feature "in A" vs "in B", letting it learn
position-symmetric quality signals. (Fitting separate vocabularies per side would
leak position into the vocabulary itself.)

## What to expect, and what it teaches

- Target: **CV log loss ~1.00–1.04** (from 1.067). The gain over rung 1 quantifies
  how much *lexical content* adds over *structure* alone.
- Calibration check still applies: LR's probabilities are usable as-is; verify the
  CV/leaderboard gap stays small like rung 1's (1.067→1.073).
- Interview line: "before reaching for a GPU, a sparse linear baseline tells you how
  much of the signal is lexical — and it costs 3 minutes of CPU."

## Implementation contract

- `src/text_model.py`: loads train, builds the feature union above
  (`TfidfVectorizer(ngram_range=(1,2), min_df=5, max_features=200_000)` word +
  `TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=5, max_features=300_000)` char),
  5-fold CV with the same folds/reporting style as `baseline.py`, fits full, writes
  `outputs/submission_rung2.csv`. Reuse `safe_json_loads` + structural features by
  importing from `baseline.py` — do not copy-paste them.
- Kaggle kernel `notebooks/kernel-rung2/` (NEW kernel, separate from rung 1 — each
  rung is its own public-notebook candidate): same robust `/kaggle/input` glob
  discovery, narrative markdown for an external reader, prints CV table comparing
  rung 1 → rung 2.
- Memory care: `min_df=5` + `max_features` caps keep the matrices < 2GB. If the
  full-train fit exceeds Kaggle's RAM, fall back to `max_features` halved.
- Acceptance: local CV < 1.05; kernel pushed (private) and COMPLETE on Kaggle.
