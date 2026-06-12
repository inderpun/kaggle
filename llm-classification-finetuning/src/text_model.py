"""Rung 2: TF-IDF sparse features + Logistic Regression for LLM preference prediction.

Architecture (v2 — tuned via diagnose_rung2.py / diagnose_rung2b.py):
    response_a, response_b, prompt → parse JSON turns
        → TF-IDF word n-grams (1–2) on A and B (shared vocabulary, separate encodings)
        → rowwise cosine(A, B) similarity (tie detector)
        → structural features from rung 1 (16 features, max-abs scaled)
        → hstack (sparse) → LogisticRegression (lbfgs, C=0.25)

    Diagnostics found: char n-grams hurt (-0.03 val), C=1 overfits; the
    regularization U-curve bottoms at C≈0.25; A/B cosine helps the tie class.

Key design: each TF-IDF vectorizer is fit on A-texts + B-texts together (shared
vocabulary), then transforms A and B separately. This lets the model see the same
word in "A context" vs "B context" without leaking position into the vocabulary.

Evaluation:
    5-fold stratified CV (same folds as baseline.py: StratifiedKFold, seed 42)
    Reporting mean ± std multiclass log loss.

Output: predictions written to outputs/submission_rung2.csv

Run:
    python3 src/text_model.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Import shared helpers from baseline (rung 1) — do NOT copy-paste
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from baseline import (  # noqa: E402
    safe_json_loads,
    extract_features,
    class_prior_log_loss,
    load_csv,
    write_submission,
    LABEL_COLS,
    RANDOM_STATE,
    N_FOLDS,
    DATA_DIR,
    OUTPUTS_DIR,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("text_model")


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def extract_texts(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return (texts_a, texts_b, texts_prompt) — one string per row."""
    texts_a: list[str] = []
    texts_b: list[str] = []
    texts_p: list[str] = []

    for _, row in df.iterrows():
        prompts = safe_json_loads(row.get("prompt", ""))
        resp_a = safe_json_loads(row.get("response_a", ""))
        resp_b = safe_json_loads(row.get("response_b", ""))
        texts_a.append(" ".join(resp_a))
        texts_b.append(" ".join(resp_b))
        texts_p.append(" ".join(prompts))

    return texts_a, texts_b, texts_p


# ---------------------------------------------------------------------------
# TF-IDF feature builders (shared vocabulary, separate encodings)
# ---------------------------------------------------------------------------

def rowwise_cosine(a: sp.csr_matrix, b: sp.csr_matrix) -> np.ndarray:
    """Cosine similarity between corresponding rows of two CSR matrices.

    The tie-detector feature: ties correlate with near-interchangeable responses,
    which per-side word counts cannot express.
    """
    dots = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    na = np.sqrt(np.asarray(a.multiply(a).sum(axis=1)).ravel())
    nb = np.sqrt(np.asarray(b.multiply(b).sum(axis=1)).ravel())
    return dots / np.maximum(na * nb, 1e-9)


def fit_tfidf_pair(
    texts_a: list[str],
    texts_b: list[str],
    *,
    analyzer: str = "word",
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 5,
    max_features: int = 200_000,
) -> TfidfVectorizer:
    """Fit a TF-IDF vectorizer on A+B texts combined (shared vocabulary).

    Args:
        texts_a: List of response_a strings.
        texts_b: List of response_b strings.
        analyzer: 'word' or 'char_wb'.
        ngram_range: n-gram range tuple.
        min_df: Minimum document frequency.
        max_features: Cap on vocabulary size.

    Returns:
        Fitted TfidfVectorizer.
    """
    combined = texts_a + texts_b
    vec = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
    )
    vec.fit(combined)
    log.info(
        "TF-IDF(%s, %s) vocabulary size: %d",
        analyzer,
        ngram_range,
        len(vec.vocabulary_),
    )
    return vec


class TextFeatureBuilder:
    """Builds and holds the four TF-IDF vectorizers for the pipeline.

    Vectorizers:
        word_vec  — word n-grams (1,2), max_features=200k
        char_vec  — char_wb n-grams (3,5), max_features=300k

    Each is fit on A+B combined, then transforms A and B separately.
    Final feature matrix per row:
        [word(A) | word(B) | char(A) | char(B) | structural(16)]
    """

    def __init__(
        self,
        word_max_features: int = 200_000,
        char_max_features: int = 300_000,
    ) -> None:
        self.word_max_features = word_max_features
        self.char_max_features = char_max_features
        self.word_vec: TfidfVectorizer | None = None
        self.char_vec: TfidfVectorizer | None = None
        # Max-abs scale for the structural block: TF-IDF lives in [0,1] while raw
        # lengths reach 1e4 — unscaled they dominate the gradient and stall the solver.
        self.struct_scale: np.ndarray | None = None

    def fit(
        self,
        texts_a: list[str],
        texts_b: list[str],
    ) -> "TextFeatureBuilder":
        """Fit vectorizers on the union of A and B texts.

        Char vectorizer is only fit when char_max_features >= 2 — diagnostics
        showed char n-grams are net noise on this task (see module docstring).
        """
        log.info("Fitting word n-gram vectorizer (shared vocab A+B)...")
        self.word_vec = fit_tfidf_pair(
            texts_a,
            texts_b,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=5,
            max_features=self.word_max_features,
        )

        if self.char_max_features >= 2:
            log.info("Fitting char n-gram vectorizer (shared vocab A+B)...")
            self.char_vec = fit_tfidf_pair(
                texts_a,
                texts_b,
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=5,
                max_features=self.char_max_features,
            )
        return self

    def transform(
        self,
        texts_a: list[str],
        texts_b: list[str],
        structural: np.ndarray,
    ) -> sp.csr_matrix:
        """Transform texts and structural features into a combined sparse matrix.

        Returns:
            Sparse matrix of shape (n_rows, word_vocab*2 + char_vocab*2 + 16).
        """
        if self.word_vec is None:
            raise RuntimeError("Call fit() before transform()")

        word_a = self.word_vec.transform(texts_a)
        word_b = self.word_vec.transform(texts_b)
        blocks = [word_a, word_b]

        if self.char_vec is not None:
            blocks.append(self.char_vec.transform(texts_a))
            blocks.append(self.char_vec.transform(texts_b))

        # A/B similarity (tie detector) on the word TF-IDF representation
        sim = rowwise_cosine(word_a, word_b).reshape(-1, 1)
        blocks.append(sp.csr_matrix(sim))

        # structural features — max-abs scale (train-fitted), then sparse for hstack
        if self.struct_scale is None:
            raise RuntimeError("Call fit_transform() before transform()")
        blocks.append(sp.csr_matrix(structural / self.struct_scale))

        combined = sp.hstack(blocks, format="csr")
        log.info("Combined feature matrix shape: %s", combined.shape)
        return combined

    def fit_transform(
        self,
        texts_a: list[str],
        texts_b: list[str],
        structural: np.ndarray,
    ) -> sp.csr_matrix:
        """Fit and transform in one step (also fits the structural scaler)."""
        self.fit(texts_a, texts_b)
        self.struct_scale = np.maximum(np.abs(structural).max(axis=0), 1.0)
        return self.transform(texts_a, texts_b, structural)


# ---------------------------------------------------------------------------
# Logistic Regression model (saga solver for sparse inputs)
# ---------------------------------------------------------------------------

def build_lr() -> LogisticRegression:
    """Build LogisticRegression suitable for sparse high-dimensional inputs.

    lbfgs (second-order) converges in far fewer passes than saga at this scale
    once all feature blocks are on comparable scales (TF-IDF in [0,1], structural
    max-abs scaled). Sparse input is supported natively.
    """
    return LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=0.25,  # regularization U-curve bottom (diagnose_rung2b.py)
        random_state=RANDOM_STATE,
    )


# ---------------------------------------------------------------------------
# CV and full fit
# ---------------------------------------------------------------------------

def cross_validate_text(
    df: pd.DataFrame,
    y: np.ndarray,
    n_folds: int = N_FOLDS,
    word_max_features: int = 200_000,
    char_max_features: int = 0,
) -> tuple[float, float]:
    """Run stratified k-fold CV on rung 2 features; return (mean_ll, std_ll).

    NOTE: Each fold re-fits the TF-IDF vectorizers on the fold's training split
    only (no leakage from the validation set). The vectorizers are not shared
    across folds — each fold builds its own shared A+B vocabulary from train rows.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_losses: list[float] = []

    log.info("Extracting texts from full training set...")
    texts_a, texts_b, _ = extract_texts(df)
    log.info("Extracting structural features from full training set...")
    structural = extract_features(df)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(structural, y), start=1):
        log.info("--- Fold %d/%d ---", fold_idx, n_folds)

        # Slice texts and structural for this fold
        tr_a = [texts_a[i] for i in train_idx]
        tr_b = [texts_b[i] for i in train_idx]
        val_a = [texts_a[i] for i in val_idx]
        val_b = [texts_b[i] for i in val_idx]
        struct_tr = structural[train_idx]
        struct_val = structural[val_idx]
        y_tr = y[train_idx]
        y_val = y[val_idx]

        # Build TF-IDF on training fold only
        builder = TextFeatureBuilder(
            word_max_features=word_max_features,
            char_max_features=char_max_features,
        )
        X_tr = builder.fit_transform(tr_a, tr_b, struct_tr)
        X_val = builder.transform(val_a, val_b, struct_val)

        # Train and evaluate
        clf = build_lr()
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_val)
        loss = log_loss(y_val, proba)
        fold_losses.append(loss)
        log.info("  Fold %d/%d  log_loss=%.5f", fold_idx, n_folds, loss)

    mean_ll = float(np.mean(fold_losses))
    std_ll = float(np.std(fold_losses))
    return mean_ll, std_ll


# ---------------------------------------------------------------------------
# Submission writer (rung 2 variant — writes to submission_rung2.csv)
# ---------------------------------------------------------------------------

def write_submission_rung2(
    clf: LogisticRegression,
    X_test: sp.csr_matrix,
    test_ids: pd.Series,
    sample_sub: pd.DataFrame,
) -> Path:
    """Predict on test set and write submission_rung2.csv."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "submission_rung2.csv"

    proba = clf.predict_proba(X_test)
    classes = clf.classes_

    pred_df = pd.DataFrame({
        "id": test_ids.values,
        LABEL_COLS[int(classes[0])]: proba[:, 0],
        LABEL_COLS[int(classes[1])]: proba[:, 1],
        LABEL_COLS[int(classes[2])]: proba[:, 2],
    })

    expected_cols = list(sample_sub.columns)
    pred_df = pred_df[expected_cols]

    id_order = sample_sub["id"].tolist()
    pred_df = pred_df.set_index("id").reindex(id_order).reset_index()

    pred_df.to_csv(out_path, index=False)
    log.info("Submission written to %s (%d rows)", out_path, len(pred_df))
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """End-to-end pipeline: load data → TF-IDF + structural features → CV → fit full → submit."""

    # --- Load data ---
    train_df = load_csv(DATA_DIR / "train.csv")
    test_df = load_csv(DATA_DIR / "test.csv")
    sample_sub = load_csv(DATA_DIR / "sample_submission.csv")
    log.info("train rows=%d  test rows=%d", len(train_df), len(test_df))

    # --- Labels ---
    y = np.argmax(train_df[LABEL_COLS].values, axis=1)
    log.info("Class distribution: %s", dict(zip(LABEL_COLS, np.bincount(y))))

    # --- Class-prior floor ---
    prior_ll = class_prior_log_loss(y)
    log.info("Class-prior log loss (floor): %.5f", prior_ll)

    # --- Cross-validation ---
    log.info("Running %d-fold stratified CV (rung 2: TF-IDF + LR)...", N_FOLDS)
    mean_ll, std_ll = cross_validate_text(
        train_df,
        y,
        n_folds=N_FOLDS,
        word_max_features=200_000,
        char_max_features=0,
    )
    log.info(
        "CV log loss: %.5f ± %.5f  (class-prior floor: %.5f  delta: %.5f)",
        mean_ll, std_ll, prior_ll, mean_ll - prior_ll,
    )

    # --- Fit full model ---
    log.info("Fitting final model on full training set...")
    log.info("Extracting texts and structural features from full training set...")
    texts_a, texts_b, _ = extract_texts(train_df)
    structural_train = extract_features(train_df)

    log.info("Extracting texts and structural features from test set...")
    test_a, test_b, _ = extract_texts(test_df)
    structural_test = extract_features(test_df)

    builder = TextFeatureBuilder(word_max_features=200_000, char_max_features=0)
    X_train_full = builder.fit_transform(texts_a, texts_b, structural_train)
    X_test_full = builder.transform(test_a, test_b, structural_test)

    clf = build_lr()
    clf.fit(X_train_full, y)

    # --- Write submission ---
    write_submission_rung2(clf, X_test_full, test_df["id"], sample_sub)

    log.info("Done.")
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("  Class-prior log loss (floor)  : %.5f", prior_ll)
    log.info("  Rung 1 CV log loss (reference): 1.06680")
    log.info("  Rung 2 CV log loss (5-fold)   : %.5f ± %.5f", mean_ll, std_ll)
    log.info("  Improvement over rung 1        : %.5f", 1.06680 - mean_ll)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
