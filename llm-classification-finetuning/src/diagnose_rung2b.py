"""Rung 2 diagnostics, round 2: regularization sweep + A/B-similarity features.

Round 1 findings: char n-grams hurt (-0.029), C=4 overfits, best = word-only C=1
at 1.0589. Round 2 tests stronger regularization and tie-aware similarity features
(cosine of A/B TF-IDF vectors — ties are "responses interchangeable", invisible to
per-side word counts).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from baseline import LABEL_COLS, RANDOM_STATE, load_csv, DATA_DIR, extract_features  # noqa: E402
from text_model import TextFeatureBuilder, extract_texts  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("diagnose2")


def rowwise_cosine(a: sp.csr_matrix, b: sp.csr_matrix) -> np.ndarray:
    """Cosine similarity between corresponding rows of two CSR matrices."""
    dots = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    na = np.sqrt(np.asarray(a.multiply(a).sum(axis=1)).ravel())
    nb = np.sqrt(np.asarray(b.multiply(b).sum(axis=1)).ravel())
    denom = np.maximum(na * nb, 1e-9)
    return dots / denom


def main() -> None:
    train_df = load_csv(DATA_DIR / "train.csv")
    y = np.argmax(train_df[LABEL_COLS].values, axis=1)

    log.info("Extracting texts + structural features ...")
    texts_a, texts_b, _ = extract_texts(train_df)
    structural = extract_features(train_df)

    idx_tr, idx_val = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    tr_a = [texts_a[i] for i in idx_tr]; tr_b = [texts_b[i] for i in idx_tr]
    va_a = [texts_a[i] for i in idx_val]; va_b = [texts_b[i] for i in idx_val]
    y_tr, y_val = y[idx_tr], y[idx_val]

    builder = TextFeatureBuilder(word_max_features=200_000, char_max_features=1)
    X_tr = builder.fit_transform(tr_a, tr_b, structural[idx_tr])
    X_val = builder.transform(va_a, va_b, structural[idx_val])

    # Regularization sweep (word-only + scaled structural as built by the builder)
    for c in (0.5, 0.25, 0.1):
        clf = LogisticRegression(solver="lbfgs", max_iter=1000, C=c,
                                 random_state=RANDOM_STATE)
        clf.fit(X_tr, y_tr)
        ll = log_loss(y_val, clf.predict_proba(X_val))
        log.info("CONFIG word C=%-5s val_log_loss=%.5f n_iter=%s", c, ll, clf.n_iter_)

    # Similarity features appended (tie detector)
    word_tr_a = builder.word_vec.transform(tr_a)
    word_tr_b = builder.word_vec.transform(tr_b)
    word_va_a = builder.word_vec.transform(va_a)
    word_va_b = builder.word_vec.transform(va_b)
    sim_tr = rowwise_cosine(word_tr_a, word_tr_b).reshape(-1, 1)
    sim_va = rowwise_cosine(word_va_a, word_va_b).reshape(-1, 1)

    X_tr_sim = sp.hstack([X_tr, sp.csr_matrix(sim_tr)], format="csr")
    X_val_sim = sp.hstack([X_val, sp.csr_matrix(sim_va)], format="csr")

    for c in (0.5, 0.25):
        clf = LogisticRegression(solver="lbfgs", max_iter=1000, C=c,
                                 random_state=RANDOM_STATE)
        clf.fit(X_tr_sim, y_tr)
        ll = log_loss(y_val, clf.predict_proba(X_val_sim))
        log.info("CONFIG word+sim C=%-5s val_log_loss=%.5f n_iter=%s", c, ll, clf.n_iter_)

    log.info("Done.")


if __name__ == "__main__":
    main()
