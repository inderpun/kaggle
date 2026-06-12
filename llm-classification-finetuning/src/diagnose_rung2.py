"""Diagnose why rung 2 (TF-IDF + LR) scored at the class-prior floor.

Single 80/20 split, four controlled configs. Prints val log loss + n_iter_ for
each so we can separate (a) non-convergence, (b) over-regularization,
(c) char-feature noise. Findings drive the rung-2 fix.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from baseline import LABEL_COLS, RANDOM_STATE, load_csv, DATA_DIR  # noqa: E402
from text_model import TextFeatureBuilder, extract_texts  # noqa: E402
from baseline import extract_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("diagnose")


def run_config(name: str, X_tr, y_tr, X_val, y_val, clf) -> None:
    clf.fit(X_tr, y_tr)
    ll = log_loss(y_val, clf.predict_proba(X_val))
    n_iter = getattr(clf, "n_iter_", None)
    log.info("CONFIG %-28s val_log_loss=%.5f  n_iter=%s", name, ll, n_iter)


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

    # --- word-only features (no char, no structural) ---
    b_word = TextFeatureBuilder(word_max_features=200_000, char_max_features=1)
    Xw_tr = b_word.fit_transform(tr_a, tr_b, structural[idx_tr])
    Xw_val = b_word.transform(va_a, va_b, structural[idx_val])

    run_config("word C=1 lbfgs mi=1000", Xw_tr, y_tr, Xw_val, y_val,
               LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0, random_state=RANDOM_STATE))
    run_config("word C=4 lbfgs mi=1000", Xw_tr, y_tr, Xw_val, y_val,
               LogisticRegression(solver="lbfgs", max_iter=1000, C=4.0, random_state=RANDOM_STATE))
    run_config("word SGD a=1e-6", Xw_tr, y_tr, Xw_val, y_val,
               SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=30, tol=1e-4,
                             random_state=RANDOM_STATE))

    # --- full features (word + char + structural), convergence check ---
    b_full = TextFeatureBuilder(word_max_features=200_000, char_max_features=300_000)
    Xf_tr = b_full.fit_transform(tr_a, tr_b, structural[idx_tr])
    Xf_val = b_full.transform(va_a, va_b, structural[idx_val])

    run_config("full C=1 lbfgs mi=1000", Xf_tr, y_tr, Xf_val, y_val,
               LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0, random_state=RANDOM_STATE))

    log.info("Done.")


if __name__ == "__main__":
    main()
