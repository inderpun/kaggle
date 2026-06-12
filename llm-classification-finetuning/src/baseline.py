"""Baseline model for the LLM preference prediction competition.

Features extracted per row (no external models, CPU-only):
  - Character length of response_a and response_b (after JSON parse)
  - Length delta (len_a - len_b) and log-ratio (log((len_a+1)/(len_b+1)))
  - Markdown signals: code fence count, bullet lines, header lines per response
  - Token-overlap (cheap set-overlap of response tokens with prompt tokens) per response
  - Turn count (number of prompt turns in the conversation)

Model: sklearn Pipeline with StandardScaler + multinomial LogisticRegression.

Evaluation:
  - 5-fold stratified CV reporting mean ± std multiclass log loss
  - Class-prior-only baseline (lower bound) computed separately

Output: data/test.csv predictions written to outputs/submission.csv.

Run:
    python3 src/baseline.py
"""
from __future__ import annotations

import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths (relative to this file, so the script works from any cwd)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
DATA_DIR = _HERE.parent / "data"
OUTPUTS_DIR = _HERE.parent / "outputs"

LABEL_COLS = ["winner_model_a", "winner_model_b", "winner_tie"]
N_FOLDS = 5
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("baseline")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def safe_json_loads(s: Any) -> list[str]:
    """Parse a JSON-encoded list of strings, with fallback for malformed input.

    Returns a list of strings (empty list for null/missing values).
    """
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return []
    try:
        val = json.loads(s)
        if isinstance(val, list):
            return [t if t is not None else "" for t in val]
        return [str(val)]
    except (json.JSONDecodeError, TypeError, ValueError):
        return [str(s)]


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV, raising with context if it's missing."""
    if not path.exists():
        log.error("File not found: %s", path)
        sys.exit(1)
    log.info("Loading %s ...", path)
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _count_code_fences(text: str) -> int:
    """Count occurrences of markdown code fences (``` or ~~~)."""
    return len(re.findall(r"^```|^~~~", text, re.MULTILINE))


def _count_bullet_lines(text: str) -> int:
    """Count lines starting with a bullet marker (-, *, +, or numbered lists)."""
    return len(re.findall(r"^\s*[-*+]\s|^\s*\d+\.\s", text, re.MULTILINE))


def _count_header_lines(text: str) -> int:
    """Count markdown header lines (# ... through ######)."""
    return len(re.findall(r"^#{1,6}\s", text, re.MULTILINE))


def _token_overlap(response_text: str, prompt_text: str) -> float:
    """Cheap word-token set overlap: |R ∩ P| / (|R| + 1).

    Measures how many unique words in the response also appear in the prompt,
    normalised by response vocabulary size.
    """
    r_tokens = set(re.findall(r"\w+", response_text.lower()))
    p_tokens = set(re.findall(r"\w+", prompt_text.lower()))
    if not r_tokens:
        return 0.0
    return len(r_tokens & p_tokens) / (len(r_tokens) + 1)


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """Build feature matrix from a DataFrame containing prompt/response columns.

    Returns an ndarray of shape (n_rows, n_features). All features are numeric
    and will be standardised downstream via the Pipeline.

    Feature layout (16 features):
        0  len_a          - total char length of response_a (all turns)
        1  len_b          - total char length of response_b (all turns)
        2  len_delta      - len_a - len_b
        3  len_log_ratio  - log((len_a+1)/(len_b+1))
        4  code_fences_a  - code fence count in response_a
        5  code_fences_b  - code fence count in response_b
        6  code_fence_delta - code_fences_a - code_fences_b
        7  bullets_a      - bullet lines in response_a
        8  bullets_b      - bullet lines in response_b
        9  bullet_delta   - bullets_a - bullets_b
        10 headers_a      - header lines in response_a
        11 headers_b      - header lines in response_b
        12 header_delta   - headers_a - headers_b
        13 overlap_a      - token overlap of response_a with prompt
        14 overlap_b      - token overlap of response_b with prompt
        15 turn_count     - number of prompt turns
    """
    rows: list[list[float]] = []

    log.info("Extracting features from %d rows ...", len(df))

    for _, row in df.iterrows():
        prompts = safe_json_loads(row.get("prompt", ""))
        resp_a = safe_json_loads(row.get("response_a", ""))
        resp_b = safe_json_loads(row.get("response_b", ""))

        text_a = " ".join(resp_a)
        text_b = " ".join(resp_b)
        text_p = " ".join(prompts)

        len_a = len(text_a)
        len_b = len(text_b)
        len_delta = len_a - len_b
        len_log_ratio = math.log((len_a + 1) / (len_b + 1))

        code_a = _count_code_fences(text_a)
        code_b = _count_code_fences(text_b)

        bullets_a = _count_bullet_lines(text_a)
        bullets_b = _count_bullet_lines(text_b)

        headers_a = _count_header_lines(text_a)
        headers_b = _count_header_lines(text_b)

        overlap_a = _token_overlap(text_a, text_p)
        overlap_b = _token_overlap(text_b, text_p)

        turn_count = len(prompts)

        rows.append([
            len_a, len_b, len_delta, len_log_ratio,
            code_a, code_b, code_a - code_b,
            bullets_a, bullets_b, bullets_a - bullets_b,
            headers_a, headers_b, headers_a - headers_b,
            overlap_a, overlap_b,
            turn_count,
        ])

    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """Construct the sklearn Pipeline: StandardScaler → LogisticRegression."""
    # Note: multi_class="multinomial" was removed in scikit-learn 1.5+.
    # With solver="lbfgs" and >2 classes the multinomial loss is used automatically.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            C=1.0,
            random_state=RANDOM_STATE,
        )),
    ])


def class_prior_log_loss(y: np.ndarray) -> float:
    """Log loss when always predicting the training class prior probabilities.

    This is the floor against which the model is compared.
    """
    n = len(y)
    classes, counts = np.unique(y, return_counts=True)
    priors = counts / n
    # Broadcast: every row gets the same prior vector
    y_pred = np.tile(priors, (n, 1))
    return float(log_loss(y, y_pred))


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = N_FOLDS,
) -> tuple[float, float]:
    """Run stratified k-fold CV; return (mean_log_loss, std_log_loss)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_losses: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        pipe = build_pipeline()
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_val)
        loss = log_loss(y_val, proba)
        fold_losses.append(loss)
        log.info("  Fold %d/%d  log_loss=%.5f", fold_idx, n_folds, loss)

    mean_ll = float(np.mean(fold_losses))
    std_ll = float(np.std(fold_losses))
    return mean_ll, std_ll


# ---------------------------------------------------------------------------
# Submission writer
# ---------------------------------------------------------------------------

def write_submission(
    model: Pipeline,
    X_test: np.ndarray,
    test_ids: pd.Series,
    sample_sub: pd.DataFrame,
) -> Path:
    """Predict on test set and write submission CSV matching the sample format.

    The sample submission defines column order and id ordering — we sort by
    that order rather than assuming test rows are already sorted.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "submission.csv"

    proba = model.predict_proba(X_test)
    classes = model.classes_  # should be [0, 1, 2] mapped to LABEL_COLS order

    # Build prediction DataFrame aligned to LABEL_COLS
    pred_df = pd.DataFrame({
        "id": test_ids.values,
        LABEL_COLS[int(classes[0])]: proba[:, 0],
        LABEL_COLS[int(classes[1])]: proba[:, 1],
        LABEL_COLS[int(classes[2])]: proba[:, 2],
    })

    # Re-order columns to match sample submission
    expected_cols = list(sample_sub.columns)
    pred_df = pred_df[expected_cols]

    # Re-order rows to match sample submission id order
    id_order = sample_sub["id"].tolist()
    pred_df = pred_df.set_index("id").reindex(id_order).reset_index()

    pred_df.to_csv(out_path, index=False)
    log.info("Submission written to %s (%d rows)", out_path, len(pred_df))
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """End-to-end pipeline: load data → features → CV → fit full → submit."""

    # --- Load data ---
    train_df = load_csv(DATA_DIR / "train.csv")
    test_df = load_csv(DATA_DIR / "test.csv")
    sample_sub = load_csv(DATA_DIR / "sample_submission.csv")

    log.info("train rows=%d  test rows=%d", len(train_df), len(test_df))

    # --- Labels (integer class index matching sklearn ordering) ---
    # LABEL_COLS order: [winner_model_a, winner_model_b, winner_tie] → classes 0,1,2
    y = np.argmax(train_df[LABEL_COLS].values, axis=1)
    log.info("Class distribution: %s", dict(zip(LABEL_COLS, np.bincount(y))))

    # --- Features ---
    X_train = extract_features(train_df)
    log.info("Feature matrix shape: %s", X_train.shape)

    X_test = extract_features(test_df)
    log.info("Test feature matrix shape: %s", X_test.shape)

    # --- Class-prior floor ---
    prior_ll = class_prior_log_loss(y)
    log.info("Class-prior log loss (floor): %.5f", prior_ll)

    # --- Cross-validation ---
    log.info("Running %d-fold stratified CV ...", N_FOLDS)
    mean_ll, std_ll = cross_validate(X_train, y)
    log.info(
        "CV log loss: %.5f ± %.5f  (class-prior floor: %.5f  delta: %.5f)",
        mean_ll, std_ll, prior_ll, mean_ll - prior_ll,
    )

    # --- Fit on full train ---
    log.info("Fitting final model on full training set ...")
    final_model = build_pipeline()
    final_model.fit(X_train, y)

    # --- Write submission ---
    write_submission(final_model, X_test, test_df["id"], sample_sub)

    log.info("Done.")
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("  Class-prior log loss (floor) : %.5f", prior_ll)
    log.info("  CV mean log loss (5-fold)    : %.5f", mean_ll)
    log.info("  CV std                       : %.5f", std_ll)
    log.info("  Improvement over prior       : %.5f", prior_ll - mean_ll)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
