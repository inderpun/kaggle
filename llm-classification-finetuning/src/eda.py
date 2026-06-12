"""Sanity-check + exploratory summary for the LLM preference dataset.

Run after `make data`. Prints schema, class balance, and basic length stats so we
understand the target distribution before modeling. No side effects beyond stdout.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("eda")

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LABEL_COLS = ["winner_model_a", "winner_model_b", "winner_tie"]


def load_train(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load train.csv, failing loudly with guidance if the data isn't present."""
    path = data_dir / "train.csv"
    if not path.exists():
        log.error("Missing %s — run `make data` (accept Kaggle rules first).", path)
        sys.exit(1)
    return pd.read_csv(path)


def summarize(df: pd.DataFrame) -> None:
    """Log schema, class balance, and response-length deltas."""
    log.info("rows=%d cols=%s", len(df), list(df.columns))

    if all(c in df.columns for c in LABEL_COLS):
        counts = df[LABEL_COLS].sum()
        log.info("class balance:\n%s", (counts / counts.sum()).round(4).to_string())

    for col in ("response_a", "response_b", "prompt"):
        if col in df.columns:
            lens = df[col].astype(str).str.len()
            log.info("%s length: mean=%.0f p50=%.0f p95=%.0f",
                     col, lens.mean(), lens.median(), lens.quantile(0.95))


def main() -> None:
    df = load_train()
    summarize(df)


if __name__ == "__main__":
    main()
