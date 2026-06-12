"""ARC-AGI-2 data loading, types, and local scoring.

Grids are lists of lists of ints (colors 0-9). A task has demonstration pairs
(`train`) and one or more test inputs. Scoring is exact-match with two attempts
per test input (pass@2) — matching the competition's server-side metric so local
eval numbers are directly comparable to the leaderboard.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("arc_io")

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

Grid = list[list[int]]


@dataclass(frozen=True)
class Pair:
    """One demonstration pair: input grid -> expected output grid."""
    input: Grid
    output: Grid


@dataclass(frozen=True)
class Task:
    """One ARC task: demo pairs plus test inputs (solutions when available)."""
    task_id: str
    train: list[Pair]
    test_inputs: list[Grid]
    test_outputs: list[Grid] | None = field(default=None)  # None for hidden test set


def load_tasks(split: str, data_dir: Path = DATA_DIR) -> list[Task]:
    """Load a split: 'training' | 'evaluation' | 'test'.

    Raises FileNotFoundError with guidance if data hasn't been downloaded.
    """
    challenges_path = data_dir / f"arc-agi_{split}_challenges.json"
    if not challenges_path.exists():
        raise FileNotFoundError(f"{challenges_path} missing — run `make data` first")

    challenges = json.loads(challenges_path.read_text())
    solutions_path = data_dir / f"arc-agi_{split}_solutions.json"
    solutions = (
        json.loads(solutions_path.read_text()) if solutions_path.exists() else {}
    )

    tasks = []
    for task_id, body in challenges.items():
        tasks.append(
            Task(
                task_id=task_id,
                train=[Pair(p["input"], p["output"]) for p in body["train"]],
                test_inputs=[t["input"] for t in body["test"]],
                test_outputs=solutions.get(task_id),
            )
        )
    log.info("loaded %d %s tasks (solutions: %s)", len(tasks), split, bool(solutions))
    return tasks


# --- submission ---------------------------------------------------------------

def build_submission(
    attempts: dict[str, list[tuple[Grid, Grid]]], out_path: Path
) -> None:
    """Write submission JSON: task_id -> [{attempt_1, attempt_2}, ...] per test input."""
    payload = {
        task_id: [{"attempt_1": a1, "attempt_2": a2} for a1, a2 in per_input]
        for task_id, per_input in attempts.items()
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    log.info("wrote %s (%d tasks)", out_path, len(payload))


# --- scoring ------------------------------------------------------------------

def score_task(predictions: list[tuple[Grid, Grid]], truths: list[Grid]) -> float:
    """Fraction of this task's test inputs solved by either attempt (pass@2)."""
    if len(predictions) != len(truths):
        raise ValueError(f"got {len(predictions)} predictions for {len(truths)} truths")
    hits = sum(1 for (a1, a2), t in zip(predictions, truths) if a1 == t or a2 == t)
    return hits / len(truths)


def score_solver(solver, tasks: list[Task]) -> float:
    """Mean per-task score of `solver(task) -> [(attempt_1, attempt_2), ...]`.

    Solver exceptions count as a zero for that task (mirrors a wrong submission,
    keeps eval runs robust while we iterate).
    """
    total = 0.0
    for task in tasks:
        if task.test_outputs is None:
            raise ValueError(f"task {task.task_id} has no solutions to score against")
        try:
            total += score_task(solver(task), task.test_outputs)
        except Exception:  # noqa: BLE001 — solver bugs shouldn't kill the eval run
            log.exception("solver failed on %s", task.task_id)
    score = total / len(tasks)
    log.info("score over %d tasks: %.4f", len(tasks), score)
    return score
