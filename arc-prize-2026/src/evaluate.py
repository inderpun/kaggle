"""Evaluate a solver against the 120 public evaluation tasks (pass@2, exact match).

This is the local stand-in for the leaderboard: same metric, same two-attempt
rule, so numbers here are directly comparable to a submission score.

Run:
    python3 src/evaluate.py [solver_name]   # default: trivial
    python3 src/evaluate.py dsl_obj [training|evaluation]
"""
from __future__ import annotations

import logging
import sys
import time

import solvers
from arc_io import load_tasks, score_solver, score_task

log = logging.getLogger("evaluate")

SOLVERS = {
    "trivial": solvers.solve_trivial,
}

try:
    from search import solve_dsl  # noqa: PLC0415
    SOLVERS["dsl"] = solve_dsl
except ImportError:
    pass  # search.py not yet present — skip registration

try:
    from search import solve_dsl_obj  # noqa: PLC0415
    SOLVERS["dsl_obj"] = solve_dsl_obj
except ImportError:
    pass  # objects.py not yet present — skip registration


def main() -> None:
    args = sys.argv[1:]
    name = args[0] if args else "trivial"
    split = args[1] if len(args) > 1 else "evaluation"

    if name not in SOLVERS:
        log.error("unknown solver %r — choose from %s", name, sorted(SOLVERS))
        sys.exit(1)

    tasks = load_tasks(split)
    solver = SOLVERS[name]

    solved_tasks: list[str] = []
    total = 0.0
    t0 = time.time()

    for task in tasks:
        if task.test_outputs is None:
            log.warning("task %s has no solutions, skipping", task.task_id)
            continue
        try:
            preds = solver(task)
            s = score_task(preds, task.test_outputs)
            total += s
            if s > 0:
                solved_tasks.append(task.task_id)
        except Exception:
            log.exception("solver failed on %s", task.task_id)

    elapsed = time.time() - t0
    n = len(tasks)
    score = total / n if n > 0 else 0.0

    print(
        f"solver={name}  split={split}  eval_pass@2={score:.4f}"
        f"  ({score * n:.0f}/{n} tasks)  elapsed={elapsed:.1f}s"
    )
    if solved_tasks:
        print(f"solved task IDs ({len(solved_tasks)}): {' '.join(solved_tasks)}")


if __name__ == "__main__":
    main()
