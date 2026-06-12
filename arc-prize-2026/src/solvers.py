"""Rung-1 trivial solvers.

These exist to prove the harness, not to score. Each solver maps a Task to a list
of (attempt_1, attempt_2) tuples — one tuple per test input — matching the
submission contract in `arc_io.build_submission`.

Solver signature: `solver(task: Task) -> list[tuple[Grid, Grid]]`.
"""
from __future__ import annotations

from collections import Counter

from arc_io import Grid, Task


def _constant_demo_output(task: Task) -> Grid | None:
    """If every demonstration produces the same output grid, return it."""
    outputs = [tuple(map(tuple, p.output)) for p in task.train]
    if len(set(outputs)) == 1:
        return [list(row) for row in outputs[0]]
    return None


def _majority_output_shape(task: Task) -> tuple[int, int]:
    """Most common (rows, cols) among demo outputs — a crude shape prior."""
    shapes = Counter((len(p.output), len(p.output[0])) for p in task.train)
    return shapes.most_common(1)[0][0]


def solve_trivial(task: Task) -> list[tuple[Grid, Grid]]:
    """Attempt 1: identity (output = input). Attempt 2: constant demo output if
    the demos all agree, else a zero-grid of the majority demo-output shape.

    Identity catches nothing on ARC-AGI-2 by design; the value is exercising the
    full predict -> score -> submit path with structurally valid grids.
    """
    constant = _constant_demo_output(task)
    rows, cols = _majority_output_shape(task)
    fallback: Grid = [[0] * cols for _ in range(rows)]

    predictions: list[tuple[Grid, Grid]] = []
    for test_input in task.test_inputs:
        attempt_1 = [row[:] for row in test_input]          # identity
        attempt_2 = constant if constant is not None else fallback
        predictions.append((attempt_1, attempt_2))
    return predictions
