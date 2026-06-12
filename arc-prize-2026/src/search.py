"""Program-search solver for ARC — implements §4 of dsl-design.md.

Iterative-deepening BFS over op compositions (depth ≤ 3), with:
  - First-demo short-circuit
  - Shape gate (SHAPE_RULES)
  - State dedup (memoize intermediate grids per depth level)
  - pass@2 strategy with rung-1 fallback

Also exports solve_dsl_obj: rung-2B object-level search (objects-design.md §4).

Exports:
    Program          = tuple[str, ...]   (sequence of op names)
    search_programs  — finds all consistent programs up to max_depth
    solve_dsl        — full solver: pass@2 predictions for a Task
    solve_dsl_obj    — rung-2B: object-aware solver
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Callable

from arc_io import Grid, Task
from dsl import OPS, SHAPE_RULES, build_color_map_op
from solvers import solve_trivial

log = logging.getLogger("search")

Program = tuple[str, ...]  # sequence of op names


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grid_key(grid: Grid) -> tuple[tuple[int, ...], ...]:
    """Hashable representation of a grid."""
    return tuple(tuple(row) for row in grid)


def _apply_seq(
    ops_seq: Program,
    grid: Grid,
    op_callables: dict[str, Callable[[Grid], Grid]],
) -> Grid:
    """Apply each op in sequence."""
    result = grid
    for name in ops_seq:
        result = op_callables[name](result)
    return result


# ---------------------------------------------------------------------------
# Shape gate
# ---------------------------------------------------------------------------

def _predicted_shape_after_seq(
    ops_seq: Program, in_shape: tuple[int, int]
) -> tuple[int, int] | None:
    """Predict output shape. Returns None if undetermined (variable-output ops)."""
    r, c = in_shape
    for name in ops_seq:
        rule = SHAPE_RULES.get(name)
        if rule is None:
            return None
        result = rule(r, c)
        if result is None:
            return None
        r, c = result
    return (r, c)


def _shape_gate_ok(
    ops_seq: Program, demo_pairs: list[tuple[Grid, Grid]]
) -> bool:
    """True if ops_seq could produce the first demo output's shape from its input."""
    if not demo_pairs:
        return True
    inp, out = demo_pairs[0]
    in_shape = (len(inp), len(inp[0]) if inp else 0)
    out_shape = (len(out), len(out[0]) if out else 0)
    pred = _predicted_shape_after_seq(ops_seq, in_shape)
    if pred is None:
        return True  # variable-output — don't gate
    return pred == out_shape


# ---------------------------------------------------------------------------
# Core search: iterative deepening with proper state dedup
# ---------------------------------------------------------------------------

def search_programs(
    task: Task,
    max_depth: int = 3,
) -> list[Program]:
    """Find all programs (op-name sequences) consistent with every demo pair.

    Implements §4: iterative-deepening BFS with first-demo short-circuit,
    shape gate, and state dedup.

    State dedup semantics: at each search depth `d`, for each step position
    `k` (0..d-1), track which intermediate grids on the FIRST demo input have
    already been fully explored (i.e., all length-(d-k-1) continuations have
    been tried). When a new prefix reaches an already-explored intermediate
    state at position k, skip it — its continuations are identical.

    The `infer_color_map` op is injected as a task-parameterized function.

    Returns list of consistent Programs, shortest first.
    """
    demo_pairs: list[tuple[Grid, Grid]] = [
        (pair.input, pair.output) for pair in task.train
    ]
    if not demo_pairs:
        return []

    # Build task-specific color map op
    color_map_fn = build_color_map_op(demo_pairs)

    # Effective ops: override placeholder
    effective_ops: dict[str, Callable[[Grid], Grid]] = dict(OPS)
    effective_ops["infer_color_map"] = color_map_fn

    op_names = list(effective_ops.keys())
    n_ops = len(op_names)

    first_inp, first_out = demo_pairs[0]
    first_out_key = _grid_key(first_out)

    consistent_programs: list[Program] = []

    for depth in range(1, max_depth + 1):
        # For each prefix length k (0..depth-1), track which intermediate
        # grids have already been fully extended with all length-(depth-k)
        # suffixes. Key: (k, grid_key) -> bool.
        # We only add a state here AFTER we've scheduled all extensions from it.
        #
        # Implementation: BFS layer by layer.
        # Layer k holds: list of (prefix_ops_tuple, intermediate_grid_on_first_demo)
        # We extend each by one op to build layer k+1.

        # Layer 0: just the initial grid
        # Each entry: (prefix: tuple[str,...], grid_on_first_demo: Grid)
        current_layer: list[tuple[tuple[str, ...], Grid]] = [((), first_inp)]
        # seen[k] = set of grid_keys that have been enqueued at layer k
        seen: list[set[tuple[tuple[int, ...], ...]]] = [set() for _ in range(depth + 1)]
        init_key = _grid_key(first_inp)
        seen[0].add(init_key)

        for step in range(depth):
            next_layer: list[tuple[tuple[str, ...], Grid]] = []
            next_seen = seen[step + 1]

            for prefix, inter_grid in current_layer:
                for op_name in op_names:
                    new_prefix = prefix + (op_name,)

                    # Shape gate: only applies to final step or if shape-constrained
                    # For intermediate steps, we can't easily gate without knowing
                    # the remaining ops. Instead, gate only when this IS the last step.
                    if step == depth - 1:
                        if not _shape_gate_ok(new_prefix, demo_pairs):
                            continue

                    new_grid = effective_ops[op_name](inter_grid)
                    new_key = _grid_key(new_grid)

                    # State dedup: skip if we've already explored this state at this step
                    if new_key in next_seen:
                        continue
                    next_seen.add(new_key)

                    if step == depth - 1:
                        # This is a complete sequence — check first-demo short-circuit
                        if new_key != first_out_key:
                            continue

                        # Verify ALL demo pairs
                        all_match = True
                        for inp, out in demo_pairs[1:]:
                            result = _apply_seq(new_prefix, inp, effective_ops)
                            if _grid_key(result) != _grid_key(out):
                                all_match = False
                                break

                        if all_match:
                            consistent_programs.append(new_prefix)
                    else:
                        next_layer.append((new_prefix, new_grid))

            current_layer = next_layer

    return consistent_programs


# ---------------------------------------------------------------------------
# pass@2 solver
# ---------------------------------------------------------------------------

def solve_dsl(task: Task) -> list[tuple[Grid, Grid]]:
    """Solve a task using DSL program search with pass@2 strategy.

    pass@2 strategy (per §4):
      1. Run search_programs to get all consistent programs.
      2. Group by their prediction on each test input.
      3. Order groups by (shortest program length in group, then group size desc).
      4. Attempt 1 = top group's prediction; Attempt 2 = second group's prediction.
      5. If fewer than 2 groups for any test input, fall back to rung-1 (solve_trivial).

    Returns list of (attempt_1, attempt_2) per test input.
    """
    programs = search_programs(task, max_depth=3)

    if not programs:
        log.debug("task %s: no programs found, using trivial fallback", task.task_id)
        return solve_trivial(task)

    log.info("task %s: found %d consistent programs", task.task_id, len(programs))

    # Build task-specific op callables (same as search does internally)
    demo_pairs_raw = [(pair.input, pair.output) for pair in task.train]
    color_map_fn = build_color_map_op(demo_pairs_raw)
    effective_ops: dict[str, Callable[[Grid], Grid]] = dict(OPS)
    effective_ops["infer_color_map"] = color_map_fn

    trivial_preds = solve_trivial(task)
    predictions: list[tuple[Grid, Grid]] = []

    for test_idx, test_input in enumerate(task.test_inputs):
        # Generate predictions from all consistent programs
        prog_predictions: list[tuple[Grid, Program]] = []
        for prog in programs:
            pred = _apply_seq(prog, test_input, effective_ops)
            prog_predictions.append((pred, prog))

        # Group by predicted grid
        groups: dict[tuple[tuple[int, ...], ...], list[Program]] = {}
        group_grid: dict[tuple[tuple[int, ...], ...], Grid] = {}
        for pred_grid, prog in prog_predictions:
            key = _grid_key(pred_grid)
            if key not in groups:
                groups[key] = []
                group_grid[key] = pred_grid
            groups[key].append(prog)

        # Sort groups: primary = shortest program in group (asc),
        # secondary = group size (desc)
        def group_sort_key(k: tuple[tuple[int, ...], ...]) -> tuple[int, int]:
            progs = groups[k]
            min_len = min(len(p) for p in progs)
            return (min_len, -len(progs))

        sorted_keys = sorted(groups.keys(), key=group_sort_key)

        attempt_1 = group_grid[sorted_keys[0]]

        if len(sorted_keys) >= 2:
            attempt_2 = group_grid[sorted_keys[1]]
        else:
            # Only one hypothesis group — use rung-1 as attempt 2
            attempt_2 = trivial_preds[test_idx][1]

        predictions.append((attempt_1, attempt_2))

        # Log the winning program
        winning_prog = groups[sorted_keys[0]][0]
        log.info(
            "task %s test[%d]: program=%s",
            task.task_id,
            test_idx,
            " -> ".join(winning_prog),
        )

    return predictions


# ---------------------------------------------------------------------------
# Rung 2B: object-level search (objects-design.md §4)
# ---------------------------------------------------------------------------

def _is_object_op(name: str) -> bool:
    """True if this op is an object-level closure (from objects.py)."""
    return name.startswith("obj_")


def _search_obj_programs(
    task: Task,
    effective_ops: dict[str, Callable[[Grid], Grid]],
    time_budget: float = 4.0,
) -> list[Program]:
    """Search for programs using the family-aware depth budget from §4.2.

    Budget rules:
    - Depth 1 and 2: search over all ops (whole-grid + object) — full BFS.
    - Depth 3: restricted to mixed W/O patterns only (§4.2):
        WOW, OWW, WWO  (W=whole-grid op, O=object op)
      OWO and OOO are excluded (too slow; rare empirically).
    - Hard wall: stop depth-3 patterns if time_budget seconds elapsed.

    First-demo short-circuit, state dedup apply throughout.
    Shape gate applied at depth 1/2 final step (not depth 3: pattern enforces it).
    """
    import time as _time

    demo_pairs: list[tuple[Grid, Grid]] = [
        (pair.input, pair.output) for pair in task.train
    ]
    if not demo_pairs:
        return []

    op_names = list(effective_ops.keys())
    grid_op_names = [n for n in op_names if not _is_object_op(n)]
    obj_op_names = [n for n in op_names if _is_object_op(n)]

    first_inp, first_out = demo_pairs[0]
    first_out_key = _grid_key(first_out)

    consistent_programs: list[Program] = []
    t_start = _time.monotonic()

    def _timed_out() -> bool:
        return _time.monotonic() - t_start > time_budget

    def _verify_all(prefix: Program) -> bool:
        """Check prefix against all demo pairs."""
        for inp, out in demo_pairs[1:]:
            result = _apply_seq(prefix, inp, effective_ops)
            if _grid_key(result) != _grid_key(out):
                return False
        return True

    # -----------------------------------------------------------------------
    # Depth 1 and 2: standard BFS with shape gate and dedup
    # -----------------------------------------------------------------------
    for depth in range(1, 3):
        if _timed_out():
            return consistent_programs

        current_layer: list[tuple[tuple[str, ...], Grid]] = [((), first_inp)]
        seen: list[set[tuple[tuple[int, ...], ...]]] = [set() for _ in range(depth + 1)]
        seen[0].add(_grid_key(first_inp))

        for step in range(depth):
            if _timed_out():
                return consistent_programs

            next_layer: list[tuple[tuple[str, ...], Grid]] = []
            next_seen = seen[step + 1]
            eval_count = 0

            for prefix, inter_grid in current_layer:
                for op_name in op_names:
                    eval_count += 1
                    if eval_count % 500 == 0 and _timed_out():
                        return consistent_programs

                    new_prefix = prefix + (op_name,)

                    # Shape gate only at final step
                    if step == depth - 1:
                        if not _shape_gate_ok(new_prefix, demo_pairs):
                            continue

                    new_grid = effective_ops[op_name](inter_grid)
                    new_key = _grid_key(new_grid)

                    if new_key in next_seen:
                        continue
                    next_seen.add(new_key)

                    if step == depth - 1:
                        if new_key != first_out_key:
                            continue
                        if _verify_all(new_prefix):
                            consistent_programs.append(new_prefix)
                    else:
                        next_layer.append((new_prefix, new_grid))

            current_layer = next_layer

    # -----------------------------------------------------------------------
    # Depth 3: restricted patterns only (§4.2)
    # Enumerate 3 patterns: WOW, OWW, WWO
    # Skip if already over time budget.
    # -----------------------------------------------------------------------

    def _search_pattern(
        name_sets: list[list[str]],
    ) -> bool:
        """BFS over a 3-step pattern. Returns False if timed out mid-search."""
        current: list[tuple[tuple[str, ...], Grid]] = [((), first_inp)]
        step_seen: list[set[tuple[tuple[int, ...], ...]]] = [set() for _ in range(4)]
        step_seen[0].add(_grid_key(first_inp))

        for step, allowed_names in enumerate(name_sets):
            if _timed_out():
                return False

            nxt: list[tuple[tuple[str, ...], Grid]] = []
            nxt_seen = step_seen[step + 1]
            is_last = (step == len(name_sets) - 1)
            eval_count = 0

            for prefix, inter_grid in current:
                for op_name in allowed_names:
                    # Check time every 200 evaluations to avoid overhead
                    eval_count += 1
                    if eval_count % 200 == 0 and _timed_out():
                        return False

                    new_prefix = prefix + (op_name,)
                    new_grid = effective_ops[op_name](inter_grid)
                    new_key = _grid_key(new_grid)

                    if new_key in nxt_seen:
                        continue
                    nxt_seen.add(new_key)

                    if is_last:
                        if new_key != first_out_key:
                            continue
                        if _verify_all(new_prefix):
                            consistent_programs.append(new_prefix)
                    else:
                        nxt.append((new_prefix, new_grid))

            current = nxt
        return True

    if not _timed_out():
        # WOW: whole-grid, object, whole-grid
        _search_pattern([grid_op_names, obj_op_names, grid_op_names])
    if not _timed_out():
        # OWW: object, whole-grid, whole-grid
        _search_pattern([obj_op_names, grid_op_names, grid_op_names])
    if not _timed_out():
        # WWO: whole-grid, whole-grid, object
        _search_pattern([grid_op_names, grid_op_names, obj_op_names])

    return consistent_programs


def _make_pass2_predictions(
    task: Task,
    programs: list[Program],
    effective_ops: dict[str, Callable[[Grid], Grid]],
    fallback_preds: list[tuple[Grid, Grid]],
    solver_label: str,
) -> list[tuple[Grid, Grid]]:
    """Apply pass@2 grouping strategy — shared by solve_dsl and solve_dsl_obj."""
    predictions: list[tuple[Grid, Grid]] = []

    for test_idx, test_input in enumerate(task.test_inputs):
        prog_predictions: list[tuple[Grid, Program]] = []
        for prog in programs:
            pred = _apply_seq(prog, test_input, effective_ops)
            prog_predictions.append((pred, prog))

        groups: dict[tuple[tuple[int, ...], ...], list[Program]] = {}
        group_grid: dict[tuple[tuple[int, ...], ...], Grid] = {}
        for pred_grid, prog in prog_predictions:
            key = _grid_key(pred_grid)
            if key not in groups:
                groups[key] = []
                group_grid[key] = pred_grid
            groups[key].append(prog)

        def group_sort_key(k: tuple[tuple[int, ...], ...]) -> tuple[int, int]:
            progs = groups[k]
            min_len = min(len(p) for p in progs)
            return (min_len, -len(progs))

        sorted_keys = sorted(groups.keys(), key=group_sort_key)

        attempt_1 = group_grid[sorted_keys[0]]
        attempt_2 = (
            group_grid[sorted_keys[1]]
            if len(sorted_keys) >= 2
            else fallback_preds[test_idx][1]
        )

        predictions.append((attempt_1, attempt_2))

        winning_prog = groups[sorted_keys[0]][0]
        log.info(
            "task %s test[%d] [%s]: program=%s",
            task.task_id,
            test_idx,
            solver_label,
            " -> ".join(winning_prog),
        )

    return predictions


def solve_dsl_obj(task: Task) -> list[tuple[Grid, Grid]]:
    """Rung 2B: object-aware solver with family-aware depth budget.

    Extends solve_dsl by adding ~126 object-level closures (generate_object_ops).
    Uses the segmentation-consistency gate (filter_segmentations) to discard
    irrelevant segmentation ops before search.

    Search budget:
    - Depth 1-2: full op set (whole-grid + gated object ops)
    - Depth 3: restricted patterns only (WOW, OWW, WWO, OWO) per §4.2

    Falls back to solve_dsl if no object programs found, then to rung-1.
    Returns pass@2 predictions.
    """
    from objects import generate_object_ops, filter_segmentations, filter_object_ops

    demo_pairs_raw = [(pair.input, pair.output) for pair in task.train]

    # Build task-specific color map op
    color_map_fn = build_color_map_op(demo_pairs_raw)
    effective_ops: dict[str, Callable[[Grid], Grid]] = dict(OPS)
    effective_ops["infer_color_map"] = color_map_fn

    # Generate and gate object ops
    all_obj_ops = generate_object_ops()
    kept_segs = filter_segmentations(demo_pairs_raw)
    gated_obj_ops = filter_object_ops(all_obj_ops, kept_segs)

    log.debug(
        "task %s: %d/%d obj ops after gate (segs kept: %s)",
        task.task_id,
        len(gated_obj_ops),
        len(all_obj_ops),
        kept_segs,
    )

    # Combined op set: DSL ops + gated object ops
    combined_ops: dict[str, Callable[[Grid], Grid]] = dict(effective_ops)
    combined_ops.update(gated_obj_ops)

    import time as _time
    t0_total = _time.monotonic()
    TOTAL_BUDGET = 4.5  # seconds per task (leaves margin for overhead + predictions)

    # Depth-1 and depth-2 are fast (<1s). Depth-3 patterns are gated by budget.
    # 2.0s gives enough time for depth-3 WOW/OWW/WWO patterns with dedup.
    programs = _search_obj_programs(task, combined_ops, time_budget=2.0)

    trivial_preds = solve_trivial(task)

    if not programs:
        log.debug(
            "task %s: no obj programs — trying plain dsl fallback", task.task_id
        )
        # Only run DSL fallback if we have remaining budget.
        # Use max_depth=2 to stay within time budget; depth-3 DSL is slow and
        # already covered by the depth-3 patterns in the object search above.
        remaining = TOTAL_BUDGET - (_time.monotonic() - t0_total)
        if remaining > 0.3:
            dsl_max_depth = 3 if remaining > 1.5 else 2
            dsl_programs = search_programs(task, max_depth=dsl_max_depth)
            if dsl_programs:
                log.info(
                    "task %s: dsl fallback found %d programs (depth≤%d)",
                    task.task_id, len(dsl_programs), dsl_max_depth,
                )
                return _make_pass2_predictions(
                    task, dsl_programs, effective_ops, trivial_preds, "dsl_fallback"
                )
        log.debug("task %s: no programs found at all, using trivial", task.task_id)
        return trivial_preds

    log.info("task %s: found %d obj programs", task.task_id, len(programs))
    return _make_pass2_predictions(
        task, programs, combined_ops, trivial_preds, "dsl_obj"
    )
