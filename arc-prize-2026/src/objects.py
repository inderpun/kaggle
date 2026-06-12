"""Object-level reasoning for ARC — rung 2B (objects-design.md §2, §3, §6).

Exports:
    Obj                         — frozen dataclass representing one object
    background_color            — infer background per grid
    objects_4conn_same_color    — segment via 4-connectivity, one color per object
    objects_8conn_same_color    — segment via 8-connectivity, one color per object
    objects_multicolor_4conn    — segment via 4-connectivity, any non-background color
    generate_object_ops         — returns ~110 Grid->Grid closures keyed by name
    filter_segmentations        — segmentation-consistency gate (§4.1)

All functions are pure and total (never raise; return sensible defaults).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from arc_io import Grid


# ---------------------------------------------------------------------------
# Obj dataclass (§2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Obj:
    """An object extracted from an ARC grid.

    Attributes:
        cells: Absolute (row, col) positions of every cell in the object.
        color: Dominant (most frequent) color among the cells.
    """
    cells: frozenset[tuple[int, int]]
    color: int

    # Derived properties (computed lazily via methods — kept off the dataclass
    # to keep the frozen hash fast and the repr clean).

    def size(self) -> int:
        """Number of cells."""
        return len(self.cells)

    def bbox(self) -> tuple[int, int, int, int]:
        """(min_row, min_col, max_row, max_col) bounding box."""
        rows = [r for r, c in self.cells]
        cols = [c for r, c in self.cells]
        return (min(rows), min(cols), max(rows), max(cols))

    def height(self) -> int:
        r0, c0, r1, c1 = self.bbox()
        return r1 - r0 + 1

    def width(self) -> int:
        r0, c0, r1, c1 = self.bbox()
        return c1 - c0 + 1

    def normalized_shape(self) -> frozenset[tuple[int, int]]:
        """Cells translated so top-left is (0, 0)."""
        r0, c0, _, _ = self.bbox()
        return frozenset((r - r0, c - c0) for r, c in self.cells)


# ---------------------------------------------------------------------------
# Background detection
# ---------------------------------------------------------------------------

def background_color(grid: Grid) -> int:
    """Return the most frequent color (background); ties broken by lowest value."""
    counts: Counter[int] = Counter(v for row in grid for v in row)
    if not counts:
        return 0
    max_count = max(counts.values())
    candidates = sorted(c for c, n in counts.items() if n == max_count)
    return candidates[0]


# ---------------------------------------------------------------------------
# Segmentation helpers
# ---------------------------------------------------------------------------

def _flood_fill_4(
    grid: Grid,
    start_r: int,
    start_c: int,
    visited: set[tuple[int, int]],
    match_fn: Callable[[int, int], bool],
) -> list[tuple[int, int]]:
    """4-connected flood fill returning all matched cells from start."""
    R = len(grid)
    C = len(grid[0]) if grid else 0
    stack = [(start_r, start_c)]
    component: list[tuple[int, int]] = []
    while stack:
        r, c = stack.pop()
        if (r, c) in visited:
            continue
        if r < 0 or r >= R or c < 0 or c >= C:
            continue
        if not match_fn(r, c):
            continue
        visited.add((r, c))
        component.append((r, c))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) not in visited:
                stack.append((nr, nc))
    return component


def _flood_fill_8(
    grid: Grid,
    start_r: int,
    start_c: int,
    visited: set[tuple[int, int]],
    match_fn: Callable[[int, int], bool],
) -> list[tuple[int, int]]:
    """8-connected flood fill returning all matched cells from start."""
    R = len(grid)
    C = len(grid[0]) if grid else 0
    stack = [(start_r, start_c)]
    component: list[tuple[int, int]] = []
    while stack:
        r, c = stack.pop()
        if (r, c) in visited:
            continue
        if r < 0 or r >= R or c < 0 or c >= C:
            continue
        if not match_fn(r, c):
            continue
        visited.add((r, c))
        component.append((r, c))
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited:
                    stack.append((nr, nc))
    return component


def _obj_from_cells(grid: Grid, cells: list[tuple[int, int]]) -> Obj:
    """Build an Obj from a list of (row, col) cells — dominant color is computed."""
    counts: Counter[int] = Counter(grid[r][c] for r, c in cells)
    color = counts.most_common(1)[0][0] if counts else 0
    return Obj(cells=frozenset(cells), color=color)


# ---------------------------------------------------------------------------
# The 3 segmentation functions (§2)
# ---------------------------------------------------------------------------

def objects_4conn_same_color(grid: Grid) -> list[Obj]:
    """Segment via 4-connectivity; each object is a single-color connected component.

    Background (most-frequent color) cells are excluded.
    """
    if not grid or not grid[0]:
        return []
    bg = background_color(grid)
    R, C = len(grid), len(grid[0])
    visited: set[tuple[int, int]] = set()
    objects: list[Obj] = []
    for r in range(R):
        for c in range(C):
            if (r, c) in visited or grid[r][c] == bg:
                continue
            color = grid[r][c]
            cells = _flood_fill_4(
                grid, r, c, visited,
                lambda rr, cc, col=color: grid[rr][cc] == col,
            )
            if cells:
                objects.append(Obj(cells=frozenset(cells), color=color))
    return objects


def objects_8conn_same_color(grid: Grid) -> list[Obj]:
    """Segment via 8-connectivity; each object is a single-color connected component.

    Background (most-frequent color) cells are excluded.
    """
    if not grid or not grid[0]:
        return []
    bg = background_color(grid)
    R, C = len(grid), len(grid[0])
    visited: set[tuple[int, int]] = set()
    objects: list[Obj] = []
    for r in range(R):
        for c in range(C):
            if (r, c) in visited or grid[r][c] == bg:
                continue
            color = grid[r][c]
            cells = _flood_fill_8(
                grid, r, c, visited,
                lambda rr, cc, col=color: grid[rr][cc] == col,
            )
            if cells:
                objects.append(Obj(cells=frozenset(cells), color=color))
    return objects


def objects_multicolor_4conn(grid: Grid) -> list[Obj]:
    """Segment via 4-connectivity; any non-background cell counts (composite objects).

    Objects can contain multiple colors; color is set to the dominant one.
    """
    if not grid or not grid[0]:
        return []
    bg = background_color(grid)
    R, C = len(grid), len(grid[0])
    visited: set[tuple[int, int]] = set()
    objects: list[Obj] = []
    for r in range(R):
        for c in range(C):
            if (r, c) in visited or grid[r][c] == bg:
                continue
            cells = _flood_fill_4(
                grid, r, c, visited,
                lambda rr, cc: grid[rr][cc] != bg,
            )
            if cells:
                objects.append(_obj_from_cells(grid, cells))
    return objects


# All three segmentation functions, keyed by name
SEGMENTATIONS: dict[str, Callable[[Grid], list[Obj]]] = {
    "4conn": objects_4conn_same_color,
    "8conn": objects_8conn_same_color,
    "multi": objects_multicolor_4conn,
}


# ---------------------------------------------------------------------------
# Selectors (§3) — pick one Obj from a list
# ---------------------------------------------------------------------------

def _sel_largest(objs: list[Obj]) -> Obj | None:
    """Select the object with the most cells."""
    if not objs:
        return None
    return max(objs, key=lambda o: o.size())


def _sel_smallest(objs: list[Obj]) -> Obj | None:
    """Select the object with the fewest cells."""
    if not objs:
        return None
    return min(objs, key=lambda o: o.size())


def _sel_most_common_shape(objs: list[Obj]) -> Obj | None:
    """Select an object whose normalized shape is the most frequent."""
    if not objs:
        return None
    shape_counts: Counter[frozenset[tuple[int, int]]] = Counter(
        o.normalized_shape() for o in objs
    )
    most_common_shape = shape_counts.most_common(1)[0][0]
    # Return the first object with that shape
    for o in objs:
        if o.normalized_shape() == most_common_shape:
            return o
    return None


def _sel_unique_shape(objs: list[Obj]) -> Obj | None:
    """Select the object whose normalized shape is unique (appears exactly once)."""
    if not objs:
        return None
    shape_counts: Counter[frozenset[tuple[int, int]]] = Counter(
        o.normalized_shape() for o in objs
    )
    unique_shapes = {s for s, n in shape_counts.items() if n == 1}
    for o in objs:
        if o.normalized_shape() in unique_shapes:
            return o
    return None


def _sel_unique_color(objs: list[Obj]) -> Obj | None:
    """Select the object whose color is unique among all objects."""
    if not objs:
        return None
    color_counts: Counter[int] = Counter(o.color for o in objs)
    unique_colors = {c for c, n in color_counts.items() if n == 1}
    for o in objs:
        if o.color in unique_colors:
            return o
    return None


def _sel_topmost(objs: list[Obj]) -> Obj | None:
    """Select the object whose topmost row is highest (smallest row index)."""
    if not objs:
        return None
    return min(objs, key=lambda o: o.bbox()[0])


def _sel_bottommost(objs: list[Obj]) -> Obj | None:
    """Select the object whose bottommost row is lowest (largest row index)."""
    if not objs:
        return None
    return max(objs, key=lambda o: o.bbox()[2])


def _sel_leftmost(objs: list[Obj]) -> Obj | None:
    """Select the object whose leftmost column is smallest."""
    if not objs:
        return None
    return min(objs, key=lambda o: o.bbox()[1])


def _sel_rightmost(objs: list[Obj]) -> Obj | None:
    """Select the object whose rightmost column is largest."""
    if not objs:
        return None
    return max(objs, key=lambda o: o.bbox()[3])


SELECTORS: dict[str, Callable[[list[Obj]], Obj | None]] = {
    "largest": _sel_largest,
    "smallest": _sel_smallest,
    "most_common_shape": _sel_most_common_shape,
    "unique_shape": _sel_unique_shape,
    "unique_color": _sel_unique_color,
    "topmost": _sel_topmost,
    "bottommost": _sel_bottommost,
    "leftmost": _sel_leftmost,
    "rightmost": _sel_rightmost,
}


# ---------------------------------------------------------------------------
# Actions on a selected object (§3) — returns Grid -> Grid
# ---------------------------------------------------------------------------

def _action_keep_only(
    grid: Grid,
    obj: Obj,
    seg_fn: Callable[[Grid], list[Obj]],
) -> Grid:
    """Keep only the selected object; erase all other non-background cells."""
    if not grid or not grid[0]:
        return [row[:] for row in grid]
    bg = background_color(grid)
    R, C = len(grid), len(grid[0])
    result = [[bg] * C for _ in range(R)]
    for r, c in obj.cells:
        result[r][c] = grid[r][c]
    return result


def _action_remove(
    grid: Grid,
    obj: Obj,
    seg_fn: Callable[[Grid], list[Obj]],
) -> Grid:
    """Remove the selected object (set its cells to background)."""
    if not grid or not grid[0]:
        return [row[:] for row in grid]
    bg = background_color(grid)
    result = [row[:] for row in grid]
    for r, c in obj.cells:
        result[r][c] = bg
    return result


def _action_crop_to(
    grid: Grid,
    obj: Obj,
    seg_fn: Callable[[Grid], list[Obj]],
) -> Grid:
    """Return the bounding box of the selected object."""
    r0, c0, r1, c1 = obj.bbox()
    return [grid[r][c0:c1 + 1] for r in range(r0, r1 + 1)]


def _action_recolor_to_dominant(
    grid: Grid,
    obj: Obj,
    seg_fn: Callable[[Grid], list[Obj]],
) -> Grid:
    """Recolor the entire grid to the selected object's color (uniform fill of non-bg)."""
    if not grid or not grid[0]:
        return [row[:] for row in grid]
    bg = background_color(grid)
    R, C = len(grid), len(grid[0])
    result = [[bg] * C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            if grid[r][c] != bg:
                result[r][c] = obj.color
    return result


# ---------------------------------------------------------------------------
# Whole-decomposition transforms (§3)
# ---------------------------------------------------------------------------

def _recolor_by_size_rank(
    grid: Grid,
    seg_fn: Callable[[Grid], list[Obj]],
) -> Grid:
    """Recolor objects by their size rank.

    Assigns: rank-1 (largest) object gets color 1, rank-2 gets color 2, etc.
    This is a proxy for the spec's "inferred mapping from demo outputs" — since
    we can't safely infer the exact target colors without demo outputs here,
    we use a canonical rank→color assignment (rank index as color 1..N).
    The search will find tasks where this canonical mapping matches.
    """
    if not grid or not grid[0]:
        return [row[:] for row in grid]
    bg = background_color(grid)
    R, C = len(grid), len(grid[0])
    objs = seg_fn(grid)
    if not objs:
        return [row[:] for row in grid]

    # Sort by size descending to assign rank
    sorted_objs = sorted(objs, key=lambda o: o.size(), reverse=True)
    result = [[bg] * C for _ in range(R)]
    for rank, obj in enumerate(sorted_objs, start=1):
        new_color = rank % 10  # keep colors in 0-9 range
        for r, c in obj.cells:
            result[r][c] = new_color
    return result


def _move_all_to_gravity(
    grid: Grid,
    seg_fn: Callable[[Grid], list[Obj]],
    direction: str = "down",
) -> Grid:
    """Move all objects as units in the given direction until they hit a wall or another object.

    direction: 'down' | 'up' | 'left' | 'right'
    """
    if not grid or not grid[0]:
        return [row[:] for row in grid]
    bg = background_color(grid)
    R, C = len(grid), len(grid[0])
    objs = seg_fn(grid)
    if not objs:
        return [row[:] for row in grid]

    result = [[bg] * C for _ in range(R)]

    if direction == "down":
        # Process objects from bottom to top so lower ones settle first
        sorted_objs = sorted(objs, key=lambda o: -o.bbox()[2])
        occupied: set[tuple[int, int]] = set()
        for obj in sorted_objs:
            # Find how far we can move this object down
            max_shift = R
            for r, c in obj.cells:
                # How many rows can we go down?
                for shift in range(1, R):
                    nr = r + shift
                    if nr >= R or (nr, c) in occupied:
                        max_shift = min(max_shift, shift - 1)
                        break
                else:
                    max_shift = min(max_shift, R - 1 - r)
            for r, c in obj.cells:
                new_r = r + max_shift
                occupied.add((new_r, c))
                result[new_r][c] = grid[r][c]

    elif direction == "up":
        sorted_objs = sorted(objs, key=lambda o: o.bbox()[0])
        occupied: set[tuple[int, int]] = set()
        for obj in sorted_objs:
            max_shift = R
            for r, c in obj.cells:
                for shift in range(1, R):
                    nr = r - shift
                    if nr < 0 or (nr, c) in occupied:
                        max_shift = min(max_shift, shift - 1)
                        break
                else:
                    max_shift = min(max_shift, r)
            for r, c in obj.cells:
                new_r = r - max_shift
                occupied.add((new_r, c))
                result[new_r][c] = grid[r][c]

    elif direction == "left":
        sorted_objs = sorted(objs, key=lambda o: o.bbox()[1])
        occupied: set[tuple[int, int]] = set()
        for obj in sorted_objs:
            max_shift = C
            for r, c in obj.cells:
                for shift in range(1, C):
                    nc = c - shift
                    if nc < 0 or (r, nc) in occupied:
                        max_shift = min(max_shift, shift - 1)
                        break
                else:
                    max_shift = min(max_shift, c)
            for r, c in obj.cells:
                new_c = c - max_shift
                occupied.add((r, new_c))
                result[r][new_c] = grid[r][c]

    elif direction == "right":
        sorted_objs = sorted(objs, key=lambda o: -o.bbox()[3])
        occupied: set[tuple[int, int]] = set()
        for obj in sorted_objs:
            max_shift = C
            for r, c in obj.cells:
                for shift in range(1, C):
                    nc = c + shift
                    if nc >= C or (r, nc) in occupied:
                        max_shift = min(max_shift, shift - 1)
                        break
                else:
                    max_shift = min(max_shift, C - 1 - c)
            for r, c in obj.cells:
                new_c = c + max_shift
                occupied.add((r, new_c))
                result[r][new_c] = grid[r][c]

    return result


def _count_objects_as_grid(
    grid: Grid,
    seg_fn: Callable[[Grid], list[Obj]],
) -> Grid:
    """Encode the object count as a 1×N strip of 1s (N = number of objects).

    Returns a 1-row grid of length N with all cells set to 1.
    If no objects found, returns [[0]].
    """
    objs = seg_fn(grid)
    n = len(objs)
    if n == 0:
        return [[0]]
    return [[1] * n]


# ---------------------------------------------------------------------------
# Op generation (§3, §6) — generates ~110 Grid->Grid closures
# ---------------------------------------------------------------------------

def generate_object_ops() -> dict[str, Callable[[Grid], Grid]]:
    """Generate all object-parameterized ops as Grid->Grid closures.

    Returns dict keyed by descriptive name.
    3 segmentations × 9 selectors × 4 actions = 108 selector-action ops
    + 3 segmentations × 3 whole-decomp transforms = 9 transform ops
    = 117 ops total.
    """
    ops: dict[str, Callable[[Grid], Grid]] = {}

    for seg_name, seg_fn in SEGMENTATIONS.items():

        # --- selector × action closures ---
        for sel_name, sel_fn in SELECTORS.items():

            # keep_only
            def make_keep_only(
                sf: Callable[[Grid], list[Obj]] = seg_fn,
                slct: Callable[[list[Obj]], Obj | None] = sel_fn,
            ) -> Callable[[Grid], Grid]:
                def op(grid: Grid) -> Grid:
                    objs = sf(grid)
                    obj = slct(objs)
                    if obj is None:
                        return [row[:] for row in grid]
                    return _action_keep_only(grid, obj, sf)
                return op

            key = f"obj_{seg_name}_{sel_name}_keep_only"
            ops[key] = make_keep_only()

            # remove
            def make_remove(
                sf: Callable[[Grid], list[Obj]] = seg_fn,
                slct: Callable[[list[Obj]], Obj | None] = sel_fn,
            ) -> Callable[[Grid], Grid]:
                def op(grid: Grid) -> Grid:
                    objs = sf(grid)
                    obj = slct(objs)
                    if obj is None:
                        return [row[:] for row in grid]
                    return _action_remove(grid, obj, sf)
                return op

            key = f"obj_{seg_name}_{sel_name}_remove"
            ops[key] = make_remove()

            # crop_to
            def make_crop_to(
                sf: Callable[[Grid], list[Obj]] = seg_fn,
                slct: Callable[[list[Obj]], Obj | None] = sel_fn,
            ) -> Callable[[Grid], Grid]:
                def op(grid: Grid) -> Grid:
                    objs = sf(grid)
                    obj = slct(objs)
                    if obj is None:
                        return [row[:] for row in grid]
                    return _action_crop_to(grid, obj, sf)
                return op

            key = f"obj_{seg_name}_{sel_name}_crop_to"
            ops[key] = make_crop_to()

            # recolor_to_dominant
            def make_recolor(
                sf: Callable[[Grid], list[Obj]] = seg_fn,
                slct: Callable[[list[Obj]], Obj | None] = sel_fn,
            ) -> Callable[[Grid], Grid]:
                def op(grid: Grid) -> Grid:
                    objs = sf(grid)
                    obj = slct(objs)
                    if obj is None:
                        return [row[:] for row in grid]
                    return _action_recolor_to_dominant(grid, obj, sf)
                return op

            key = f"obj_{seg_name}_{sel_name}_recolor_to_dominant"
            ops[key] = make_recolor()

        # --- whole-decomposition transforms ---

        def make_recolor_by_size(
            sf: Callable[[Grid], list[Obj]] = seg_fn,
        ) -> Callable[[Grid], Grid]:
            def op(grid: Grid) -> Grid:
                return _recolor_by_size_rank(grid, sf)
            return op

        ops[f"obj_{seg_name}_recolor_by_size_rank"] = make_recolor_by_size()

        def make_gravity(
            sf: Callable[[Grid], list[Obj]] = seg_fn,
            direction: str = "down",
        ) -> Callable[[Grid], Grid]:
            def op(grid: Grid) -> Grid:
                return _move_all_to_gravity(grid, sf, direction)
            return op

        ops[f"obj_{seg_name}_move_gravity_down"] = make_gravity(seg_fn, "down")
        ops[f"obj_{seg_name}_move_gravity_up"] = make_gravity(seg_fn, "up")
        ops[f"obj_{seg_name}_move_gravity_left"] = make_gravity(seg_fn, "left")
        ops[f"obj_{seg_name}_move_gravity_right"] = make_gravity(seg_fn, "right")

        def make_count(
            sf: Callable[[Grid], list[Obj]] = seg_fn,
        ) -> Callable[[Grid], Grid]:
            def op(grid: Grid) -> Grid:
                return _count_objects_as_grid(grid, sf)
            return op

        ops[f"obj_{seg_name}_count_objects_as_grid"] = make_count()

    return ops


# ---------------------------------------------------------------------------
# Segmentation-consistency gate (§4.1)
# ---------------------------------------------------------------------------

def _seg_signature(grid: Grid, seg_fn: Callable[[Grid], list[Obj]]) -> tuple[int, frozenset[int]]:
    """(object_count, set_of_colors) for one grid under one segmentation."""
    objs = seg_fn(grid)
    return (len(objs), frozenset(o.color for o in objs))


def filter_segmentations(
    demo_pairs: list[tuple[Grid, Grid]],
    threshold: float = 0.5,
) -> set[str]:
    """Return the set of segmentation names to keep for this task (§4.1).

    For each segmentation, check if the object count/color sets of inputs and
    outputs are 'related' across all demo pairs.  A segmentation is kept if at
    least `threshold` fraction of demo pairs have a meaningful relationship
    (non-trivial count on both sides, or matching color sets).

    If no segmentation passes, keep all (fail-open so we don't lose coverage).
    """
    if not demo_pairs:
        return set(SEGMENTATIONS.keys())

    kept: set[str] = set()

    for seg_name, seg_fn in SEGMENTATIONS.items():
        score = 0.0
        for inp, out in demo_pairs:
            in_count, in_colors = _seg_signature(inp, seg_fn)
            out_count, out_colors = _seg_signature(out, seg_fn)

            # Relationship signals:
            # (a) both sides have >0 objects (segmentation fires on both)
            both_nontrivial = in_count > 0 and out_count > 0
            # (b) colors overlap between input objects and output objects
            color_overlap = bool(in_colors & out_colors)
            # (c) same object count (transformation preserves count)
            same_count = (in_count == out_count) and in_count > 0
            # (d) count difference is small (≤ 2 objects gained/lost)
            close_count = both_nontrivial and abs(in_count - out_count) <= 2

            if both_nontrivial and (color_overlap or same_count or close_count):
                score += 1.0

        frac = score / len(demo_pairs)
        if frac >= threshold:
            kept.add(seg_name)

    if not kept:
        # Fail-open: keep all segmentations
        return set(SEGMENTATIONS.keys())

    return kept


def filter_object_ops(
    ops: dict[str, Callable[[Grid], Grid]],
    kept_segs: set[str],
) -> dict[str, Callable[[Grid], Grid]]:
    """Return only the ops belonging to kept segmentations."""
    filtered: dict[str, Callable[[Grid], Grid]] = {}
    for name, fn in ops.items():
        if not name.startswith("obj_"):
            # Non-object op — always keep
            filtered[name] = fn
            continue
        # Extract segmentation name: "obj_{seg_name}_..."
        rest = name[4:]  # strip "obj_"
        seg = rest.split("_")[0]
        if seg in kept_segs:
            filtered[name] = fn
    return filtered
