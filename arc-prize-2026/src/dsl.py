"""ARC DSL — ~30 pure Grid->Grid operations for program-search solving.

All ops are *total functions*: they never raise and return the input unchanged
when the operation is inapplicable. Type hints and one-line docstrings on every op.

Exports:
    OPS         — dict[str, Callable[[Grid], Grid]]
    SHAPE_RULES — dict[str, Callable[[int,int], tuple[int,int]|None]]
                  None means "any output shape acceptable / don't gate".
    build_color_map_op — factory that returns a task-parameterized infer_color_map op.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable

from arc_io import Grid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _copy(grid: Grid) -> Grid:
    """Return a deep copy of the grid."""
    return [row[:] for row in grid]


def _rows(grid: Grid) -> int:
    return len(grid)


def _cols(grid: Grid) -> int:
    return len(grid[0]) if grid else 0


def _dominant_nonzero(grid: Grid) -> int:
    """Return the most frequent non-zero color; 0 if grid is all zeros."""
    counts: Counter[int] = Counter()
    for row in grid:
        for v in row:
            if v != 0:
                counts[v] += 1
    if not counts:
        return 0
    return counts.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Geometry (8 ops) — D8 symmetry group
# ---------------------------------------------------------------------------

def identity(grid: Grid) -> Grid:
    """Pass the grid through unchanged — identity element of composition."""
    return _copy(grid)


def rotate90(grid: Grid) -> Grid:
    """Rotate 90° clockwise — very common ARC symmetry motif."""
    R, C = _rows(grid), _cols(grid)
    if R == 0 or C == 0:
        return _copy(grid)
    result = [[grid[R - 1 - c][r] for c in range(R)] for r in range(C)]
    return result


def rotate180(grid: Grid) -> Grid:
    """Rotate 180° — equivalent to flip_horizontal ∘ flip_vertical."""
    return [row[::-1] for row in grid[::-1]]


def rotate270(grid: Grid) -> Grid:
    """Rotate 270° clockwise (= 90° counter-clockwise)."""
    R, C = _rows(grid), _cols(grid)
    if R == 0 or C == 0:
        return _copy(grid)
    result = [[grid[c][C - 1 - r] for c in range(R)] for r in range(C)]
    return result


def flip_horizontal(grid: Grid) -> Grid:
    """Mirror left↔right."""
    return [row[::-1] for row in grid]


def flip_vertical(grid: Grid) -> Grid:
    """Mirror top↔bottom."""
    return grid[::-1]


def transpose(grid: Grid) -> Grid:
    """Transpose along main diagonal (row i col j → row j col i)."""
    R, C = _rows(grid), _cols(grid)
    if R == 0 or C == 0:
        return _copy(grid)
    return [[grid[r][c] for r in range(R)] for c in range(C)]


def anti_transpose(grid: Grid) -> Grid:
    """Transpose along anti-diagonal (row i col j → row C-1-j col R-1-i)."""
    R, C = _rows(grid), _cols(grid)
    if R == 0 or C == 0:
        return _copy(grid)
    # output has C rows and R cols; output[i][j] = grid[R-1-j][C-1-i]
    return [[grid[R - 1 - j][C - 1 - i] for j in range(R)] for i in range(C)]


# ---------------------------------------------------------------------------
# Scaling & tiling (8 ops)
# ---------------------------------------------------------------------------

def upscale_2x(grid: Grid) -> Grid:
    """Scale up 2× by duplicating every cell into a 2×2 block."""
    result = []
    for row in grid:
        new_row = [v for v in row for _ in range(2)]
        result.append(new_row)
        result.append(new_row[:])
    return result


def upscale_3x(grid: Grid) -> Grid:
    """Scale up 3× by duplicating every cell into a 3×3 block."""
    result = []
    for row in grid:
        new_row = [v for v in row for _ in range(3)]
        for _ in range(3):
            result.append(new_row[:])
    return result


def tile_2x2(grid: Grid) -> Grid:
    """Tile the grid in a 2×2 arrangement (double width and height)."""
    result = []
    for row in grid:
        result.append(row + row)
    for row in grid:
        result.append(row + row)
    return result


def tile_3x3(grid: Grid) -> Grid:
    """Tile the grid in a 3×3 arrangement (triple width and height)."""
    result = []
    for _ in range(3):
        for row in grid:
            result.append(row * 3)
    return result


def mirror_tile_right(grid: Grid) -> Grid:
    """Tile horizontally as [grid | h-flip(grid)]."""
    flipped = flip_horizontal(grid)
    return [r1 + r2 for r1, r2 in zip(grid, flipped)]


def mirror_tile_down(grid: Grid) -> Grid:
    """Tile vertically as [grid / v-flip(grid)]."""
    return _copy(grid) + flip_vertical(grid)


def compress_2x(grid: Grid) -> Grid:
    """Inverse of upscale_2x: collapse 2×2 uniform blocks into single cells.

    If each non-overlapping 2×2 block is uniform, halve dimensions; else return input.
    """
    R, C = _rows(grid), _cols(grid)
    if R == 0 or C == 0 or R % 2 != 0 or C % 2 != 0:
        return _copy(grid)
    result = []
    for br in range(0, R, 2):
        row_out = []
        for bc in range(0, C, 2):
            vals = {grid[br + dr][bc + dc] for dr in range(2) for dc in range(2)}
            if len(vals) != 1:
                return _copy(grid)  # not uniform — inapplicable
            row_out.append(vals.pop())
        result.append(row_out)
    return result


def deduplicate_rows_cols(grid: Grid) -> Grid:
    """Remove consecutive duplicate rows, then consecutive duplicate columns."""
    # Deduplicate rows
    deduped_rows: list[list[int]] = []
    for row in grid:
        if not deduped_rows or row != deduped_rows[-1]:
            deduped_rows.append(row)
    if not deduped_rows:
        return _copy(grid)
    # Deduplicate columns
    C = len(deduped_rows[0])
    keep_cols: list[int] = []
    prev_col: list[int] | None = None
    for c in range(C):
        col = [deduped_rows[r][c] for r in range(len(deduped_rows))]
        if col != prev_col:
            keep_cols.append(c)
            prev_col = col
    return [[row[c] for c in keep_cols] for row in deduped_rows]


# ---------------------------------------------------------------------------
# Cropping & selection (7 ops)
# ---------------------------------------------------------------------------

def crop_to_content(grid: Grid) -> Grid:
    """Bounding box of non-zero cells; return input if grid is all zeros."""
    R, C = _rows(grid), _cols(grid)
    rows_with: list[int] = [r for r in range(R) if any(grid[r][c] != 0 for c in range(C))]
    cols_with: list[int] = [c for c in range(C) if any(grid[r][c] != 0 for r in range(R))]
    if not rows_with or not cols_with:
        return _copy(grid)
    r0, r1 = rows_with[0], rows_with[-1]
    c0, c1 = cols_with[0], cols_with[-1]
    return [grid[r][c0:c1 + 1] for r in range(r0, r1 + 1)]


def remove_border(grid: Grid) -> Grid:
    """Strip the 1-cell outer frame; return input if grid is too small."""
    R, C = _rows(grid), _cols(grid)
    if R <= 2 or C <= 2:
        return _copy(grid)
    return [grid[r][1:C - 1] for r in range(1, R - 1)]


def top_left_quadrant(grid: Grid) -> Grid:
    """Return the top-left quadrant of the grid."""
    R, C = _rows(grid), _cols(grid)
    if R == 0 or C == 0:
        return _copy(grid)
    return [grid[r][:C // 2] for r in range(R // 2)]


def top_right_quadrant(grid: Grid) -> Grid:
    """Return the top-right quadrant of the grid."""
    R, C = _rows(grid), _cols(grid)
    if R == 0 or C == 0:
        return _copy(grid)
    return [grid[r][C // 2:] for r in range(R // 2)]


def bottom_left_quadrant(grid: Grid) -> Grid:
    """Return the bottom-left quadrant of the grid."""
    R, C = _rows(grid), _cols(grid)
    if R == 0 or C == 0:
        return _copy(grid)
    return [grid[r][:C // 2] for r in range(R // 2, R)]


def bottom_right_quadrant(grid: Grid) -> Grid:
    """Return the bottom-right quadrant of the grid."""
    R, C = _rows(grid), _cols(grid)
    if R == 0 or C == 0:
        return _copy(grid)
    return [grid[r][C // 2:] for r in range(R // 2, R)]


def largest_color_rectangle(grid: Grid) -> Grid:
    """Bounding box of the most frequent non-zero color."""
    dominant = _dominant_nonzero(grid)
    if dominant == 0:
        return _copy(grid)
    R, C = _rows(grid), _cols(grid)
    rows_with = [r for r in range(R) if any(grid[r][c] == dominant for c in range(C))]
    cols_with = [c for c in range(C) if any(grid[r][c] == dominant for r in range(R))]
    if not rows_with or not cols_with:
        return _copy(grid)
    r0, r1 = rows_with[0], rows_with[-1]
    c0, c1 = cols_with[0], cols_with[-1]
    return [grid[r][c0:c1 + 1] for r in range(r0, r1 + 1)]


# ---------------------------------------------------------------------------
# Color ops (4 ops)
# ---------------------------------------------------------------------------

def swap_two_dominant_colors(grid: Grid) -> Grid:
    """Swap the two most frequent non-zero colors."""
    counts: Counter[int] = Counter(v for row in grid for v in row if v != 0)
    if len(counts) < 2:
        return _copy(grid)
    top2 = [c for c, _ in counts.most_common(2)]
    a, b = top2[0], top2[1]
    mapping = {a: b, b: a}
    return [[mapping.get(v, v) for v in row] for row in grid]


def replace_background_with_most_common(grid: Grid) -> Grid:
    """Replace all 0s with the most frequent non-zero color."""
    fill = _dominant_nonzero(grid)
    if fill == 0:
        return _copy(grid)
    return [[fill if v == 0 else v for v in row] for row in grid]


def keep_dominant_color_only(grid: Grid) -> Grid:
    """Zero out all non-zero cells except the most frequent non-zero color."""
    dominant = _dominant_nonzero(grid)
    if dominant == 0:
        return _copy(grid)
    return [[v if v == dominant else 0 for v in row] for row in grid]


def infer_color_map(grid: Grid) -> Grid:
    """Placeholder — replaced per-task by build_color_map_op(); returns identity."""
    return _copy(grid)


def build_color_map_op(pairs: list[tuple[Grid, Grid]]) -> Callable[[Grid], Grid]:
    """Factory: infer a pixel-wise color permutation from demo pairs and return it as a Grid->Grid op.

    Returns identity op if no consistent permutation exists (shapes differ, or mapping is contradictory).
    """
    mapping: dict[int, int] = {}
    for inp, out in pairs:
        if _rows(inp) != _rows(out) or _cols(inp) != _cols(out):
            return _copy  # type: ignore[return-value]
        for row_i, row_o in zip(inp, out):
            for vi, vo in zip(row_i, row_o):
                if vi in mapping:
                    if mapping[vi] != vo:
                        return _copy  # type: ignore[return-value]
                else:
                    mapping[vi] = vo

    if not mapping:
        return _copy  # type: ignore[return-value]

    def _apply(grid: Grid) -> Grid:
        return [[mapping.get(v, v) for v in row] for row in grid]

    return _apply


# ---------------------------------------------------------------------------
# Physics-ish (3 ops)
# ---------------------------------------------------------------------------

def gravity_down(grid: Grid) -> Grid:
    """Non-zero cells fall to the bottom of each column (gravity down)."""
    R, C = _rows(grid), _cols(grid)
    if R == 0 or C == 0:
        return _copy(grid)
    result = [[0] * C for _ in range(R)]
    for c in range(C):
        col = [grid[r][c] for r in range(R)]
        nonzero = [v for v in col if v != 0]
        zeros = [0] * (R - len(nonzero))
        new_col = zeros + nonzero
        for r in range(R):
            result[r][c] = new_col[r]
    return result


def gravity_left(grid: Grid) -> Grid:
    """Non-zero cells fall to the left of each row (gravity left)."""
    result = []
    for row in grid:
        nonzero = [v for v in row if v != 0]
        zeros = [0] * (len(row) - len(nonzero))
        result.append(nonzero + zeros)
    return result


def outline_objects(grid: Grid) -> Grid:
    """Keep only boundary cells of non-zero regions; interior cells become 0."""
    R, C = _rows(grid), _cols(grid)
    if R == 0 or C == 0:
        return _copy(grid)
    result = [[0] * C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 0:
                continue
            # A cell is a boundary cell if any 4-neighbor is 0 or out-of-bounds
            is_boundary = False
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= R or nc < 0 or nc >= C or grid[nr][nc] == 0:
                    is_boundary = True
                    break
            if is_boundary:
                result[r][c] = grid[r][c]
    return result


# ---------------------------------------------------------------------------
# OPS registry (all static ops — infer_color_map placeholder included)
# ---------------------------------------------------------------------------

OPS: dict[str, Callable[[Grid], Grid]] = {
    # Geometry
    "identity": identity,
    "rotate90": rotate90,
    "rotate180": rotate180,
    "rotate270": rotate270,
    "flip_horizontal": flip_horizontal,
    "flip_vertical": flip_vertical,
    "transpose": transpose,
    "anti_transpose": anti_transpose,
    # Scaling & tiling
    "upscale_2x": upscale_2x,
    "upscale_3x": upscale_3x,
    "tile_2x2": tile_2x2,
    "tile_3x3": tile_3x3,
    "mirror_tile_right": mirror_tile_right,
    "mirror_tile_down": mirror_tile_down,
    "compress_2x": compress_2x,
    "deduplicate_rows_cols": deduplicate_rows_cols,
    # Cropping & selection
    "crop_to_content": crop_to_content,
    "remove_border": remove_border,
    "top_left_quadrant": top_left_quadrant,
    "top_right_quadrant": top_right_quadrant,
    "bottom_left_quadrant": bottom_left_quadrant,
    "bottom_right_quadrant": bottom_right_quadrant,
    "largest_color_rectangle": largest_color_rectangle,
    # Color
    "infer_color_map": infer_color_map,  # placeholder; overridden per-task in search.py
    "swap_two_dominant_colors": swap_two_dominant_colors,
    "replace_background_with_most_common": replace_background_with_most_common,
    "keep_dominant_color_only": keep_dominant_color_only,
    # Physics-ish
    "gravity_down": gravity_down,
    "gravity_left": gravity_left,
    "outline_objects": outline_objects,
}


# ---------------------------------------------------------------------------
# SHAPE_RULES — per-op shape transformer for the shape gate
# Returns (out_rows, out_cols) given (in_rows, in_cols), or None if unconstrained.
# ---------------------------------------------------------------------------

def _shape_identity(r: int, c: int) -> tuple[int, int]:
    return (r, c)


def _shape_rotate(r: int, c: int) -> tuple[int, int]:
    return (c, r)  # 90/270 swap dims


def _shape_upscale2(r: int, c: int) -> tuple[int, int]:
    return (r * 2, c * 2)


def _shape_upscale3(r: int, c: int) -> tuple[int, int]:
    return (r * 3, c * 3)


def _shape_tile2x2(r: int, c: int) -> tuple[int, int]:
    return (r * 2, c * 2)


def _shape_tile3x3(r: int, c: int) -> tuple[int, int]:
    return (r * 3, c * 3)


def _shape_mirror_right(r: int, c: int) -> tuple[int, int]:
    return (r, c * 2)


def _shape_mirror_down(r: int, c: int) -> tuple[int, int]:
    return (r * 2, c)


def _shape_compress2(r: int, c: int) -> tuple[int, int] | None:
    if r % 2 == 0 and c % 2 == 0:
        return (r // 2, c // 2)
    return None  # inapplicable


def _shape_remove_border(r: int, c: int) -> tuple[int, int] | None:
    if r > 2 and c > 2:
        return (r - 2, c - 2)
    return None


def _shape_quadrant(r: int, c: int) -> tuple[int, int]:
    return (r // 2, c // 2)


SHAPE_RULES: dict[str, Callable[[int, int], tuple[int, int] | None]] = {
    "identity": _shape_identity,
    "rotate90": _shape_rotate,
    "rotate180": _shape_identity,
    "rotate270": _shape_rotate,
    "flip_horizontal": _shape_identity,
    "flip_vertical": _shape_identity,
    "transpose": _shape_rotate,
    "anti_transpose": _shape_rotate,
    "upscale_2x": _shape_upscale2,
    "upscale_3x": _shape_upscale3,
    "tile_2x2": _shape_tile2x2,
    "tile_3x3": _shape_tile3x3,
    "mirror_tile_right": _shape_mirror_right,
    "mirror_tile_down": _shape_mirror_down,
    "compress_2x": _shape_compress2,
    "deduplicate_rows_cols": None,  # output shape varies
    "crop_to_content": None,        # output shape varies
    "remove_border": _shape_remove_border,
    "top_left_quadrant": _shape_quadrant,
    "top_right_quadrant": _shape_quadrant,
    "bottom_left_quadrant": _shape_quadrant,
    "bottom_right_quadrant": _shape_quadrant,
    "largest_color_rectangle": None,  # output shape varies
    "infer_color_map": _shape_identity,
    "swap_two_dominant_colors": _shape_identity,
    "replace_background_with_most_common": _shape_identity,
    "keep_dominant_color_only": _shape_identity,
    "gravity_down": _shape_identity,
    "gravity_left": _shape_identity,
    "outline_objects": _shape_identity,
}
