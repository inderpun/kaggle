"""Unit tests for dsl.py — runnable via `python3 src/test_dsl.py`.

Tests every geometry op against hand-computed results, tile/upscale/compress
round-trips, and infer_color_map positive + negative cases.

Uses plain assert statements — no pytest required.
"""
from __future__ import annotations

from dsl import (
    anti_transpose,
    build_color_map_op,
    compress_2x,
    crop_to_content,
    deduplicate_rows_cols,
    flip_horizontal,
    flip_vertical,
    gravity_down,
    gravity_left,
    identity,
    keep_dominant_color_only,
    largest_color_rectangle,
    mirror_tile_down,
    mirror_tile_right,
    outline_objects,
    remove_border,
    replace_background_with_most_common,
    rotate180,
    rotate270,
    rotate90,
    swap_two_dominant_colors,
    tile_2x2,
    tile_3x3,
    transpose,
    upscale_2x,
    upscale_3x,
    bottom_left_quadrant,
    bottom_right_quadrant,
    top_left_quadrant,
    top_right_quadrant,
)

# Hand-computed reference grid: 2 rows × 3 cols
GRID_2x3 = [
    [1, 2, 3],
    [4, 5, 6],
]

PASS = 0
FAIL = 0


def check(name: str, got, expected) -> None:
    global PASS, FAIL
    if got == expected:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")
        print(f"        expected: {expected}")
        print(f"        got:      {got}")


# ---------------------------------------------------------------------------
# Geometry ops
# ---------------------------------------------------------------------------

def test_identity():
    result = identity(GRID_2x3)
    check("identity", result, [[1, 2, 3], [4, 5, 6]])


def test_rotate90():
    # 2x3 -> 3x2 clockwise:
    # col 0 bottom->top becomes row 0: 4, 1
    # col 1 bottom->top becomes row 1: 5, 2
    # col 2 bottom->top becomes row 2: 6, 3
    result = rotate90(GRID_2x3)
    check("rotate90", result, [[4, 1], [5, 2], [6, 3]])


def test_rotate180():
    # Reverse rows and reverse each row
    result = rotate180(GRID_2x3)
    check("rotate180", result, [[6, 5, 4], [3, 2, 1]])


def test_rotate270():
    # 2x3 -> 3x2 counter-clockwise
    # col C-1-0=2 top->bottom becomes row 0: 3,6
    # col C-1-1=1 top->bottom becomes row 1: 2,5
    # col C-1-2=0 top->bottom becomes row 2: 1,4
    result = rotate270(GRID_2x3)
    check("rotate270", result, [[3, 6], [2, 5], [1, 4]])


def test_flip_horizontal():
    result = flip_horizontal(GRID_2x3)
    check("flip_horizontal", result, [[3, 2, 1], [6, 5, 4]])


def test_flip_vertical():
    result = flip_vertical(GRID_2x3)
    check("flip_vertical", result, [[4, 5, 6], [1, 2, 3]])


def test_transpose():
    # 2x3 -> 3x2: row i col j -> row j col i
    result = transpose(GRID_2x3)
    check("transpose", result, [[1, 4], [2, 5], [3, 6]])


def test_anti_transpose():
    # 2x3 -> 3x2 (note: output is R_out=C_in=3, C_out=R_in=2)
    # anti_transpose: result[r][c] = grid[R-1-c][C-1-r]
    # R=2, C=3
    # result[0][0] = grid[1][2] = 6
    # result[0][1] = grid[0][2] = 3
    # result[1][0] = grid[1][1] = 5
    # result[1][1] = grid[0][1] = 2
    # result[2][0] = grid[1][0] = 4
    # result[2][1] = grid[0][0] = 1
    result = anti_transpose(GRID_2x3)
    check("anti_transpose", result, [[6, 3], [5, 2], [4, 1]])


# ---------------------------------------------------------------------------
# D8 group properties
# ---------------------------------------------------------------------------

def test_rotate_4_is_identity():
    g = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    r4 = rotate90(rotate90(rotate90(rotate90(g))))
    check("rotate90^4 == identity", r4, g)


def test_rotate180_double_flip():
    # rotate180 == flip_h ∘ flip_v
    g = [[1, 2, 3], [4, 5, 6]]
    r180 = rotate180(g)
    double_flip = flip_horizontal(flip_vertical(g))
    check("rotate180 == flip_h∘flip_v", r180, double_flip)


def test_transpose_anti_transpose():
    g = [[1, 2, 3], [4, 5, 6]]
    # anti_transpose == flip_h ∘ transpose ∘ flip_h
    a = anti_transpose(g)
    b = flip_horizontal(transpose(flip_horizontal(g)))
    check("anti_transpose == fh∘t∘fh", a, b)


# ---------------------------------------------------------------------------
# Scaling & tiling round-trips
# ---------------------------------------------------------------------------

def test_upscale_2x_shape():
    g = [[1, 2], [3, 4]]
    result = upscale_2x(g)
    check("upscale_2x shape", (len(result), len(result[0])), (4, 4))


def test_upscale_2x_values():
    g = [[1, 2], [3, 4]]
    result = upscale_2x(g)
    expected = [
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4],
    ]
    check("upscale_2x values", result, expected)


def test_upscale_3x_shape():
    g = [[1, 2], [3, 4]]
    result = upscale_3x(g)
    check("upscale_3x shape", (len(result), len(result[0])), (6, 6))


def test_compress_roundtrip():
    g = [[1, 2], [3, 4]]
    up = upscale_2x(g)
    down = compress_2x(up)
    check("upscale_2x ∘ compress_2x roundtrip", down, g)


def test_compress_non_uniform():
    g = [[1, 2, 3, 4], [1, 2, 3, 4]]
    result = compress_2x(g)
    check("compress_2x non-uniform returns input", result, g)


def test_tile_2x2_shape():
    g = [[1, 2], [3, 4]]
    result = tile_2x2(g)
    check("tile_2x2 shape", (len(result), len(result[0])), (4, 4))


def test_tile_3x3_shape():
    g = [[1, 2], [3, 4]]
    result = tile_3x3(g)
    check("tile_3x3 shape", (len(result), len(result[0])), (6, 6))


def test_mirror_tile_right():
    g = [[1, 2], [3, 4]]
    result = mirror_tile_right(g)
    expected = [[1, 2, 2, 1], [3, 4, 4, 3]]
    check("mirror_tile_right", result, expected)


def test_mirror_tile_down():
    g = [[1, 2], [3, 4]]
    result = mirror_tile_down(g)
    expected = [[1, 2], [3, 4], [3, 4], [1, 2]]
    check("mirror_tile_down", result, expected)


def test_deduplicate_rows_cols():
    g = [[1, 1, 2], [1, 1, 2], [3, 3, 4]]
    result = deduplicate_rows_cols(g)
    check("deduplicate_rows_cols", result, [[1, 2], [3, 4]])


# ---------------------------------------------------------------------------
# Cropping & selection
# ---------------------------------------------------------------------------

def test_crop_to_content():
    g = [
        [0, 0, 0],
        [0, 1, 2],
        [0, 3, 0],
    ]
    result = crop_to_content(g)
    check("crop_to_content", result, [[1, 2], [3, 0]])


def test_crop_to_content_all_zero():
    g = [[0, 0], [0, 0]]
    result = crop_to_content(g)
    check("crop_to_content all-zero returns input", result, g)


def test_remove_border():
    g = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    result = remove_border(g)
    check("remove_border 3x3", result, [[5]])


def test_remove_border_too_small():
    g = [[1, 2], [3, 4]]
    result = remove_border(g)
    check("remove_border 2x2 returns input", result, g)


def test_top_left_quadrant():
    g = [[1, 2, 3, 4], [5, 6, 7, 8]]
    result = top_left_quadrant(g)
    check("top_left_quadrant", result, [[1, 2]])


def test_top_right_quadrant():
    g = [[1, 2, 3, 4], [5, 6, 7, 8]]
    result = top_right_quadrant(g)
    check("top_right_quadrant", result, [[3, 4]])


def test_bottom_left_quadrant():
    g = [[1, 2, 3, 4], [5, 6, 7, 8]]
    result = bottom_left_quadrant(g)
    check("bottom_left_quadrant", result, [[5, 6]])


def test_bottom_right_quadrant():
    g = [[1, 2, 3, 4], [5, 6, 7, 8]]
    result = bottom_right_quadrant(g)
    check("bottom_right_quadrant", result, [[7, 8]])


def test_largest_color_rectangle():
    g = [
        [1, 0, 0],
        [1, 2, 2],
        [0, 2, 2],
    ]
    # Color 2 appears 4 times, color 1 appears 2 times → dominant = 2
    # Rows with color 2: rows 1,2; Cols with color 2: cols 1,2
    # Bounding box: rows [1..2], cols [1..2]
    result = largest_color_rectangle(g)
    check("largest_color_rectangle", result, [[2, 2], [2, 2]])


# ---------------------------------------------------------------------------
# Color ops
# ---------------------------------------------------------------------------

def test_swap_two_dominant_colors():
    g = [[1, 1, 2], [1, 2, 0]]  # 1 appears 3x, 2 appears 2x
    result = swap_two_dominant_colors(g)
    check("swap_two_dominant_colors", result, [[2, 2, 1], [2, 1, 0]])


def test_replace_background():
    g = [[0, 1], [2, 0]]
    result = replace_background_with_most_common(g)
    check("replace_background_with_most_common", result, [[1, 1], [2, 1]])


def test_keep_dominant_color_only():
    g = [[1, 1, 2], [3, 1, 0]]
    result = keep_dominant_color_only(g)
    check("keep_dominant_color_only", result, [[1, 1, 0], [0, 1, 0]])


# ---------------------------------------------------------------------------
# infer_color_map — positive and negative cases
# ---------------------------------------------------------------------------

def test_infer_color_map_positive():
    """Consistent color map: 1->2, 2->3."""
    pairs = [
        ([[1, 2], [2, 1]], [[2, 3], [3, 2]]),
        ([[1, 1], [2, 2]], [[2, 2], [3, 3]]),
    ]
    fn = build_color_map_op(pairs)
    result = fn([[1, 2], [2, 1]])
    check("infer_color_map positive", result, [[2, 3], [3, 2]])


def test_infer_color_map_negative_contradictory():
    """Contradictory: 1->2 in pair 0, 1->3 in pair 1 → identity fallback."""
    pairs = [
        ([[1, 2]], [[2, 3]]),   # 1->2, 2->3
        ([[1, 2]], [[3, 4]]),   # 1->3 contradicts 1->2
    ]
    fn = build_color_map_op(pairs)
    result = fn([[1, 2]])
    check("infer_color_map contradictory → identity", result, [[1, 2]])


def test_infer_color_map_negative_shapes_differ():
    """Shape mismatch → identity fallback."""
    pairs = [
        ([[1, 2], [3, 4]], [[1, 2, 0]]),
    ]
    fn = build_color_map_op(pairs)
    result = fn([[5, 6]])
    check("infer_color_map shape mismatch → identity", result, [[5, 6]])


# ---------------------------------------------------------------------------
# Physics-ish
# ---------------------------------------------------------------------------

def test_gravity_down():
    g = [[1, 0], [0, 2], [3, 0]]
    result = gravity_down(g)
    check("gravity_down", result, [[0, 0], [1, 0], [3, 2]])


def test_gravity_left():
    g = [[0, 1, 0, 2], [3, 0, 0, 4]]
    result = gravity_left(g)
    check("gravity_left", result, [[1, 2, 0, 0], [3, 4, 0, 0]])


def test_outline_objects():
    g = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    result = outline_objects(g)
    # Interior cell (1,1) has all 4 neighbors non-zero → not boundary → 0
    check("outline_objects interior zeroed", result, [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ])


def test_outline_objects_single_cell():
    g = [[0, 0], [0, 5]]
    result = outline_objects(g)
    # Single non-zero cell — all its neighbors are either OOB or 0 → boundary
    check("outline_objects single cell stays", result, [[0, 0], [0, 5]])


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("DSL unit tests")
    print("=" * 60)

    test_functions = [
        test_identity,
        test_rotate90,
        test_rotate180,
        test_rotate270,
        test_flip_horizontal,
        test_flip_vertical,
        test_transpose,
        test_anti_transpose,
        test_rotate_4_is_identity,
        test_rotate180_double_flip,
        test_transpose_anti_transpose,
        test_upscale_2x_shape,
        test_upscale_2x_values,
        test_upscale_3x_shape,
        test_compress_roundtrip,
        test_compress_non_uniform,
        test_tile_2x2_shape,
        test_tile_3x3_shape,
        test_mirror_tile_right,
        test_mirror_tile_down,
        test_deduplicate_rows_cols,
        test_crop_to_content,
        test_crop_to_content_all_zero,
        test_remove_border,
        test_remove_border_too_small,
        test_top_left_quadrant,
        test_top_right_quadrant,
        test_bottom_left_quadrant,
        test_bottom_right_quadrant,
        test_largest_color_rectangle,
        test_swap_two_dominant_colors,
        test_replace_background,
        test_keep_dominant_color_only,
        test_infer_color_map_positive,
        test_infer_color_map_negative_contradictory,
        test_infer_color_map_negative_shapes_differ,
        test_gravity_down,
        test_gravity_left,
        test_outline_objects,
        test_outline_objects_single_cell,
    ]

    for fn in test_functions:
        fn()

    print("=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)

    if FAIL > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
