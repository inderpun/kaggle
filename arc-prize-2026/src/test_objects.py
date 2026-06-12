"""Tests for objects.py — plain asserts, no pytest required.

Run from src/: python3 test_objects.py
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from objects import (
    Obj,
    background_color,
    objects_4conn_same_color,
    objects_8conn_same_color,
    objects_multicolor_4conn,
    generate_object_ops,
    filter_segmentations,
    filter_object_ops,
    SEGMENTATIONS,
    SELECTORS,
)


# ---------------------------------------------------------------------------
# Hand-built test grids
# ---------------------------------------------------------------------------

# 5x5 grid with two well-separated objects:
#   Object A: color 1, top-left 2x2 block
#   Object B: color 2, bottom-right 2x2 block
# Background: 0 (most frequent — fills the rest)
GRID_TWO_OBJECTS: list[list[int]] = [
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 2, 2],
    [0, 0, 0, 2, 2],
]

# 5x5 grid with two objects touching diagonally (only connected by 8-connectivity):
#   Object A: color 3, cell (0,0) only
#   Object B: color 3, cell (1,1) only
# Background: 0
GRID_DIAGONAL: list[list[int]] = [
    [3, 0, 0, 0, 0],
    [0, 3, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
]

# 7x7 grid with 3 objects for selector testing:
#   Object A: color 1, size 1 (single cell at (0,0)) — smallest, leftmost, topmost
#   Object B: color 2, size 4 (2x2 at rows 2-3, cols 2-3) — middle
#   Object C: color 3, size 9 (3x3 at rows 4-6, cols 4-6) — largest, rightmost, bottommost
# Background: 0
GRID_THREE_OBJECTS: list[list[int]] = [
    [1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 2, 0, 0, 0],
    [0, 0, 2, 2, 0, 0, 0],
    [0, 0, 0, 0, 3, 3, 3],
    [0, 0, 0, 0, 3, 3, 3],
    [0, 0, 0, 0, 3, 3, 3],
]

# Grid for recolor_by_size_rank: same 3-object grid
GRID_RECOLOR = GRID_THREE_OBJECTS


# ---------------------------------------------------------------------------
# Test: background_color
# ---------------------------------------------------------------------------

def test_background_color() -> None:
    assert background_color(GRID_TWO_OBJECTS) == 0, "background should be 0 (most frequent)"
    # Grid where 1 is most frequent
    grid_1bg: list[list[int]] = [[1, 1, 1], [1, 2, 1], [1, 1, 1]]
    assert background_color(grid_1bg) == 1


# ---------------------------------------------------------------------------
# Test: segmentation on the two-object grid
# ---------------------------------------------------------------------------

def test_4conn_two_objects() -> None:
    objs = objects_4conn_same_color(GRID_TWO_OBJECTS)
    assert len(objs) == 2, f"expected 2 objects, got {len(objs)}"
    colors = {o.color for o in objs}
    assert colors == {1, 2}, f"expected colors {{1, 2}}, got {colors}"
    sizes = sorted(o.size() for o in objs)
    assert sizes == [4, 4], f"expected sizes [4, 4], got {sizes}"


def test_8conn_two_objects() -> None:
    objs = objects_8conn_same_color(GRID_TWO_OBJECTS)
    assert len(objs) == 2, f"expected 2 objects, got {len(objs)}"
    colors = {o.color for o in objs}
    assert colors == {1, 2}


def test_4conn_diagonal_stays_split() -> None:
    """4-connectivity should NOT merge diagonally-touching cells of the same color."""
    objs = objects_4conn_same_color(GRID_DIAGONAL)
    assert len(objs) == 2, f"4-conn should produce 2 separate objects, got {len(objs)}"


def test_8conn_diagonal_merges() -> None:
    """8-connectivity SHOULD merge diagonally-touching cells of the same color."""
    objs = objects_8conn_same_color(GRID_DIAGONAL)
    assert len(objs) == 1, f"8-conn should merge diagonal cells into 1 object, got {len(objs)}"
    assert objs[0].size() == 2


def test_multicolor_4conn_two_objects() -> None:
    objs = objects_multicolor_4conn(GRID_TWO_OBJECTS)
    assert len(objs) == 2, f"expected 2 objects, got {len(objs)}"


def test_multicolor_composite() -> None:
    """Composite object: two colors 4-connected to each other."""
    grid: list[list[int]] = [
        [0, 0, 0],
        [0, 1, 2],  # 1 and 2 are adjacent (4-connected)
        [0, 0, 0],
    ]
    objs = objects_multicolor_4conn(grid)
    assert len(objs) == 1, f"multicolor should merge adjacent diff-color cells: got {len(objs)}"
    assert objs[0].size() == 2


# ---------------------------------------------------------------------------
# Test: selectors on the three-object grid
# ---------------------------------------------------------------------------

def _get_objs_3() -> list[Obj]:
    return objects_4conn_same_color(GRID_THREE_OBJECTS)


def test_sel_largest() -> None:
    objs = _get_objs_3()
    obj = SELECTORS["largest"](objs)
    assert obj is not None
    assert obj.color == 3, f"largest should be color-3 (9 cells), got color {obj.color}"
    assert obj.size() == 9


def test_sel_smallest() -> None:
    objs = _get_objs_3()
    obj = SELECTORS["smallest"](objs)
    assert obj is not None
    assert obj.color == 1, f"smallest should be color-1 (1 cell), got color {obj.color}"
    assert obj.size() == 1


def test_sel_unique_color() -> None:
    objs = _get_objs_3()
    # All 3 colors are unique — returns first unique-color object
    obj = SELECTORS["unique_color"](objs)
    assert obj is not None  # should find something (all colors unique here)


def test_sel_topmost() -> None:
    objs = _get_objs_3()
    obj = SELECTORS["topmost"](objs)
    assert obj is not None
    assert obj.color == 1, f"topmost should be color-1 (at row 0), got color {obj.color}"


def test_sel_bottommost() -> None:
    objs = _get_objs_3()
    obj = SELECTORS["bottommost"](objs)
    assert obj is not None
    assert obj.color == 3, f"bottommost should be color-3 (rows 4-6), got color {obj.color}"


def test_sel_leftmost() -> None:
    objs = _get_objs_3()
    obj = SELECTORS["leftmost"](objs)
    assert obj is not None
    assert obj.color == 1, f"leftmost should be color-1 (col 0), got color {obj.color}"


def test_sel_rightmost() -> None:
    objs = _get_objs_3()
    obj = SELECTORS["rightmost"](objs)
    assert obj is not None
    assert obj.color == 3, f"rightmost should be color-3 (cols 4-6), got color {obj.color}"


def test_sel_most_common_shape() -> None:
    # Grid with two objects of the same shape (1x1), one of different shape (2x2)
    grid: list[list[int]] = [
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 2, 2, 0],
        [0, 0, 2, 2, 0],
        [0, 0, 0, 0, 3],
    ]
    objs = objects_4conn_same_color(grid)
    obj = SELECTORS["most_common_shape"](objs)
    # 1x1 shapes: color-1 (size 1) and color-3 (size 1) both have 1x1 shape
    assert obj is not None
    assert obj.size() == 1  # should pick a 1x1 object (the most common shape)


def test_sel_unique_shape() -> None:
    # Grid: two 1x1 objects and one 2x2 (unique shape is 2x2)
    grid: list[list[int]] = [
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 2, 2, 0],
        [0, 0, 2, 2, 0],
        [0, 0, 0, 0, 3],
    ]
    objs = objects_4conn_same_color(grid)
    obj = SELECTORS["unique_shape"](objs)
    assert obj is not None
    assert obj.color == 2, f"unique shape (2x2) should be color-2, got color {obj.color}"


# ---------------------------------------------------------------------------
# Test: actions
# ---------------------------------------------------------------------------

def test_action_keep_only() -> None:
    """keep_only: only the selected object's cells remain, rest is background."""
    ops = generate_object_ops()
    op = ops["obj_4conn_largest_keep_only"]
    result = op(GRID_THREE_OBJECTS)
    # Only color-3 (9 cells) should remain; rest should be 0
    nonzero_vals = [result[r][c] for r in range(7) for c in range(7) if result[r][c] != 0]
    assert all(v == 3 for v in nonzero_vals), f"keep_only should leave only color 3, got {set(nonzero_vals)}"
    assert len(nonzero_vals) == 9


def test_action_remove() -> None:
    """remove: removes the selected object; others stay."""
    ops = generate_object_ops()
    op = ops["obj_4conn_largest_remove"]
    result = op(GRID_THREE_OBJECTS)
    # Color-3 cells should now be 0
    color3_cells = [(r, c) for r in range(7) for c in range(7) if GRID_THREE_OBJECTS[r][c] == 3]
    for r, c in color3_cells:
        assert result[r][c] == 0, f"expected cell ({r},{c}) to be 0 after remove, got {result[r][c]}"
    # Color-1 and color-2 cells should be intact
    color1_cells = [(r, c) for r in range(7) for c in range(7) if GRID_THREE_OBJECTS[r][c] == 1]
    for r, c in color1_cells:
        assert result[r][c] == 1


def test_action_crop_to() -> None:
    """crop_to: returns bounding box of selected object."""
    ops = generate_object_ops()
    op = ops["obj_4conn_largest_crop_to"]
    result = op(GRID_THREE_OBJECTS)
    # Largest object is color-3, 3x3 at rows 4-6, cols 4-6
    assert len(result) == 3, f"expected 3 rows, got {len(result)}"
    assert len(result[0]) == 3, f"expected 3 cols, got {len(result[0])}"
    assert all(result[r][c] == 3 for r in range(3) for c in range(3))


def test_action_recolor_to_dominant() -> None:
    """recolor_to_dominant: all non-background cells become the selected object's color."""
    ops = generate_object_ops()
    op = ops["obj_4conn_smallest_recolor_to_dominant"]
    result = op(GRID_THREE_OBJECTS)
    # Smallest is color-1; all non-zero cells should become 1
    nonzero = [result[r][c] for r in range(7) for c in range(7) if result[r][c] != 0]
    assert all(v == 1 for v in nonzero), f"expected all non-bg cells to be 1, got {set(nonzero)}"


# ---------------------------------------------------------------------------
# Test: whole-decomposition transforms
# ---------------------------------------------------------------------------

def test_recolor_by_size_rank() -> None:
    """Largest gets rank 1, next gets rank 2, smallest gets rank 3."""
    ops = generate_object_ops()
    op = ops["obj_4conn_recolor_by_size_rank"]
    result = op(GRID_THREE_OBJECTS)
    # Color-3 (size 9, rank 1) -> color 1
    # Color-2 (size 4, rank 2) -> color 2
    # Color-1 (size 1, rank 3) -> color 3
    # Check rank-1 cells (original color-3, rows 4-6 cols 4-6)
    rank1_cells = [(r, c) for r in range(4, 7) for c in range(4, 7)]
    for r, c in rank1_cells:
        assert result[r][c] == 1, f"rank-1 cell ({r},{c}) expected 1, got {result[r][c]}"
    # Check rank-3 cell (original color-1, row 0 col 0)
    assert result[0][0] == 3, f"rank-3 cell (0,0) expected 3, got {result[0][0]}"


def test_count_objects_as_grid() -> None:
    """Count returns a 1×N strip where N = number of objects."""
    ops = generate_object_ops()
    op = ops["obj_4conn_count_objects_as_grid"]
    result = op(GRID_THREE_OBJECTS)
    assert len(result) == 1, f"expected 1 row, got {len(result)}"
    assert len(result[0]) == 3, f"expected 3 cols (3 objects), got {len(result[0])}"
    assert result[0] == [1, 1, 1], f"expected [1, 1, 1], got {result[0]}"


# ---------------------------------------------------------------------------
# Test: op generation count
# ---------------------------------------------------------------------------

def test_op_count() -> None:
    """Should generate ~117 ops (108 selector-action + 9 transforms)."""
    ops = generate_object_ops()
    obj_ops = {k: v for k, v in ops.items() if k.startswith("obj_")}
    # 3 segs × 9 selectors × 4 actions = 108
    # 3 segs × (1 recolor_by_size + 4 gravity + 1 count) = 3 × 6 = 18
    # Total = 126
    assert len(obj_ops) >= 100, f"expected >=100 object ops, got {len(obj_ops)}"
    print(f"  Generated {len(obj_ops)} object ops")


# ---------------------------------------------------------------------------
# Test: segmentation-consistency gate
# ---------------------------------------------------------------------------

def test_filter_segmentations_keeps_relevant() -> None:
    """Gate should keep segmentations where objects are consistent across demo pairs."""
    # Simple demo pair: input has 2 objects, output has 2 objects, same colors
    inp: list[list[int]] = [
        [1, 0, 2],
        [0, 0, 0],
        [0, 0, 0],
    ]
    out: list[list[int]] = [
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 2],
    ]
    demo_pairs = [(inp, out)]
    kept = filter_segmentations(demo_pairs)
    # 4conn and 8conn should both keep these (2 objects on each side, same colors)
    assert "4conn" in kept or "8conn" in kept, f"expected at least one segmentation kept, got {kept}"


def test_filter_segmentations_failopen() -> None:
    """Gate should fail-open (keep all) if no segmentation passes."""
    # Degenerate: all-zero input, all-zero output
    inp: list[list[int]] = [[0, 0], [0, 0]]
    out: list[list[int]] = [[0, 0], [0, 0]]
    demo_pairs = [(inp, out)]
    kept = filter_segmentations(demo_pairs)
    # No objects in either side; gate should fail-open
    assert len(kept) == 3, f"expected all 3 segmentations kept (fail-open), got {kept}"


# ---------------------------------------------------------------------------
# Test: Obj dataclass properties
# ---------------------------------------------------------------------------

def test_obj_properties() -> None:
    """Test Obj.size, bbox, height, width, normalized_shape."""
    cells = frozenset([(0, 0), (0, 1), (1, 0), (1, 1)])
    obj = Obj(cells=cells, color=5)
    assert obj.size() == 4
    assert obj.bbox() == (0, 0, 1, 1)
    assert obj.height() == 2
    assert obj.width() == 2
    ns = obj.normalized_shape()
    assert ns == frozenset([(0, 0), (0, 1), (1, 0), (1, 1)])

    # Test translation invariance
    cells2 = frozenset([(3, 4), (3, 5), (4, 4), (4, 5)])
    obj2 = Obj(cells=cells2, color=5)
    assert obj.normalized_shape() == obj2.normalized_shape(), "normalized shape should be translation-invariant"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    tests = [
        test_background_color,
        test_4conn_two_objects,
        test_8conn_two_objects,
        test_4conn_diagonal_stays_split,
        test_8conn_diagonal_merges,
        test_multicolor_4conn_two_objects,
        test_multicolor_composite,
        test_sel_largest,
        test_sel_smallest,
        test_sel_unique_color,
        test_sel_topmost,
        test_sel_bottommost,
        test_sel_leftmost,
        test_sel_rightmost,
        test_sel_most_common_shape,
        test_sel_unique_shape,
        test_action_keep_only,
        test_action_remove,
        test_action_crop_to,
        test_action_recolor_to_dominant,
        test_recolor_by_size_rank,
        test_count_objects_as_grid,
        test_op_count,
        test_filter_segmentations_keeps_relevant,
        test_filter_segmentations_failopen,
        test_obj_properties,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
