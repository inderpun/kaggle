# Rung 2B: Object-Level Reasoning — Design & Learning Notes

*Extends [dsl-design.md](dsl-design.md). Read that first.*

## 1. Why grids aren't the right abstraction

Rung 2's whole-grid ops scored 51/1000 on training and **0/120 on ARC-AGI-2** — the
measured cliff. The reason: humans don't see ARC tasks as pixel arrays. They see
**objects** — connected blobs with color, shape, size, position — and rules like
"keep the largest", "move the red one onto the blue one", "recolor by size rank".
This is the **cognitive-priors** argument baked into ARC's design (objectness is one
of Chollet's core priors, alongside geometry, counting, and goal-directedness).
ARC-AGI-2 filtered out whole-grid solvable tasks; object tasks survived.

## 2. Objects as a representation

```python
@dataclass(frozen=True)
class Obj:
    cells: frozenset[tuple[int, int]]   # absolute (row, col)
    color: int                          # dominant color
    # derived, precomputed: size, bbox, height, width, normalized shape
```

**Segmentation is itself a hypothesis.** A grid decomposes differently under
different rules, so we segment three ways and let search pick what works:
- `objects_4conn_same_color` — orthogonal connectivity, one color per object (default)
- `objects_8conn_same_color` — diagonal counts as connected
- `objects_multicolor_4conn` — any non-background cells connect (composite objects)

Background = most frequent color (usually 0, not always — infer per grid).

## 3. New op families (object-parameterized, still `Grid -> Grid`)

To stay composable with rung 2's BFS, object ops are *generated as closures* per
(selector, action) combination — each generated op is still `Grid -> Grid`:

**Selectors** (pick one object): `largest, smallest, most_common_shape, unique_shape,
unique_color, topmost, bottommost, leftmost, rightmost`

**Actions on selection**: `keep_only` (erase the rest), `remove` (erase it),
`crop_to` (return its bounding box), `recolor_to_dominant`

**Whole-decomposition transforms**: `recolor_by_size_rank` (largest→color of largest
demo-output object, etc. — inferred mapping), `move_all_to_gravity` (objects fall as
units), `count_objects_as_grid` (1×N strip encoding the count — for counting tasks)

3 segmentations × (9 selectors × 4 actions) + 3 transforms ≈ **~110 generated ops**.

## 4. Keeping search tractable

Branching factor jumps from 30 to ~140 — `140³ ≈ 2.7M` is too many. Mitigations, in
order of leverage:

1. **Segmentation-consistency gate**: before search, test each of the 3 segmentations
   on demo pairs — if input and output object counts/colors are wholly unrelated
   under a segmentation, drop its ops for this task. Usually kills ⅔ of object ops.
2. **Depth budget by family**: object ops mostly appear alone or with one geometry
   op. Search: depth ≤ 2 over the full op set, plus depth 3 restricted to
   (whole-grid op) ∘ (object op) ∘ (whole-grid op) patterns.
3. Existing pruning (first-demo short-circuit, shape gate, dedup) carries over.

Budget target: ≤ 5s per task on CPU, 120 tasks < 10 min.

## 5. What this teaches (interview framing)

- **Representation beats search depth**: 110 well-chosen ops at depth 2 outperform
  30 ops at depth 4 because the primitives match the domain's *causal structure*.
  Same lesson as feature engineering, same lesson as choosing the right abstraction
  in system design.
- **Core Knowledge priors**: ARC encodes the hypothesis that intelligence needs
  built-in priors (objectness, counting, geometry). Rung 2B is us hand-coding the
  objectness prior; rung 3 (test-time training) tries to *learn* it.

## 6. Implementation contract

- `src/objects.py` — `Obj` dataclass, the 3 segmentation functions, selectors,
  actions, op generation (`generate_object_ops() -> dict[str, Callable]`), the
  segmentation-consistency gate. Pure functions, total, type-hinted.
- `src/search.py` — extend to accept an op registry + the family-aware depth budget
  (§4.2); keep the existing API working (`solve_dsl` unchanged, new `solve_dsl_obj`).
- Register `"dsl_obj"` in `evaluate.py`.
- `src/test_objects.py` — segmentation on a hand-built 2-object grid (both
  connectivities), each selector on a 3-object grid, `recolor_by_size_rank` round-trip.
- Acceptance: tests green; eval split < 10 min; report training AND evaluation
  pass@2 with solved ids + programs. Target: **eval_pass@2 > 0** (any nonzero v2
  score is the rung's win condition); training should rise well above 51.
