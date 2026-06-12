# DSL + Program Search for ARC — Design & Learning Notes

*Rung 2 of our approach ladder. Written to be liftable into the self-learning hub.*

## 1. Why programs, not models?

A neural net trained on 1000 ARC tasks learns *those tasks' rules* — but every test
task has a **novel** rule, so there's nothing to interpolate. What generalizes is the
**process of finding rules**, not the rules themselves.

Program induction reframes solving as search: find a program `p` (a composition of
grid transformations) such that `p(input_i) == output_i` for **every** demonstration
pair. Then `p(test_input)` is your answer. Three properties make this work:

- **Verification is free.** Unlike open-ended generation, we can *prove* a candidate
  explains the demos before using it. Search + verification beats generation whenever
  verification is cheap — the same insight behind test-time-compute reasoning models.
- **Occam's razor becomes executable.** Many programs may fit the demos; shorter
  programs generalize better. Formally this is **Minimum Description Length (MDL)**:
  the simplest consistent hypothesis is most likely the intended rule. We implement
  it by literally preferring shorter compositions.
- **Interpretability is total.** A solved task isn't a softmax — it's
  `crop_to_content ∘ rotate90`, a sentence you can defend in a writeup.

## 2. What a DSL is, and the central trade-off

A **domain-specific language** is a curated vocabulary of operations matched to the
domain's structure. ARC grids have geometry (rotation/reflection), topology (objects,
holes), and color algebra — so those become the primitives.

The design tension: **expressivity vs. searchability**. With `k` ops and search depth
`d`, the space is ~`k^d`:

| ops | depth | programs |
|---|---|---|
| 30 | 2 | 900 |
| 30 | 3 | 27,000 |
| 30 | 4 | 810,000 |
| 100 | 4 | 100,000,000 |

Every op you add buys coverage and costs search depth. Hodel's `arc-dsl` (the
reference open-source ARC DSL) has ~160 primitives but needs guided search; rung 2
stays parameter-free and shallow so **brute-force BFS is exhaustive and honest** —
we'll know exactly what depth-3 composition of 30 ops can and cannot solve.

## 3. The operation inventory (~30 ops, all `Grid -> Grid`)

All ops are total functions (never raise; return input unchanged if inapplicable).

**Geometry (8)** — the D8 symmetry group, the most common ARC motif:
`identity, rotate90, rotate180, rotate270, flip_horizontal, flip_vertical, transpose, anti_transpose`

**Scaling & tiling (8)** — "the output is a bigger/repeated version of the input":
`upscale_2x, upscale_3x, tile_2x2, tile_3x3, mirror_tile_right` (input | h-flip),
`mirror_tile_down`, `compress_2x` (inverse of upscale when blocks are uniform), `deduplicate_rows_cols` (collapse consecutive duplicate rows/cols)

**Cropping & selection (7)** — "the output is a piece of the input":
`crop_to_content` (bounding box of non-zero), `remove_border` (strip 1-cell frame),
`top_left_quadrant, top_right_quadrant, bottom_left_quadrant, bottom_right_quadrant`,
`largest_color_rectangle` (bounding box of the most frequent non-zero color)

**Color (4)** —
`infer_color_map` ★ (see below), `swap_two_dominant_colors`, `replace_background_with_most_common`, `keep_dominant_color_only` (zero out all but most frequent non-zero color)

**Physics-ish (3)** —
`gravity_down` (non-zero cells fall within columns), `gravity_left`, `outline_objects` (keep only boundary cells of non-zero regions)

★ `infer_color_map` is the one *task-parameterized* op: if a single consistent
pixel-wise color permutation maps every demo input to its output (shapes equal), apply
that learned mapping. It's inferred once per task from the demos — a tiny taste of
rung 3's "learn at test time" idea.

## 4. Search algorithm

**Iterative-deepening BFS over compositions, depth ≤ 3**, with verification and
pruning:

```
for depth in 1..3:
  for each op-sequence of that depth (k^d, generated lazily):
    out = apply sequence to demo_1 input          # cheapest check first
    if out != demo_1 output: continue             # short-circuit
    if sequence explains ALL demo pairs exactly:
        record (sequence, depth)
```

Pruning that keeps this fast:
- **First-demo short-circuit** (above) — kills ~all candidates in one apply.
- **Shape gate**: before applying, if the op can't produce the demo output's shape
  from the input shape (e.g. `upscale_2x` when shapes are equal), skip. Implemented
  as a cheap shape-transform table per op.
- **State dedup**: per task, memoize intermediate grids at each depth; two prefixes
  producing the same grid are equivalent — explore one.

**pass@2 strategy**: collect *all* consistent programs, group by their prediction on
the test input, order groups by (shortest program, then frequency). Attempt 1 = top
group's prediction; attempt 2 = second group's (or rung-1 fallback). Two *different
hypotheses*, not the same one twice — this is how pass@2 is meant to be used.

## 5. Honest expectations

ARC-AGI-2 was filtered against exactly this kind of shallow symbolic solving.
Parameter-free depth-3 BFS should solve **a few of the 120 eval tasks** (low single
digits). That's the point of the rung: a measured, interpretable floor that tells us
*which* task families need objects (rung 2B), parameters, or learning (rung 3).
On v1-era ARC this same machinery would score several times higher — the gap **is**
the ARC-AGI-2 thesis, and quantifying it on our own harness is writeup material.

## 6. Implementation contract (for the implementing agent)

- `src/dsl.py` — every op as a pure function `Grid -> Grid`, total (no exceptions),
  plus `OPS: dict[str, Callable]` registry and `SHAPE_RULES` for the shape gate.
  Type hints + docstring (one line: what pattern it targets) on every op.
- `src/search.py` — `search_programs(task, max_depth=3) -> list[Program]` where
  `Program = tuple[str, ...]` (op names), implementing §4 exactly (short-circuit,
  shape gate, dedup). `solve_dsl(task) -> list[tuple[Grid, Grid]]` implementing the
  pass@2 strategy with rung-1 fallback for unsolved tasks.
- Register `"dsl": solve_dsl` in `evaluate.py`'s `SOLVERS`.
- Unit tests `src/test_dsl.py`: each geometry op against a hand-computed 2x3 grid;
  `tile`/`upscale`/`compress` round-trips; `infer_color_map` positive + negative case.
  Runnable via `python3 -m pytest src/test_dsl.py` (or plain `python3 src/test_dsl.py`
  with asserts if pytest unavailable).
- Acceptance: `make score SOLVER=dsl` runs the 120 eval tasks in < 5 minutes on CPU
  and reports `eval_pass@2 > 0`. Log every solved task id + its program.
