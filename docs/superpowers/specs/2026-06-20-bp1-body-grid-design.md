# Slice BP1 — Deterministic Integer Broadphase: THE BODY GRID + CELL TABLE (the beachhead) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIRST slice (the BEACHHEAD) of FLAGSHIP #23
> (DETERMINISTIC INTEGER BROADPHASE, `hf::sim::broad`). Every rigid solver (convex/fric/persist/gjk) iterates
> ALL i<j body pairs — the O(n²) "all-pairs small scene" caveat documented across the suite. This flagship builds
> a deterministic integer spatial-hash broadphase that produces a bit-identical candidate-pair set, letting the
> rigid solvers scale. BP1 is the structural beachhead: a uniform **body grid** + its **CSR cell table** (bodies
> bucketed into cells via count→scan→emit), bit-exact CPU↔Vulkan↔Metal. It is PURE int32 (cell quantization +
> ascending-index scatter, NO fxmul/int64/sqrt) → the shaders MSL-GENERATE NATIVELY (a true GPU pass on BOTH
> backends, the strongest proof tier — the `grain_cell_*` tier). NEW header `engine/sim/broad.h`, namespace
> `hf::sim::broad`, `#include "sim/gjk.h"` READ-ONLY (transitively convex/fric/persist/fpx, all BYTE-FROZEN).
> Branch: `slice-bp1`. See [[hazard-forge-broad-roadmap]], [[hazard-forge-docs-style]].

**Goal:** Create `engine/sim/broad.h` with `BodyGrid` (a bounded dense grid keyed on a cell size) + `MakeBodyGrid`
+ `BodyCellOf` / `FlatBodyCellId` / `BodyCellCount` + `BodyCellTable` (CSR: `cellStart` prefix-sum + `cellBodies`
scatter) + `BuildBodyCellTable` (count→scan→emit on bodies) + a `BodyGridMeasure` summary. Add the pure-int32
GPU shaders `shaders/broad_cell_count.comp.hlsl` / `broad_cell_scan.comp.hlsl` / `broad_cell_emit.comp.hlsl`
(MSL-native — IN `hf_gen_msl`) that mirror the CPU `BuildBodyCellTable` byte-for-byte, plus the showcase
`--broad-cell-shot` (Vulkan) / `--broad-cell` (Metal). Bake the integer golden `broad_cell`.

## Design call: the per-body CSR cell table — the frozen grain grid, applied to bodies

`grain.h` already ships the exact count→scan→emit grid-hash, MSL-native on both backends (`grain_cell_{count,
scan,emit}`). BP1 is that machinery, keyed on `fpx::FxBody` instead of `GrainParticle`. The grid buckets each
body's center cell; later slices add the 27-cell stencil + AABB pair cull (BP2). BP1 nails the structural store.
- **`BodyGrid { fpx::FxCell cellMin; fpx::FxCell gridDim; fx cellSize; };`** — the bounded dense grid (the
  `grain.h::GrainGrid` twin), `cellSize` is the cell edge (a Q16.16 length; for the broadphase it is sized to the
  bodies' AABB extent — for BP1 it is an input, default e.g. `2·maxRadius` or a passed value; the 27-cell stencil
  in BP2 assumes a body's AABB spans ≤1 cell, so `cellSize ≥ 2·maxRadius`; document the choice + the large-body
  caveat that BP2 resolves).
- **`MakeBodyGrid(const std::vector<fpx::FxBody>& bodies, fx cellSize)`** — the `MakeGrainGrid` twin: `cellMin` =
  the min cell over all body centers, `gridDim` = (maxCell − minCell + 1) per axis. Empty → a 1×1×1 grid at
  origin (the deterministic degenerate). Pure int32 (`FloorDiv`/`FxCell`).
- **`BodyCellOf(pos, cellSize)`** = `fpx::FxCell{FloorDiv(pos.x,cellSize),...}` (the `GrainCellOf` twin);
  **`FlatBodyCellId(cell, grid)`** = `fpx::CellId(cell − cellMin, gridDim)` (the `FlatGrainCellId` twin);
  **`BodyCellCount(grid)`** = `gridDim.x*y*z`.
- **`BodyCellTable { std::vector<uint32_t> cellStart; std::vector<uint32_t> cellBodies; };`** — CSR:
  `cellStart` has `cellCount+1` exclusive-prefix-sum entries (`cellStart[c]..cellStart[c+1]` is cell `c`'s slice;
  the last == body count), `cellBodies` holds the body indices grouped by cell, ASCENDING body index within each
  cell. The `GrainCellTable` twin.
- **`BuildBodyCellTable(bodies, grid)`** — count→scan→emit: (1) per-cell body count; (2) exclusive prefix-sum →
  `cellStart`; (3) scatter each body index into its cell's slice in ASCENDING body order. **THE DET-CRUX (the
  grain.h:293 / FL2 lesson — spell it out):** the EMIT is the single-thread ascending-body scatter (a parallel
  atomic cursor would make the within-cell order GPU-scheduling-dependent → non-deterministic). The count pass is
  per-body-disjoint (race-free); only the emit scatter is the ordered pass. The shaders reproduce exactly: a
  per-body count, a serial prefix-sum scan, a single-thread ascending emit.
- **`BodyGridMeasure { uint32_t bodies; uint32_t cells; uint32_t occupiedCells; uint32_t maxCellOccupancy; };`** —
  a deterministic summary the showcase/test asserts (pure function of the inputs).

## Reuse map (file:line — the implementer MUST ground these before coding)
- **fpx.h (read-only — REUSE, do NOT redefine):** `fpx::FloorDiv` (:177), `fpx::FxCell` (:183),
  `fpx::BroadphaseCell`/`fpx::CellId` (:190/:196), `fpx::FxBody` (:116 — `pos`, `radius` :124, `flags`/
  `kFlagDynamic` :133). The cell quantization is reused verbatim.
- **grain.h (read-only — the STRUCTURE to mirror per-body, do NOT call its grain-typed fns):**
  `grain.h::GrainCellOf` (:245), `FlatGrainCellId` (:252), `GrainCellCount` (:258), `MakeGrainGrid` (:265),
  `GrainCellTable` (:296), `BuildGrainCellTable` (:301) — BP1 reproduces these shapes keyed on `FxBody`. The
  DET-CRUX comment (grain.h:286-295) is the determinism contract to copy.
- **The proof-tier convention (PURE int32 → MSL-NATIVE):** the broadphase cell math is pure int32 (FloorDiv +
  compares + scatter, NO fxmul/int64/sqrt), so `broad_cell_{count,scan,emit}.comp.hlsl` go IN `hf_gen_msl`
  (`metal_headless/CMakeLists.txt`, the `grain_cell_*` rows :363-368 are the template) — a TRUE GPU pass on BOTH
  backends with strict zero-differing-pixel cross-vendor (the STRONGEST tier, stronger than GJK's int64).
- **The showcase + shader precedent:** the grain `--grain-neighbors`/`--grain-integrate` cell showcases + the
  `gjk`/`convex` int32 showcase structure (the GPU==CPU memcmp + the integer diagnostic render). Mirror for
  `--broad-cell`.
- **Registration:** `scripts/verify.ps1` (append `broad_cell` + `--broad-cell-shot` to `$vkShots`; the `gjk_*`
  rows are the template), `metal_headless/CMakeLists.txt` (`hf_gen_msl` — ADD `broad_cell_count/scan/emit.comp`,
  they ARE MSL-native), `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller rebakes the
  JSON golden — do NOT**), a NEW `tests/broad_test.cpp` (+ register in the test CMake like `gjk_test`).

## Design decisions (locked)
1. **NEW header `engine/sim/broad.h`** (namespace `hf::sim::broad`, `#include "sim/gjk.h"` read-only): `BodyGrid`,
   `MakeBodyGrid`, `BodyCellOf`, `FlatBodyCellId`, `BodyCellCount`, `BodyCellTable`, `BuildBodyCellTable`,
   `BodyGridMeasure` + `MeasureBodyGrid`. gjk.h/convex.h/fric.h/persist.h/fpx.h BYTE-FROZEN.
2. **New shaders `shaders/broad_cell_{count,scan,emit}.comp.hlsl` (PURE int32, MSL-NATIVE → IN `hf_gen_msl`)** —
   reproduce `BuildBodyCellTable` byte-for-byte: count (per-body, race-free), scan (serial exclusive prefix-sum),
   emit (single-thread ascending scatter). Write `cellStart[]` + `cellBodies[]` to SSBOs. NO new RHI (ride the
   existing compute-dispatch + SSBO path the `grain_cell` shots use). The GPU exercises the EXACT int32 ops → a
   divergence is what the host GPU==CPU memcmp catches.
3. **Showcase `--broad-cell-shot <out>` (Vulkan) AND `--broad-cell` (Metal) — WIRE BOTH.** Build a fixed scene: a
   deterministic spread of bodies (e.g. a 3D lattice + a few clustered, varied radii within the cell-size bound)
   at fixed poses. Vulkan dispatches the 3 broad_cell shaders + memcmps the GPU `cellStart`/`cellBodies` vs the
   CPU `BuildBodyCellTable`; Metal runs the GPU shaders too (MSL-native — the strongest tier, NOT a CPU ref).
   BOTH render an integer diagnostic (a top-down XZ view, bodies colored by flat cell id — mirror the grain cell
   diagnostic). Golden = `tests/golden/metal/broad_cell.png` (Mac-baked by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU:** `broad-cell: {bodies:<N>, cells:<C>, occupied:<O>} GPU==CPU BIT-EXACT` — the GPU
     `cellStart`+`cellBodies` == the CPU byte-for-byte; assert.
   - **(2) determinism:** `broad-cell determinism: two runs BYTE-IDENTICAL`.
   - **(3) total partition (correctness):** every body lands in exactly one cell and the CSR is a TOTAL partition
     — `cellStart[cellCount] == N`, `Σ per-cell counts == N`, every body index appears EXACTLY once in
     `cellBodies`, and within each cell the indices are ASCENDING. Print `broad-cell correct:
     {partition:true, ascending:true}`; assert.
   - **Golden discipline: ONLY `tests/golden/metal/broad_cell.png`; do NOT commit it.** Existing 209 image
     goldens UNTOUCHED.
5. **Cross-backend bar (int32 MSL-native → strict):** Vulkan GPU==CPU bit-exact AND Metal GPU==CPU bit-exact
   (both backends run the real shader); cross-vendor ZERO differing pixels.
6. **Tests — NEW `tests/broad_test.cpp`:** `BuildBodyCellTable` is a total partition (each body once, ascending
   within cell, `cellStart` monotone, last == N); `MakeBodyGrid` tightly bounds the bodies (cellMin/gridDim
   correct, empty → 1×1×1); `FlatBodyCellId` ∈ [0, cellCount); `MeasureBodyGrid` is a pure function. Clean under
   `windows-msvc-asan`. Register `broad_test` in the test CMake.
7. **Introspect.** Add exactly `deterministic-broadphase-grid` (features) + `--broad-cell-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** The shot rides the existing compute-dispatch + SSBO path (the `grain_cell` shot's seam). `rhi.h` +
  backend dirs UNCHANGED. `engine/sim/gjk.h` + convex.h/fric.h/persist.h/fpx.h/grain.h + ALL other sim headers +
  ALL existing shaders + `engine/physics/`/`nav/`/`anim/` UNCHANGED. NEW files only: `engine/sim/broad.h`,
  `shaders/broad_cell_{count,scan,emit}.comp.hlsl`, `tests/broad_test.cpp` (+ the showcase/introspect/verify
  edits). Report the seam: three new MSL-native shaders (IN hf_gen_msl), no RHI change, no frozen-file edit.

## Out of scope (YAGNI — later slices)
The 27-cell stencil + AABB pair cull + the grid-pairs==all-pairs equivalence proof (BP2 — the crux), the
broadphase-driven box/hull world steps (BP3/BP4), lockstep (BP5), lit render (BP6). The large-body
AABB-cell-span insert is a BP2 concern (BP1's bodies fit ≤1 cell by the cellSize bound). BP1 claims ONLY: a
deterministic, bit-exact (CPU↔Vulkan↔Metal, both GPU-native) body grid + CSR cell table, with the integer golden
+ the three proofs. NOTE: bounded dense grid (a sparse 2-cluster scene allocates the bounding-box cell volume —
fine for the canonical pile scenes, the `MakeGrainGrid` precedent; a hashed grid is a future refinement).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing + the new `broad_test`). Clean under `windows-msvc-asan`.
2. **proofs + visual:** `--broad-cell-shot` on Vulkan: the 3 proofs + exit 0, under the Vulkan-validation gate
   (the conan Khronos layer) → ZERO VUID. **VERIFY the diagnostic shows the bodies colored by cell id coherently
   (a clean cell tiling, no garbage).**
3. Metal: `visual_test --broad-cell` → new golden `tests/golden/metal/broad_cell.png`; two runs DIFF 0.0000.
   **Confirm `visual_test.mm` in the diff; confirm `broad_cell_*.comp` ARE in `hf_gen_msl` (MSL-native, both
   backends run the real shader).** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `broad_cell.png` added; the other
   209 byte-identical. `git diff master --stat -- tests/golden` = ONLY `broad_cell.png` (metal) + the introspect
   json (controller rebake).
5. Introspect: exactly `+deterministic-broadphase-grid` + `--broad-cell-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` + `engine/sim/gjk.h`/convex.h/fric.h/persist.h/fpx.h/grain.h + ALL other sim headers
   + ALL existing shaders byte-unchanged; three new MSL-native shaders, no RHI change). `scripts/verify.ps1`
   updated; `broad_cell_{count,scan,emit}.comp` ADDED to `hf_gen_msl`.
