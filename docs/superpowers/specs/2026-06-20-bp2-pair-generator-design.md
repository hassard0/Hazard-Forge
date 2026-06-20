# Slice BP2 — Deterministic Integer Broadphase: THE CANDIDATE-PAIR GENERATOR (the crux) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SECOND slice (THE CRUX) of FLAGSHIP #23
> (DETERMINISTIC INTEGER BROADPHASE, `hf::sim::broad`). BP1 built the body grid + CSR cell table. BP2 builds the
> payoff: a **27-cell-stencil candidate-pair generator** that, for each body, scans its grid neighborhood and
> emits the canonical i<j AABB-overlapping pairs — the same candidate-pair set the O(n²) all-pairs scan
> produces, only via the grid. The make-or-break is the **EQUIVALENCE PROOF**: the grid-emitted pair set is
> BYTE-IDENTICAL (after canonical sort) to `fpx::BuildPairs` (the all-pairs reference) — the broadphase is
> *provably bit-transparent*, producing the identical pair list the slow path would, just faster. PURE int32
> (stencil + AABB compares + ascending scatter, NO fxmul/int64) → the shaders MSL-GENERATE NATIVELY (true GPU
> both backends, the strongest tier). APPEND to `engine/sim/broad.h` (BP1 + gjk.h/convex.h/fric.h/persist.h/
> fpx.h/grain.h BYTE-FROZEN). Branch: `slice-bp2`. See [[hazard-forge-broad-roadmap]], [[hazard-forge-docs-style]].

**Goal:** Extend `engine/sim/broad.h` (additive — BP1 byte-unchanged) with `BroadphaseAccept` (reuses
`fpx::AabbOverlap` over `fpx::BodyAabb`) + `CountBroadphasePairs` + `BuildBroadphasePairs` (the 27-cell stencil
over the BP1 cell table → canonical i<j `fpx::FxPair[]`, count→scan→emit) + `BroadphasePairsEqual` /
`PairSetEquivalentToAllPairs` (the equivalence check vs `fpx::BuildPairs`) + a `BroadphasePairMeasure`. Add the
pure-int32 GPU shaders `shaders/broad_pair_count.comp.hlsl` / `broad_pair_scan.comp.hlsl` /
`broad_pair_emit.comp.hlsl` (MSL-native — IN `hf_gen_msl`), plus the showcase `--broad-pair-shot` (Vulkan) /
`--broad-pair` (Metal). Bake the integer golden `broad_pair`.

## Design call: the 27-cell stencil + canonical i<j de-dup — bit-transparent vs all-pairs

For a body `i` at cell `c`, every body whose AABB can overlap `i`'s lies in the 3×3×3 = **27-cell stencil**
around `c` (PROVIDED `cellSize ≥` the max body AABB diameter, so two overlapping AABBs are always within ±1
cell — the BP1 cell-size bound). BP2 scans `i`'s stencil and emits `(i, j)` for each body `j` in those cells with
**`j > i`** (the canonical de-dup — each unordered pair emitted ONCE, from the lower-index body's stencil) AND
`fpx::AabbOverlap(BodyAabb(i), BodyAabb(j))`. This produces the SAME set `fpx::CountPairs`/`BuildPairs` produce
all-pairs, because the stencil is symmetric (i in j's stencil ⟺ j in i's, under the cell-size bound) so the
`j>i` guard captures each overlapping pair exactly once.
- **THE NOVELTY — canonical de-dup + ordered emit (the determinism crux).** Unlike grain's per-grain neighbor
  list (asymmetric, i sees j AND j sees i, never de-duped), a rigid pair list is canonical i<j ONCE. The rule:
  emit `(i,j)` only when `j>i` (the `fpx.h:283` discipline), restricted to the 27-cell stencil. The EMIT is the
  **single-thread ascending-body scatter** (the BP1/grain DET-CRUX — a parallel atomic cursor would make the
  within-list order GPU-schedule-dependent → nondeterministic). For each body i, the stencil cells are visited in
  a FIXED order and bodies within a cell in ascending index, so the per-i emit order is deterministic (NOT
  necessarily ascending-j across cells — that is fine; the GPU mirrors it bit-for-bit and the equivalence proof
  compares SETS via a canonical sort).
- **`BuildBroadphasePairs(bodies, grid, cellTable, perBodyOffset, pairsOut)`** — count→scan→emit:
  (1) COUNT: per body i, count `j>i` in the 27-cell stencil with overlapping AABB (per-body-disjoint, race-free);
  (2) SCAN: exclusive prefix-sum → `perBodyOffset` (the serial scan); (3) EMIT: single-thread ascending body i,
  scatter each accepted `(i,j)` into `i`'s slice. Returns the deterministic `fpx::FxPair[]` + offsets.
- **THE EQUIVALENCE PROOF — `PairSetEquivalentToAllPairs`.** Sort the BP2 grid-pair list by `(i,j)` and sort the
  `fpx::BuildPairs(world)` all-pairs list by `(i,j)` (it is already i-then-j ordered), then compare as SETS:
  same count, byte-identical after sort. **This is the make-or-break falsifiable claim** — if the stencil misses
  a pair (e.g. a body larger than the cell-size span) or duplicates one, grid-pairs ≠ all-pairs and the proof
  FAILS LOUDLY (self-policing). The proof is EXACT (a byte memcmp of sorted lists), not within-band.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **BP1 `engine/sim/broad.h` (read it; APPEND only after `MeasureBodyGrid`, before the namespace close):**
  `BodyGrid`, `MakeBodyGrid`, `BodyCellOf`, `FlatBodyCellId`, `BodyCellCount`, `BodyCellTable`,
  `BuildBodyCellTable`. BP1 byte-frozen.
- **fpx.h (read-only — REUSE verbatim):** `fpx::FxAabb` (:210), `fpx::BodyAabb` (:217), `fpx::AabbOverlap` (:228
  — the six-compare predicate), `fpx::FxPair` (:255), `fpx::CountPairs` (:239) + `fpx::BuildPairs` (:264 — **THE
  ALL-PAIRS REFERENCE the equivalence proof compares against**). `fpx::FxBody` (:116, `radius` :124). The
  candidate-pair shape + the AABB predicate are reused; BP2 only grid-accelerates which `j`'s are tested.
- **fpx.h cell helpers (read-only):** `fpx::FxCell`, `fpx::CellId` — for the 27-cell stencil iteration over the
  BP1 grid.
- **grain.h (read-only — the 27-cell stencil STRUCTURE to mirror):** `grain.h::BuildGrainNeighborList` (:393 —
  the per-grain 27-cell stencil scan; BP2 mirrors its stencil iteration but with the `j>i` de-dup + the AABB
  predicate instead of the radius test). The DET-CRUX (single-thread ascending emit).
- **The proof-tier convention (PURE int32 → MSL-NATIVE):** stencil iteration + AABB compares + scatter are pure
  int32, so `broad_pair_{count,scan,emit}.comp.hlsl` go IN `hf_gen_msl` (the BP1 `broad_cell_*` + grain
  `grain_neighbor_*` tier) — true GPU both backends, strict zero cross-vendor.
- **The showcase + shader precedent:** BP1's `--broad-cell` (the 3-shader dispatch + GPU==CPU memcmp + diagnostic)
  and the grain `--grain-neighbors`. Mirror for `--broad-pair`.
- **Registration:** `scripts/verify.ps1` (append `broad_pair` + `--broad-pair-shot` to `$vkShots`),
  `metal_headless/CMakeLists.txt` (`hf_gen_msl` — ADD `broad_pair_count/scan/emit.comp`, MSL-native),
  `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller rebakes the JSON golden — do NOT**),
  append to `tests/broad_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/sim/broad.h`** (BP1 byte-frozen): `BroadphaseAccept`, `CountBroadphasePairs`,
   `BuildBroadphasePairs`, `PairSetEquivalentToAllPairs`, `BroadphasePairMeasure` + `MeasureBroadphasePairs`. Pure
   int32, FIXED stencil + ascending-emit order. NO new RHI; three new MSL-native shaders.
2. **New shaders `shaders/broad_pair_{count,scan,emit}.comp.hlsl` (PURE int32, MSL-NATIVE → IN `hf_gen_msl`)** —
   reproduce `BuildBroadphasePairs` byte-for-byte (count per-body race-free; scan serial; emit single-thread
   ascending). Read the BP1 grid + cell table from SSBOs; write `pairsOut` + offsets. NO new RHI.
3. **Showcase `--broad-pair-shot <out>` (Vulkan) AND `--broad-pair` (Metal) — WIRE BOTH (CRITICAL: the Metal
   showcase MUST be wired in visual_test.mm — grep your own visual_test.mm for `--broad-pair` before reporting
   DONE).** Build a fixed scene of UNIFORM-radius DYNAMIC bodies (no large/static body — so `cellSize ≥ 2·radius`
   makes the ±1 stencil EXACT; the large-body/static handling is BP3's concern, noted below): a deterministic
   cloud where many bodies' AABBs overlap (some clustered) so the pair set is non-trivial. Vulkan dispatches the
   3 broad_pair shaders (over the BP1 grid) + memcmps the GPU `pairsOut`/offsets vs the CPU
   `BuildBroadphasePairs`; Metal runs the GPU shaders too (MSL-native). BOTH render an integer diagnostic
   (top-down XZ, the AABB footprints + a segment per emitted candidate pair). Golden =
   `tests/golden/metal/broad_pair.png` (Mac-baked by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU:** `broad-pair: {bodies:<N>, pairs:<P>} GPU==CPU BIT-EXACT` — the GPU `pairsOut`+offsets ==
     the CPU `BuildBroadphasePairs` byte-for-byte; assert.
   - **(2) determinism:** `broad-pair determinism: two runs BYTE-IDENTICAL`.
   - **(3) THE EQUIVALENCE PROOF:** `broad-pair equivalence: grid-pairs == all-pairs {pairs:<P>} BYTE-IDENTICAL` —
     the BP2 grid-pair set (sorted by (i,j)) == `fpx::BuildPairs` all-pairs (sorted) byte-for-byte; assert. This
     is THE crux: the broadphase is bit-transparent (no pair missed, none duplicated).
   - **Golden discipline: ONLY `tests/golden/metal/broad_pair.png`; do NOT commit it.** Existing 210 image
     goldens UNTOUCHED.
5. **Cross-backend bar (int32 MSL-native → strict):** Vulkan GPU==CPU AND Metal GPU==CPU bit-exact; cross-vendor
   ZERO differing pixels.
6. **Tests — APPEND to `tests/broad_test.cpp`:** `BuildBroadphasePairs` over a uniform-body cloud emits the
   canonical i<j set; `PairSetEquivalentToAllPairs` is TRUE (grid-pairs == `fpx::BuildPairs`, the equivalence) for
   several scenes (sparse, clustered, all-overlapping); each pair appears once, i<j; the emit is deterministic
   (two calls byte-equal); a degenerate single-body scene → zero pairs. Clean under `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-broadphase-pairs` (features) + `--broad-pair-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** The shot rides the existing compute-dispatch + SSBO path (BP1's `broad_cell` seam). `rhi.h` +
  backend dirs UNCHANGED. `engine/sim/broad.h` APPEND-only (BP1 frozen); gjk.h/convex.h/fric.h/persist.h/fpx.h/
  grain.h + ALL other sim headers + ALL existing shaders + `engine/physics/`/`nav/`/`anim/` UNCHANGED. NEW files:
  `shaders/broad_pair_{count,scan,emit}.comp.hlsl` only. Report the seam: three new MSL-native shaders (IN
  hf_gen_msl), no RHI change, no frozen-file edit, broad.h append-only.

## Out of scope (YAGNI — later slices)
The broadphase-driven box/hull world steps (BP3/BP4 — they consume the BP2 pair list), lockstep (BP5), lit render
(BP6). **The large-body / static handling is DEFERRED to BP3** (where the static floor appears): BP2's scene is
uniform-radius dynamic bodies (the ±1 stencil exact). BP3 will handle a body whose AABB spans many cells (the
floor) via an AABB-cell-span insert OR a separate dynamic-vs-static all-pairs pass — and the equivalence proof
(carried into BP3) self-polices it. BP2 documents the cell-size constraint (`cellSize ≥ 2·maxRadius`) as the
condition under which the ±1 stencil is exact. BP2 claims ONLY: a deterministic, bit-exact (CPU↔Vulkan↔Metal,
both GPU-native) candidate-pair generator that is PROVABLY equivalent to the all-pairs reference for
cell-size-bounded bodies, with the integer golden + the three proofs (incl. the equivalence).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing + the appended BP2 `broad_test` cases). Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--broad-pair-shot` on Vulkan: the 3 proofs (incl. the equivalence) + exit 0, under the
   Vulkan-validation gate (conan Khronos layer) → ZERO VUID. **VERIFY the diagnostic shows the AABB footprints +
   the candidate-pair segments coherently (clustered bodies densely connected, distant bodies unconnected).**
3. Metal: `visual_test --broad-pair` → new golden `tests/golden/metal/broad_pair.png`; two runs DIFF 0.0000.
   **Confirm `visual_test.mm` in the diff (the Metal showcase IS wired); confirm `broad_pair_*.comp` ARE in
   `hf_gen_msl`.** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `broad_pair.png` added; the other
   210 byte-identical. `git diff master --stat -- tests/golden` = ONLY `broad_pair.png` (metal) + the introspect
   json (controller rebake).
5. Introspect: exactly `+deterministic-broadphase-pairs` + `--broad-pair-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` + BP1's broad.h code + gjk.h/convex.h/fric.h/persist.h/fpx.h/grain.h + ALL other sim
   headers + ALL existing shaders byte-unchanged; broad.h APPEND-only; three new MSL-native shaders, no RHI
   change). `scripts/verify.ps1` updated; `broad_pair_{count,scan,emit}.comp` ADDED to `hf_gen_msl`.
