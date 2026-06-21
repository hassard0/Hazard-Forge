# Slice BP3 — Deterministic Integer Broadphase: THE BROADPHASE-DRIVEN BOX WORLD STEP (the scaling beat) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The THIRD slice (THE SCALING BEAT) of FLAGSHIP #23
> (DETERMINISTIC INTEGER BROADPHASE, `hf::sim::broad`). BP1 built the body grid, BP2 the candidate-pair generator
> (proven equivalent to all-pairs). BP3 puts it to work: a **broadphase-driven box world step**
> (`StepConvexWorldBP`) that reproduces `convex::StepConvexWorld`'s 5-pass tick with the all-pairs O(n²) solve +
> de-penetration loops REPLACED by iteration over the BP2 candidate-pair list — so a LARGE box scene (256+
> bodies) settles deterministically. The make-or-break is the **SCALE PROOF**: the broadphase-driven step settles
> a large scene BYTE-IDENTICAL to the all-pairs reference on the SAME scene — the broadphase changes ONLY
> performance, not a single bit of the result (provable bit-transparency, extended from BP2's pair-set proof to
> the full dynamics). APPEND to `engine/sim/broad.h` (BP1/BP2 + convex.h/gjk.h/fric.h/persist.h/fpx.h/grain.h
> BYTE-FROZEN). Branch: `slice-bp3`. See [[hazard-forge-broad-roadmap]], [[hazard-forge-gpu-tdr-chunking]],
> [[hazard-forge-docs-style]].

**Goal:** Extend `engine/sim/broad.h` (additive — BP1/BP2 byte-unchanged) with `BroadStepConfig` + the
**static-aware broadphase** `BuildBroadphasePairsWithStatics` (dynamics via the BP2 grid stencil + a
dynamic-vs-static all-pairs pass — the large-body handling deferred from BP2) + `StepConvexWorldBP` /
`StepConvexWorldBPN` (the `StepConvexWorld` shell with the all-pairs solve/de-pen loops replaced by the
**(i,j)-sorted** broadphase pair list, re-built each tick). Add the int64 GPU shader
`shaders/broad_convex_step.comp.hlsl` (Vulkan-only, **chunked dispatch** per the TDR lesson) + the showcase
`--broad-convex-shot` (Vulkan) / `--broad-convex` (Metal). Bake the integer golden `broad_convex`.

## Design call: replace the all-pairs loops with the SORTED candidate list, re-broadphased each tick

`convex::StepConvexWorld` (convex.h:856) is a 5-pass tick: (1) predict-integrate + damping; (2) per-body world
inverse inertia; (3) impulse solve — `solveIters` Gauss-Seidel sweeps over ALL i<j pairs in fixed order; (4)
position de-penetration — `posIters` sweeps over ALL i<j pairs; (5) orientation (done in 1). BP3 reproduces 1, 2,
5 VERBATIM and replaces the **all-pairs `for i: for j>i` loops in (3) and (4)** with iteration over the
broadphase candidate-pair list.
- **THE GAUSS-SEIDEL ORDER CRUX (the make-or-break for the scale proof).** GS is ORDER-DEPENDENT (a pair sees
  earlier pairs' mutations). For `StepConvexWorldBP` to be byte-identical to the all-pairs `StepConvexWorld`, it
  MUST visit the candidate pairs in the SAME order the all-pairs loop visits the box-overlapping subset: **(i,j)
  ascending**. BP2's `BuildBroadphasePairs` emits i-ascending but j-in-stencil-order, so BP3 **sorts the
  broadphase pair list by (i,j)** before the solve (the same canonical sort BP2's equivalence proof uses). Then:
  the all-pairs loop visits every i<j pair, skipping non-overlapping ones (a box overlap ⟹ AABB overlap ⟹ in the
  candidate set); `StepConvexWorldBP` visits exactly the AABB-candidate pairs in (i,j) order, processing the same
  box-overlapping subset in the same order → byte-identical GS result. The AABB-non-overlapping pairs the
  all-pairs loop visits are no-ops (BoxSat finds no overlap), so omitting them changes nothing. This is WHY the
  scale proof holds.
- **THE STATIC / LARGE-BODY HANDLING (deferred from BP2).** A static floor is a large box spanning many cells;
  the BP2 27-cell stencil over center-cells would MISS its pairs. The fix (the scout's recommendation):
  **`BuildBroadphasePairsWithStatics`** = the BP2 grid stencil over the DYNAMIC bodies (statics excluded from the
  grid) ∪ a **dynamic-vs-static all-pairs pass** (each dynamic body × each static body, AABB-tested — O(n·k), k =
  #statics, small; the floor is one body). The combined list is canonicalized i<j and the **equivalence proof
  (carried into BP3) self-polices it**: the combined broadphase pair set (sorted) == `fpx::BuildPairs` all-pairs
  (sorted) over ALL bodies — if the static handling misses a pair, the proof FAILS LOUDLY. (Document this as the
  resolution of BP2's deferred large-body caveat; statics are typically few, so the O(n·k) static pass is cheap.)
- **`StepConvexWorldBP(ConvexWorld& world, const BroadStepConfig& cfg)`** — each tick: (1) integrate+damp
  (verbatim); (2) per-body inertia (verbatim); **(2.5) RE-BUILD the broadphase pair list from the CURRENT
  positions** (`MakeBodyGrid` over dynamics + `BuildBroadphasePairsWithStatics` + sort by (i,j)) — the per-tick
  re-broadphase (positions change each tick, the `fpx.h` SimTick realism); (3) `solveIters` GS sweeps over the
  SORTED pair list (`BoxSatStable`+`BuildManifold`+`SolveManifoldImpulse` verbatim per pair, the static-static
  skip); (4) `posIters` de-pen sweeps over the SORTED pair list (verbatim); (5) orientation. Pure integer, FIXED
  order. `StepConvexWorldBPN(world, cfg, ticks)`.
- **`BroadStepConfig`** = `convex::ConvexStepConfig` (reused) + `fx cellSize` (the grid cell size, ≥ 2·maxDynamicRadius).

> **TDR (the GJ4 lesson — MANDATORY here, [[hazard-forge-gpu-tdr-chunking]]):** the GPU `broad_convex_step.comp`
> runs `StepConvexWorldBPN` single-thread; at 256+ bodies × solveIters × posIters × pairs × ticks this WILL
> exceed the Windows ~2s TDR watchdog. **Chunk the dispatch into bounded tick-batches with a barrier between**
> (bit-identical to one big dispatch — only the dispatch boundary moves; the GJ4 fix shape). **Gate the GPU==CPU
> determinism proof ≥25 runs** (a low-rate TDR flake passes a single run). The per-tick re-broadphase happens
> inside the shader too (the pair list is GPU-rebuilt each tick) OR is host-fed per chunk — the implementer picks
> the form that keeps GPU==CPU bit-exact; document it.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **BP1/BP2 `engine/sim/broad.h` (read; APPEND after BP2's `MeasureBroadphasePairs`):** `BodyGrid`/`MakeBodyGrid`,
  `BuildBodyCellTable`, `BuildBroadphasePairs`, `PairSetEquivalentToAllPairs`, `fpx::FxPair`. BP1/BP2 frozen.
- **convex.h (read-only — REUSE verbatim, do NOT redefine):** READ `StepConvexWorld` IN FULL (convex.h:856-934 —
  the 5-pass body; the all-pairs loops at :883-897 [solve] + :907-932 [de-pen] are what BP3 replaces).
  `ConvexWorld`/`ConvexStepConfig` (:741/:749), `BoxSatStable`, `BuildManifold`, `SolveManifoldImpulse` (:651),
  `FxBoxInvInertiaBody`/`WorldInvInertia` (:606/:622), `MeasureStack` (:942 — the rest/pen summary),
  `fpx::IntegrateBodyFull`. `StepConvexWorldBP` mirrors `StepConvexWorld` line-for-line with the pair-loop swap.
- **The proof-tier convention (convex.h:16-23):** int64 (the narrowphase) → `shaders/broad_convex_step.comp.hlsl`
  is **VULKAN-ONLY** (NOT in `hf_gen_msl`); Metal `--broad-convex` runs the CPU `StepConvexWorldBPN` → byte-
  identical; the Vulkan side carries the GPU==CPU memcmp (CHUNKED). The shader runs the world step single-thread
  (the `convex_step.comp`/`hull_step.comp` convention) over the GPU-rebuilt (or host-fed) sorted pair list.
- **The showcase + shader precedent:** `convex::`'s `--convex-stack-shot` (the settling-stack compute proof +
  GPU==CPU memcmp + the StackMeasure rest/pen assertions + side-view render) AND the GJ4 `--gjk-settle-shot`
  (the CHUNKED-dispatch pattern + the GPU==CPU 25x determinism — copy the chunking). Mirror for `--broad-convex`.
- **Registration:** `scripts/verify.ps1` (append `broad_convex` + `--broad-convex-shot` to `$vkShots`),
  `metal_headless/CMakeLists.txt` (`hf_gen_msl` — do NOT add `broad_convex_step.comp`; Vulkan-only, int64),
  `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller rebakes the JSON golden — do NOT**),
  append to `tests/broad_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/sim/broad.h`** (BP1/BP2 byte-frozen): `BroadStepConfig`, `BuildBroadphasePairsWithStatics`,
   `StepConvexWorldBP`, `StepConvexWorldBPN`. Pure integer, FIXED order, the SORTED (i,j) pair list. NO new RHI;
   one new Vulkan-only shader (CHUNKED dispatch).
2. **New shader `shaders/broad_convex_step.comp.hlsl` (int64, VULKAN-ONLY)** — runs `StepConvexWorldBPN` verbatim,
   single thread, **CHUNKED into bounded tick-batches with a barrier between** (the GJ4 TDR fix shape). NOT in
   `hf_gen_msl`.
3. **Showcase `--broad-convex-shot <out>` (Vulkan) AND `--broad-convex` (Metal) — WIRE BOTH (CRITICAL: grep your
   own visual_test.mm for `--broad-convex` before reporting DONE — the last slices' implementers forgot the Metal
   showcase).** A LARGE scene: a static floor (a big box) + 256+ dynamic boxes dropped into a settling pile.
   Vulkan dispatches `broad_convex_step.comp` (chunked) + memcmps the GPU final world vs the CPU
   `StepConvexWorldBPN`; Metal runs the CPU path. BOTH render the settled pile (a side-view; reuse the
   convex-stack render). Golden = `tests/golden/metal/broad_convex.png` (Mac-baked by the CONTROLLER — DO NOT
   commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU:** `broad-convex: {bodies:<N>, ticks:<K>} GPU==CPU BIT-EXACT` — the GPU final bodies == the
     CPU `StepConvexWorldBPN` byte-for-byte; assert. (Run the shot ≥25× in the gate to rule out a TDR flake.)
   - **(2) determinism:** `broad-convex determinism: two runs BYTE-IDENTICAL`.
   - **(3) THE SCALE PROOF:** `broad-convex scale: {bodies:<N≥256>} broadphase==all-pairs BYTE-IDENTICAL` — the
     CPU `StepConvexWorldBPN` final world == the CPU `convex::StepConvexWorldN` (all-pairs) final world on the
     SAME large scene, byte-for-byte; assert. The broadphase changes ONLY performance, not a bit. ALSO assert the
     pile RESTED (`MeasureStack`: maxSpeed below band, maxPen within slop+band) — a non-trivial settle.
   - **Golden discipline: ONLY `tests/golden/metal/broad_convex.png`; do NOT commit it.** Existing 211 image
     goldens UNTOUCHED.
5. **Cross-backend bar (int64 → strict):** Vulkan GPU==CPU bit-exact (≥25× clean — no TDR flake); Metal CPU-ref
   byte-identical; cross-vendor ZERO differing pixels.
6. **Tests — APPEND to `tests/broad_test.cpp`:** `BuildBroadphasePairsWithStatics` is equivalent to all-pairs over
   a scene WITH a large static floor (the static handling self-policed by the equivalence); `StepConvexWorldBPN`
   == `convex::StepConvexWorldN` byte-identical on a moderate scene (the scale/bit-transparency proof, CPU-only —
   exhaustive); the broadphase step brings the pile to rest; deterministic (two runs byte-equal). Clean under
   `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-broadphase-convex-step` (features) + `--broad-convex-shot`
   (showcases) in `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON
   golden — the controller does that.**

## RHI seam additions (summary)
- **None.** The shot rides the existing compute-dispatch + SSBO path (the `convex_step`/`gjk_settle` seam, incl.
  the chunked-dispatch barrier which already exists). `rhi.h` + backend dirs UNCHANGED. `engine/sim/broad.h`
  APPEND-only (BP1/BP2 frozen); convex.h/gjk.h/fric.h/persist.h/fpx.h/grain.h + ALL other sim headers + ALL
  existing shaders + `engine/physics/`/`nav/`/`anim/` UNCHANGED. NEW file: `shaders/broad_convex_step.comp.hlsl`
  only. Report the seam: one new Vulkan-only shader, no RHI change, no frozen-file edit, broad.h append-only.

## Out of scope (YAGNI — later slices)
The broadphase-driven HULL step (BP4 — `StepHullWorldBP`, the GJK/EPA narrowphase + the hull-AABB-from-supports),
lockstep (BP5), lit render (BP6). Broadphase-driven friction/warm-start (the fric/persist BP steps) are deferred
(they "drop in identically" — the convex.h:728 discipline). BP3 claims ONLY: a deterministic, bit-exact
(CPU↔Vulkan↔Metal) broadphase-driven BOX world step that is PROVABLY bit-transparent (byte-identical to the
all-pairs reference) and scales to 256+ bodies, with the integer golden + the three proofs (incl. the scale
proof). CAVEATS inherited: the convex within-band (linear de-pen, angDamp stability); the broadphase adds ZERO
new approximation; the static pass is O(n·k) (k small).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing + the appended BP3 `broad_test` cases — incl. the
   CPU-only scale/equivalence proofs). Clean under `windows-msvc-asan` (separate build + test — do NOT chain
   `cmake --build >nul && ctest`, it can run a stale binary).
2. **proofs + visual:** `--broad-convex-shot` on Vulkan: the 3 proofs (incl. the scale proof) + exit 0, under the
   Vulkan-validation gate (conan Khronos layer) → ZERO VUID. **Run the shot ≥25× — EVERY run GPU==CPU BIT-EXACT,
   no "differ"/DEVICE_LOST/VkResult=- (the TDR check).** VERIFY the image shows a coherent settled large pile.
3. Metal: `visual_test --broad-convex` → new golden `tests/golden/metal/broad_convex.png`; two runs DIFF 0.0000.
   **Confirm `visual_test.mm` in the diff; confirm `broad_convex_step.comp` is NOT in `hf_gen_msl`.** Cross-vendor
   STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `broad_convex.png` added; the
   other 211 byte-identical. `git diff master --stat -- tests/golden` = ONLY `broad_convex.png` (metal) + the
   introspect json (controller rebake).
5. Introspect: exactly `+deterministic-broadphase-convex-step` + `--broad-convex-shot` added; introspect test
   updated.
6. Seam grep clean (`rhi.h` + BP1/BP2 broad.h code + convex.h/gjk.h/fric.h/persist.h/fpx.h/grain.h + ALL other
   sim headers + ALL existing shaders byte-unchanged; broad.h APPEND-only; one new Vulkan-only shader, no RHI
   change). `scripts/verify.ps1` updated; `broad_convex_step.comp` NOT in `hf_gen_msl`.
