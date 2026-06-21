# Slice BP4 — Deterministic Integer Broadphase: THE BROADPHASE-DRIVEN HULL WORLD STEP (the new-capability beat) — Design

> Autonomous-session spec. Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FOURTH slice of
> FLAGSHIP #23 (DETERMINISTIC INTEGER BROADPHASE, `hf::sim::broad`). BP3 made the BOX world step broadphase-driven
> and proved it byte-identical to all-pairs at 256 bodies. BP4 does the SAME for arbitrary convex HULLS:
> `StepHullWorldBP` reproduces `gjk::StepHullWorld`'s tick with the all-pairs O(n²) GJK/EPA narrowphase loops
> REPLACED by the broadphase candidate-pair list — so a large MIXED-HULL pile (tetra/octa/wedge/box) settles
> deterministically and scales. The make-or-break is again the **SCALE PROOF**: byte-identical to the all-pairs
> `gjk::StepHullWorldN` on the same scene (the broadphase is bit-transparent for hulls too). This is the
> capability payoff — deterministic broadphase + general convex hulls together, scaling. APPEND to
> `engine/sim/broad.h` (BP1-BP3 + gjk.h/convex.h/fric.h/persist.h/fpx.h/grain.h BYTE-FROZEN). Branch:
> `slice-bp4`. See [[hazard-forge-broad-roadmap]], [[hazard-forge-gpu-tdr-chunking]], [[hazard-forge-docs-style]].

**Goal:** Extend `engine/sim/broad.h` (additive — BP1-BP3 byte-unchanged) with `BuildHullAabb` (the hull's tight
world AABB from `gjk::Support` along the ±6 axes — the one new int64 geometry helper) + a hull-aware broadphase
pair builder (the BP3 static-aware builder using hull AABBs instead of `fpx::BodyAabb`) + `StepHullWorldBP` /
`StepHullWorldBPN` (the `gjk::StepHullWorld` shell with the all-pairs loops → the (i,j)-sorted broadphase pair
list, re-built each tick). Add the int64 GPU shader `shaders/broad_hull_step.comp.hlsl` (Vulkan-only, **chunked at
1 tick/dispatch from the start** — the hull step is heavy, the GJ4/BP3 TDR lesson) + the showcase
`--broad-hull-shot` (Vulkan) / `--broad-hull` (Metal). Bake the integer golden `broad_hull`.

## Design call: BP3 for hulls — swap the AABB source + the narrowphase, keep the (i,j)-sorted GS order

This is the direct hull twin of BP3. The ONLY differences from `StepConvexWorldBP`:
- **The broadphase AABB is the hull's TRUE world AABB** (not a `body.radius` sphere box). `BuildHullAabb(hull,
  body)` = the min/max over `gjk::Support(hull, body, ±X/±Y/±Z)` (six support queries → the tight axis-aligned
  box; int64 via `gjk::Support`/`FxRotate`). The broadphase grid cell + the `AabbOverlap` predicate use this hull
  AABB. (Cell size ≥ the max hull AABB diameter so the ±1 stencil stays exact; large/static hulls handled by the
  BP3 dynamic-vs-static all-pairs pass.)
- **The narrowphase is `gjk::HullContact`** (GJK→EPA→manifold), not `BoxSatStable`. `StepHullWorldBP` reproduces
  `gjk::StepHullWorld` (GJ4) line-for-line with the all-pairs solve/de-pen loops replaced by iteration over the
  **(i,j)-sorted** broadphase pair list (the BP3 Gauss-Seidel-order crux — sort so the GS result matches the
  all-pairs visiting order; the broadphase candidate set ⊇ every GJK-overlapping pair, processed in the same
  order → byte-identical). Re-broadphase each tick from current positions.
- **`StepHullWorldBP(HullWorld& world, const BroadStepConfig& cfg)`** — `gjk::StepHullWorld`'s shell:
  integrate+damp → per-tick re-broadphase (`MakeBodyGrid` over the hull-AABB-derived cells + the hull pair
  builder + sort (i,j)) → solve sweeps over the sorted pair list (`gjk::HullContact`+`SolveManifoldImpulse`
  verbatim) → de-pen sweeps → orient. `StepHullWorldBPN(world, cfg, ticks)`.

**TDR (apply up front — the BP3 lesson, [[hazard-forge-gpu-tdr-chunking]]):** the GPU `broad_hull_step.comp` runs
`StepHullWorldBPN` single-thread; the hull GJK/EPA narrowphase is HEAVIER than box SAT, so **chunk at 1
tick/dispatch from the start** (don't reuse BP3's value). Keep the scene MODERATE (e.g. ~48-64 mixed hulls × ~120
ticks) so the CPU all-pairs scale reference (the verification cost) stays fast — the scale proof needs a
non-trivial count, NOT 256. Verify GPU==CPU determinism with a TIMING argument + ~3 runs (1 tick/dispatch is
provably ≪ the 2s watchdog), NOT a 30× loop.

## Reuse map (file:line)
- **BP1-BP3 `engine/sim/broad.h` (APPEND after BP3's `StepConvexWorldBPN`):** `BodyGrid`/`MakeBodyGrid`,
  `BuildBroadphasePairs`, `BuildBroadphasePairsWithStatics`, `BroadStepConfig`, `PairSetEquivalentToAllPairs`,
  `fpx::FxPair`, the (i,j)-sort. BP1-BP3 frozen.
- **gjk.h (read-only — REUSE verbatim):** READ `gjk::StepHullWorld`/`StepHullWorldN` (GJ4) IN FULL — the shell
  BP4 mirrors with the pair-loop swap; `gjk::HullContact`, `gjk::HullWorld`, `gjk::FxHull`, `gjk::Support` (:99 —
  for `BuildHullAabb`), `gjk::FxHullInvInertiaBody`, `gjk::MeasureHullStack`. `convex::SolveManifoldImpulse`/
  `ContactManifold`. Do NOT modify gjk.h/convex.h/fpx.h/etc — BYTE-FROZEN.
- **Proof-tier:** int64 → `broad_hull_step.comp` VULKAN-ONLY (NOT in `hf_gen_msl`); Metal `--broad-hull` runs CPU
  `StepHullWorldBPN`. The GJ4 `--gjk-settle-shot` chunked dispatch + the BP3 `--broad-convex-shot` are the
  precedents.
- **Registration:** `scripts/verify.ps1` (append `broad_hull` + `--broad-hull-shot` to `$vkShots`),
  `metal_headless/CMakeLists.txt` (`hf_gen_msl` — do NOT add `broad_hull_step.comp`), `engine/editor/introspect.cpp`
  + `tests/introspect_test.cpp` (**controller rebakes the JSON golden**), append to `tests/broad_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/sim/broad.h`** (BP1-BP3 byte-frozen): `BuildHullAabb`, the hull-AABB broadphase pair
   builder (or a `HullWorld` overload of `BuildBroadphasePairsWithStatics`), `StepHullWorldBP`, `StepHullWorldBPN`.
   Pure integer, the SORTED (i,j) pair list. NO new RHI; one new Vulkan-only shader (chunk=1).
2. **New shader `shaders/broad_hull_step.comp.hlsl` (int64, VULKAN-ONLY)** — runs `StepHullWorldBPN` verbatim,
   single thread, **chunked at 1 tick/dispatch** with a barrier between (TDR-safe by construction). NOT in
   `hf_gen_msl`.
3. **Showcase `--broad-hull-shot <out>` (Vulkan) AND `--broad-hull` (Metal) — WIRE BOTH (grep your own
   visual_test.mm for `--broad-hull` before reporting DONE — the recurring Metal-showcase miss).** A MODERATE
   mixed-hull scene: a static floor + ~48-64 dynamic mixed hulls (tetra/octa/wedge/box) dropped into a settling
   pile. Vulkan dispatches `broad_hull_step.comp` (chunked) + memcmps GPU final world vs CPU `StepHullWorldBPN`;
   Metal runs the CPU path. BOTH render the settled pile (a side-view; reuse the gjk-settle render). Golden =
   `tests/golden/metal/broad_hull.png` (Mac-baked by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly):**
   - **(1) GPU==CPU:** `broad-hull: {bodies:<N>, ticks:<K>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** `broad-hull determinism: two runs BYTE-IDENTICAL`.
   - **(3) THE SCALE PROOF:** `broad-hull scale: {bodies:<N>} broadphase==all-pairs BYTE-IDENTICAL` — CPU
     `StepHullWorldBPN` final == CPU `gjk::StepHullWorldN` (all-pairs) final on the SAME scene, byte-for-byte;
     ALSO assert the pile RESTED (`gjk::MeasureHullStack`: maxSpeed below band, maxPen within band).
   - Golden discipline: ONLY `tests/golden/metal/broad_hull.png`; do NOT commit it. Existing 212 goldens UNTOUCHED.
5. **Cross-backend bar (int64 → strict):** Vulkan GPU==CPU (3-run + timing, no TDR); Metal CPU-ref byte-identical;
   cross-vendor ZERO differing pixels.
6. **Tests — APPEND to `tests/broad_test.cpp`:** `BuildHullAabb` bounds the hull's verts (every world vert inside
   the AABB); the hull broadphase pair set is equivalent to all-pairs over a mixed-hull scene WITH a static floor;
   `StepHullWorldBPN` == `gjk::StepHullWorldN` byte-identical on a moderate scene (the scale/bit-transparency
   proof, CPU-only); the pile rests; deterministic (two runs byte-equal). Clean under `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-broadphase-hull-step` (features) + `--broad-hull-shot` (showcases) +
   update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the controller does.**

## RHI seam additions (summary)
- **None.** The shot rides the existing compute-dispatch + SSBO path (the `gjk-settle`/`broad-convex` seam, incl.
  the chunked-dispatch barrier). `rhi.h` + backend dirs UNCHANGED. `engine/sim/broad.h` APPEND-only (BP1-BP3
  frozen); gjk.h/convex.h/fric.h/persist.h/fpx.h/grain.h + ALL other sim headers + ALL existing shaders UNCHANGED.
  NEW file: `shaders/broad_hull_step.comp.hlsl` only.

## Out of scope (YAGNI)
Lockstep (BP5 — the HullWorld is pure data, the BP3/GJ5 harness retargets over StepHullWorldBP), lit render (BP6).
Broadphase-driven friction/warm-start (deferred — "drops in identically"). BP4 claims ONLY: a deterministic,
bit-exact (CPU↔Vulkan↔Metal) broadphase-driven HULL world step, provably bit-transparent (byte-identical to the
all-pairs hull reference), scaling to dozens of mixed convex hulls, with the integer golden + the three proofs.
CAVEATS inherited: the GJ2-GJ4 within-band (single-point manifold rock, diagonal hull inertia, EPA bound,
floor-cell-size); the broadphase adds ZERO new approximation.

## Verification gate
1. `ctest --preset windows-msvc-debug -R "broad|introspect"` green (the targeted slice tests — the full sweep is
   the flagship-end verify.ps1). Clean under `windows-msvc-asan` (SEPARATE build + test, no chained stale-binary).
2. **proofs + visual:** `--broad-hull-shot` on Vulkan: the 3 proofs + exit 0 under the conan validation layer →
   ZERO VUID. **~3 runs all GPU==CPU (1 tick/dispatch is TDR-safe by construction — NO 30× loop).** VERIFY a
   coherent settled mixed-hull pile.
3. Metal: `visual_test --broad-hull` → `tests/golden/metal/broad_hull.png`; two runs DIFF 0.0000. Confirm
   `visual_test.mm` in the diff; `broad_hull_step.comp` NOT in `hf_gen_msl`. Cross-vendor STRICT ZERO.
4. **Render-invariance:** ONLY `broad_hull.png` added; the other 212 byte-identical (+ the controller introspect
   rebake).
5. Introspect: exactly `+deterministic-broadphase-hull-step` + `--broad-hull-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` + BP1-BP3 broad.h code + gjk.h/convex.h/fric.h/persist.h/fpx.h/grain.h + ALL other
   sim headers + ALL existing shaders byte-unchanged; broad.h APPEND-only; one new Vulkan-only shader, no RHI
   change). `broad_hull_step.comp` NOT in `hf_gen_msl`.
