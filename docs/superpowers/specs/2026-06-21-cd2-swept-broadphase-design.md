# Slice CD2 — Deterministic Integer CCD: THE SWEPT-AABB BROADPHASE — Design

> Autonomous-session spec. Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SECOND slice of
> FLAGSHIP #24 (DETERMINISTIC INTEGER CCD, `hf::sim::ccd`). CD1 built the time-of-impact primitive
> (`ConservativeAdvance`). CD2 builds the broadphase that feeds it: a **swept-AABB** candidate-pair generator —
> each body's axis-aligned box is the union of its box at the tick's START and END poses, so a FAST mover's
> sweep overlaps obstacles the discrete (instantaneous-AABB) broadphase would miss. The pair generation over the
> swept boxes reuses the frozen #23 broadphase grid + the count→scan→emit compaction with ZERO new broadphase
> code — and is PURE int32, so the shaders MSL-GENERATE NATIVELY (a true GPU pass on both backends, the strongest
> tier). The make-or-break is the **EQUIVALENCE/SUPERSET PROOF**: the swept-pair set ⊇ every pair a fast mover
> actually needs — a pair the discrete broadphase MISSES (because the bodies don't overlap at either endpoint
> instant but the sweep crosses) is PRESENT in the swept set. APPEND to `engine/sim/ccd.h` (CD1 + broad.h/gjk.h/
> convex.h/fpx.h/fric.h/persist.h/grain.h BYTE-FROZEN). Branch: `slice-cd2`. See [[hazard-forge-ccd-roadmap]],
> [[hazard-forge-docs-style]].

**Goal:** Extend `engine/sim/ccd.h` (additive — CD1 byte-unchanged) with `SweptHullAabb` (the union of
`broad::BuildHullAabb` at the start pose and the integrated end pose) + `BuildSweptBroadphasePairs` (the #23
broadphase grid + candidate-pair generator run over the swept AABBs) + `SweptPairsSupersetOfDiscrete` (the
superset check). Add the pure-int32 GPU shaders `shaders/ccd_swept_{count,scan,emit}.comp.hlsl` (MSL-native — IN
`hf_gen_msl`), plus the showcase `--ccd-swept-shot` (Vulkan) / `--ccd-swept` (Metal). Bake the integer golden
`ccd_swept`.

## Design call: swept AABB = union(start, end); grid+pair over the swept boxes is pure int32 (MSL-native)

- **`SweptHullAabb(hull, body, dt) → fpx::FxAabb`** — `broad::BuildHullAabb(hull, body)` at the START pose ∪
  `broad::BuildHullAabb(hull, bodyEnd)` where `bodyEnd` = a COPY of `body` integrated forward by `dt`
  (`fpx::IntegrateBodyFull`, ZERO gravity — sweep along the current velocity). The union is per-axis min/max. The
  end-pose support queries are int64 (the int64 BOUNDARY); but the RESULT is an `FxAabb` of int32 Q16.16 coords.
- **The grid + pair passes are PURE int32 over the swept AABBs.** The candidate-pair generation — cell assignment
  (`FloorDiv`), the 27-cell stencil, the canonical `i<j` de-dup, the `fpx::AabbOverlap` predicate, count→scan→emit
  — is byte-for-byte the #23 `broad::BuildBroadphasePairs`/`BuildHullBroadphasePairsWithStatics` machinery, just
  keyed on the SWEPT AABBs instead of the instantaneous ones. So **`BuildSweptBroadphasePairs(world, dt, ...)`** =
  compute the swept AABB per body, then delegate to the frozen broadphase pair generator over those AABBs (reuse
  it verbatim — pass the swept AABBs in, OR set each body's effective broadphase box to the swept box; document
  the exact reuse). Cell size ≥ the max SWEPT-AABB diameter so the ±1 stencil stays exact; statics handled by the
  #23 dynamic-vs-static pass.
- **The int32/int64 split (the proof-tier call).** The swept-AABB PRECOMPUTE (the int64 part: integrate end pose
  + 2× `BuildHullAabb` + union) is computed on the HOST/CPU and fed identically to the CPU reference AND the GPU
  (the established "int64 boundary done on the CPU-ref, int32 keyed on the GPU" pattern). The
  `ccd_swept_{count,scan,emit}.comp` shaders take the precomputed swept AABBs (int32 lo/hi coords) as SSBO input
  and do the PURE int32 grid+pair work → **MSL-native on both backends** (the `broad_pair_*` tier), strict
  zero-differing-pixel cross-vendor. The GPU==CPU memcmp is over the int32 grid+pair output (the pair list +
  offsets), exactly like BP2.

## Reuse map (file:line)
- **CD1 `engine/sim/ccd.h` (APPEND after CD1's `MeasureCcdToi`):** the CD1 types, the `dt` config. CD1 frozen.
- **broad.h (read-only — REUSE verbatim):** `broad::BuildHullAabb` (broad.h:583 — the per-body world AABB),
  `broad::InflateAabb` (:608), `broad::BuildBroadphasePairs`/`BuildHullBroadphasePairsWithStatics` (:637 — the
  candidate-pair generator to run over the swept AABBs), `broad::MakeBodyGrid`, `broad::PairSetEquivalentToAllPairs`
  (the equivalence-proof idiom), `fpx::FxAabb`/`AabbOverlap`/`FxPair`. The `broad_pair_{count,scan,emit}.comp`
  (the MSL-native int32 shader pattern to mirror). Do NOT modify broad.h/gjk.h/fpx.h/etc — BYTE-FROZEN.
- **fpx.h (read-only):** `fpx::IntegrateBodyFull` (the end-pose integrate), `gjk::HullWorld`/`gjk::FxHull`.
- **The proof-tier convention (PURE int32 → MSL-NATIVE):** `ccd_swept_{count,scan,emit}.comp.hlsl` go IN
  `hf_gen_msl` (the `broad_pair_*` rows are the template) — true GPU both backends.
- **The showcase + shader precedent:** BP2's `--broad-pair` (the 3-shader dispatch + GPU==CPU memcmp +
  equivalence proof + diagnostic). Mirror for `--ccd-swept`.
- **Registration:** `scripts/verify.ps1` (append `ccd_swept` + `--ccd-swept-shot` to `$vkShots`),
  `metal_headless/CMakeLists.txt` (`hf_gen_msl` — ADD `ccd_swept_{count,scan,emit}.comp`, MSL-native),
  `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller rebakes the JSON golden**), append to
  `tests/ccd_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/sim/ccd.h`** (CD1 byte-frozen): `SweptHullAabb`, `BuildSweptBroadphasePairs`,
   `SweptPairsSupersetOfDiscrete`, `SweptPairMeasure` + `MeasureSweptPairs`. Pure int32 grid+pair (the swept-AABB
   precompute is the int64 boundary). NO new RHI; three new MSL-native shaders.
2. **New shaders `shaders/ccd_swept_{count,scan,emit}.comp.hlsl` (PURE int32, MSL-NATIVE → IN `hf_gen_msl`)** —
   reproduce the grid+pair generation over the (host-precomputed) swept AABBs byte-for-byte. Read the swept AABBs
   + grid from SSBOs; write the pair list + offsets. NO new RHI.
3. **Showcase `--ccd-swept-shot <out>` (Vulkan) AND `--ccd-swept` (Metal) — WIRE BOTH (grep your own
   visual_test.mm for `--ccd-swept` before reporting DONE).** A scene with FAST movers: dynamic bodies with large
   per-tick velocities whose sweeps cross obstacles they don't instantaneously overlap, plus some slow/clustered
   bodies. Vulkan dispatches the 3 `ccd_swept` shaders (over the host-precomputed swept AABBs) + memcmps GPU pair
   list vs CPU `BuildSweptBroadphasePairs`; Metal runs the GPU shaders too (MSL-native). BOTH render an integer
   diagnostic (top-down XZ: each body's swept-AABB footprint + a segment per emitted swept pair). Golden =
   `tests/golden/metal/ccd_swept.png` (Mac-baked by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU:** `ccd-swept: {bodies:<N>, pairs:<P>} GPU==CPU BIT-EXACT` — the GPU pair list + offsets == the
     CPU `BuildSweptBroadphasePairs` byte-for-byte; assert.
   - **(2) determinism:** `ccd-swept determinism: two runs BYTE-IDENTICAL`.
   - **(3) THE SUPERSET PROOF:** `ccd-swept superset: {sweptPairs:<S>, fastMissedByDiscrete:<M>} swept⊇discrete
     BYTE-CHECKED` — the swept-pair set CONTAINS every pair the discrete (instantaneous-AABB)
     `broad::BuildHullBroadphasePairs` produces, AND it additionally contains the fast-mover pairs the discrete
     set MISSES (a sweep that crosses an obstacle non-overlapping at both endpoints) — `M > 0` for the fast-mover
     scene, proving the swept broadphase catches what the discrete one would tunnel past; assert both (`swept ⊇
     discrete` AND `M > 0`).
   - **Golden discipline: ONLY `tests/golden/metal/ccd_swept.png`; do NOT commit it.** Existing 216 goldens
     UNTOUCHED.
5. **Cross-backend bar (int32 MSL-native → strict):** Vulkan GPU==CPU AND Metal GPU==CPU bit-exact; cross-vendor
   ZERO differing pixels.
6. **Tests — APPEND to `tests/ccd_test.cpp`:** `SweptHullAabb` contains both the start and end world hull verts;
   `BuildSweptBroadphasePairs` is a canonical i<j set; `SweptPairsSupersetOfDiscrete` is TRUE (swept ⊇ discrete)
   for several scenes; for a fast mover, the swept set contains a pair the discrete set lacks (M>0); deterministic.
   Clean under `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-ccd-swept` (features) + `--ccd-swept-shot` (showcases) + update
   `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the controller does.**

## RHI seam additions (summary)
- **None.** The shot rides the existing compute-dispatch + SSBO path (the `broad_pair` seam). `rhi.h` + backend
  dirs UNCHANGED. `engine/sim/ccd.h` APPEND-only (CD1 frozen); broad.h/gjk.h/convex.h/fpx.h/fric.h/persist.h/
  grain.h + ALL other sim headers + ALL existing shaders UNCHANGED. NEW files:
  `shaders/ccd_swept_{count,scan,emit}.comp.hlsl` only. Report the seam: three new MSL-native shaders, no RHI
  change, no frozen-file edit, ccd.h append-only.

## Out of scope (YAGNI — later slices)
The substepped CCD world step (CD3 — consumes the swept pairs), the bullet-wall beat (CD4), lockstep (CD5), lit
render (CD6). CD2 claims ONLY: a deterministic, bit-exact (CPU↔Vulkan↔Metal, both GPU-native) swept-AABB
candidate-pair generator that is a PROVABLE SUPERSET of the discrete broadphase (catching fast-mover pairs the
discrete one misses), with the integer golden + the three proofs. CAVEATS: the swept AABB is conservative (a
straight-line sweep box; a rotating body's true swept volume is larger than the union of two pose-AABBs, so add a
documented margin or note the limit); convex hulls only; the swept AABBs inflate the candidate count.

## Verification gate
1. `ctest --preset windows-msvc-debug -R "ccd|broad|introspect"` green. Clean under `windows-msvc-asan` (separate
   build + test).
2. **proofs + visual:** `--ccd-swept-shot` on Vulkan: the 3 proofs (incl. the superset) + exit 0, under the conan
   validation layer → ZERO VUID. **VERIFY the diagnostic shows the swept-AABB footprints + the swept-pair
   segments, with fast movers connected to obstacles their instantaneous box wouldn't reach.**
3. Metal: `visual_test --ccd-swept` → `tests/golden/metal/ccd_swept.png`; two runs DIFF 0.0000. Confirm
   `visual_test.mm` in the diff; `ccd_swept_*.comp` ARE in `hf_gen_msl`. Cross-vendor STRICT ZERO.
4. **Render-invariance:** ONLY `ccd_swept.png` added; the other 216 byte-identical (+ controller introspect rebake).
5. Introspect: exactly `+deterministic-ccd-swept` + `--ccd-swept-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` + CD1 ccd.h code + broad.h/gjk.h/convex.h/fpx.h/fric.h/persist.h/grain.h + ALL other
   sim headers + ALL existing shaders byte-unchanged; ccd.h APPEND-only; three new MSL-native shaders, no RHI
   change). `ccd_swept_{count,scan,emit}.comp` ADDED to `hf_gen_msl`.
