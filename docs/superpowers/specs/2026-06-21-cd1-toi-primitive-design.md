# Slice CD1 — Deterministic Integer CCD: THE TIME-OF-IMPACT PRIMITIVE (conservative advancement core) — Design

> Autonomous-session spec. Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIRST slice (the
> BEACHHEAD) of FLAGSHIP #24 (DETERMINISTIC INTEGER CONTINUOUS COLLISION DETECTION via conservative advancement,
> `hf::sim::ccd`). A discrete solver lets a fast/thin body TUNNEL through geometry in one tick. CD1 builds the
> core primitive that prevents it: a deterministic integer **time-of-impact (TOI)** between a moving pair of
> convex hulls, computed by **conservative advancement** — repeatedly query the closest distance (the FROZEN
> `gjk::Gjk`, which already returns the gap + witnesses for a separated pair), advance the bodies by a step
> guaranteed not to overshoot, and stop when the gap closes. Bit-identical CPU↔Vulkan↔Metal. NEW header
> `engine/sim/ccd.h`, namespace `hf::sim::ccd`, `#include "sim/broad.h"` READ-ONLY (transitively freezing
> gjk/convex/fric/persist/fpx/grain). Branch: `slice-cd1`. See [[hazard-forge-ccd-roadmap]],
> [[hazard-forge-docs-style]].

**Goal:** Create `engine/sim/ccd.h` with `FxToi` (the {toi, hit} result) + `ClosingSpeedBound` (the conservative
upper-bound-on-closing-speed) + `ConservativeAdvance(hullA, bodyA, hullB, bodyB, dt)` (the integer TOI loop using
each body's own `vel`/`angVel`) + a `CcdToiMeasure` summary. Add the int64 GPU shader `shaders/ccd_toi.comp.hlsl`
(Vulkan-only, copies `ConservativeAdvance` verbatim) + the showcase `--ccd-toi-shot` (Vulkan) / `--ccd-toi`
(Metal). Bake the integer golden `ccd_toi`.

## Design call: conservative advancement = advance by gap / (conservative max closing speed), never overshoot

For a separated convex pair, `gjk::Gjk(hullA, bodyA, hullB, bodyB)` returns `separation` (origin → closest point
of the Minkowski difference A−B; `FxLength(separation)` = the gap) + `closestA`/`closestB` (the witness points).
The contact normal is `n = FxNormalize(separation)`. Conservative advancement advances time by a step that
provably cannot close the gap, so it never tunnels past the true impact:
- **The conservative closing-speed bound (the determinism + correctness CRUX — spell it out).** The fastest the
  gap can close is bounded above by the linear closing speed PLUS the angular contribution of both bodies:
  `bound = FxDot(relVelLin, n) + |angVelA|·rMaxA + |angVelB|·rMaxB`, where `relVelLin = bodyB.vel − bodyA.vel`,
  `|angVelA| = FxLength(bodyA.angVel)`, and `rMaxA` = the max distance from `bodyA.pos` to any of `hullA`'s world
  vertices (the body's bounding radius — compute from the hull verts via the support extents, or `max FxLength(worldVert − pos)`).
  **The angular terms are MANDATORY** — without them a rotating pair's gap can close faster than the linear bound
  predicts and the loop OVERSHOOTS the true TOI (the one subtle correctness bug; CD1's hand-check MUST include a
  rotating case). If `bound ≤ 0` → the pair is RECEDING (or rotating apart) → no impact this dt → return
  `{toi = dt, hit = 0}`.
- **The advance step:** `advance = fxdiv(gap, bound)` — the time to close the gap at the maximum possible closing
  speed = a LOWER bound on the true TOI (so it never overshoots). Accumulate `t += advance`; integrate a COPY of
  both bodies forward by `advance` (`fpx::IntegrateBodyFull` with sub-dt = `advance`, NO gravity — CCD advances
  along the current velocity only; the resolve happens later in CD3). Re-`Gjk` at the new pose. Repeat until
  `gap ≤ kContactEps` (→ `{t, hit = 1}`) OR `t ≥ dt` (→ `{dt, hit = 0}`) OR a fixed `kToiMaxIter` bound is hit
  (→ the current `{t, hit}` — deterministic, within-band; document the band). If `Gjk` reports OVERLAP at the
  start (already touching) → `{toi = 0, hit = 1}`.
- **`FxToi { fx toi; uint32_t hit; uint32_t iterations; }`** — the result (`hit` 0/1 for the std430 GPU mirror).
- **`ConservativeAdvance(const gjk::FxHull& hullA, const fpx::FxBody& bodyA, const gjk::FxHull& hullB, const
  fpx::FxBody& bodyB, fx dt) → FxToi`** — the loop above, pure integer, FIXED iteration bound. Uses `bodyA.vel`/
  `bodyA.angVel` (each body carries its own velocity). Statics (zero vel + zero angVel) contribute nothing.
- **`ClosingSpeedBound(bodyA, rMaxA, bodyB, rMaxB, n) → fx`** — the conservative bound helper (factored out so the
  test can hand-check it). `BodyMaxRadius(hull, body)` — the max `FxLength(worldVert − body.pos)` over the hull's
  world verts (reuse `gjk::Support`/the hull verts; int64).

## Reuse map (file:line — the implementer MUST ground these before coding)
- **gjk.h (read-only — REUSE verbatim):** `gjk::Gjk` (gjk.h:457 — the closest-distance source) + `gjk::GjkResult`
  (gjk.h:257: `overlap`, `separation`, `closestA`, `closestB`) + `gjk::FxHull` + `gjk::Support`. **The core CCD
  primitive is a frozen `Gjk` call.** Do NOT modify gjk.h/broad.h/convex.h/fpx.h/fric.h/persist.h/grain.h —
  BYTE-FROZEN.
- **fpx.h (read-only):** `fpx::FxBody` (fpx.h:116 — pos/vel/invMass/flags/radius/orient/angVel),
  `fpx::IntegrateBodyFull` (fpx.h:479 — the sub-dt advance; confirm its signature: it integrates pos from vel +
  orient from angVel over a dt, with a gravity arg — pass ZERO gravity for the TOI advance), `fpx::FxLength`
  (:96), `fpx::FxNormalize` (:319), `fpx::fxdiv` (:311), `convex::FxDot`.
- **The proof-tier convention (convex.h:16-23):** int64 (Gjk's `FxDot`/`FxRotate`/`fxdiv`) →
  `shaders/ccd_toi.comp.hlsl` is **VULKAN-SPIR-V-ONLY** (NOT in `hf_gen_msl`); Metal `--ccd-toi` runs the CPU
  `ConservativeAdvance` → byte-identical by construction; the Vulkan side carries the GPU==CPU memcmp. The shader
  copies `ConservativeAdvance` verbatim; one GPU thread per candidate pair. The `gjk_distance.comp` precedent.
- **The showcase + shader precedent:** GJ2's `--gjk-distance-shot` (the int64 one-thread-per-pair compute proof +
  GPU==CPU memcmp + the integer diagnostic render). Mirror for `--ccd-toi`.
- **Registration:** `scripts/verify.ps1` (append `ccd_toi` + `--ccd-toi-shot` to `$vkShots`),
  `metal_headless/CMakeLists.txt` (`hf_gen_msl` — do NOT add `ccd_toi.comp`; int64 Vulkan-only),
  `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller rebakes the JSON golden**), a NEW
  `tests/ccd_test.cpp` (+ register in the test CMake like `broad_test`).

## Design decisions (locked)
1. **NEW header `engine/sim/ccd.h`** (namespace `hf::sim::ccd`, `#include "sim/broad.h"` read-only): `FxToi`,
   `kContactEps`, `kToiMaxIter`, `BodyMaxRadius`, `ClosingSpeedBound`, `ConservativeAdvance`, `CcdToiMeasure` +
   `MeasureCcdToi`. Pure integer, FIXED iteration bound. NO new RHI; one new Vulkan-only shader.
2. **New shader `shaders/ccd_toi.comp.hlsl` (int64, VULKAN-ONLY)** — copies `ConservativeAdvance` (incl. the
   embedded `Gjk`) verbatim; one thread per candidate pair; writes the `FxToi` to an SSBO. NOT in `hf_gen_msl`.
3. **Showcase `--ccd-toi-shot <out>` (Vulkan) AND `--ccd-toi` (Metal) — WIRE BOTH (grep your own visual_test.mm
   for `--ccd-toi` before reporting DONE — the recurring Metal-showcase miss).** A fixed set of pairs: a body
   moving straight toward a static obstacle (a clean hit at a hand-computable TOI), a body moving AWAY (receding →
   no hit), a body moving tangentially (grazing/no hit), and a ROTATING body approaching (the angular-bound case).
   Vulkan dispatches `ccd_toi.comp` + memcmps the GPU `FxToi[]` vs the CPU `ConservativeAdvance`; Metal runs the
   CPU path. BOTH render an integer diagnostic (each pair: the mover's start pose, its swept path to the TOI pose,
   marked against the obstacle). Golden = `tests/golden/metal/ccd_toi.png` (Mac-baked by the CONTROLLER — DO NOT
   commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU:** `ccd-toi: {pairs:<P>, hits:<H>} GPU==CPU BIT-EXACT` — the GPU `FxToi[]` == the CPU one
     byte-for-byte; assert.
   - **(2) determinism:** `ccd-toi determinism: two runs BYTE-IDENTICAL`.
   - **(3) closed-form correctness:** for the straight-approach pair (known velocity, known gap), the returned
     `toi` matches the hand-computed Q16.16 value within the documented band; AND the ROTATING-approach pair's
     TOI is ≤ the true impact (never overshoots — the conservative guarantee; a small clearance check at the TOI
     pose confirms `Gjk` reports the pair still separated-or-touching, NOT overlapping). Print `ccd-toi correct:
     {straightInBand:true, rotatingNoOvershoot:true}`; assert.
   - **(4) no false impact:** the receding + tangential pairs return `hit=0, toi=dt`. Print `ccd-toi receding:
     {hit:0}`; assert.
   - **Golden discipline: ONLY `tests/golden/metal/ccd_toi.png`; do NOT commit it.** Existing 215 image goldens
     UNTOUCHED.
5. **Cross-backend bar (int64 → strict):** Vulkan GPU==CPU bit-exact; Metal CPU-ref byte-identical; cross-vendor
   ZERO differing pixels.
6. **Tests — NEW `tests/ccd_test.cpp`:** `ConservativeAdvance` straight-approach matches a hand-computed TOI
   within band; the ROTATING-approach TOI does not overshoot (the TOI pose is non-overlapping per `Gjk`); a
   receding pair → `hit=0, toi=dt`; an already-overlapping pair → `toi=0, hit=1`; `ClosingSpeedBound` is
   conservative (≥ the true projected closing speed for a hand case incl. rotation); deterministic (two calls
   byte-equal). Clean under `windows-msvc-asan`. Register `ccd_test` in the test CMake.
7. **Introspect.** Add exactly `deterministic-ccd-toi` (features) + `--ccd-toi-shot` (showcases) +
   update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the controller does.**

## RHI seam additions (summary)
- **None.** The shot rides the existing compute-dispatch + SSBO path (the `gjk_distance` seam). `rhi.h` + backend
  dirs UNCHANGED. gjk.h/broad.h/convex.h/fpx.h/fric.h/persist.h/grain.h + ALL other sim headers + ALL existing
  shaders UNCHANGED. NEW files only: `engine/sim/ccd.h`, `shaders/ccd_toi.comp.hlsl`, `tests/ccd_test.cpp` (+ the
  showcase/introspect/verify edits). Report the seam: one new Vulkan-only shader, no RHI change, no frozen-file edit.

## Out of scope (YAGNI — later slices)
The swept-AABB broadphase (CD2), the substepped CCD world step (CD3), the bullet-wall beat (CD4), lockstep (CD5),
lit render (CD6). CD1 claims ONLY: a deterministic, bit-exact (CPU↔Vulkan↔Metal) conservative-advancement TOI for
a moving convex pair, with the integer golden + the four proofs. CAVEATS: conservative advancement is iterative
(the TOI is bounded above by the true impact, converges from below; the fixed `kToiMaxIter`/`kContactEps` budget
can stop sub-ε before contact — a documented within-band gap, the GJ2-GJ4 EPA-band lineage); convex hulls only;
the closing-speed bound's angular term uses the body bounding radius (conservative — may take more substeps for
grazing/rotating pairs).

## Verification gate
1. `ctest --preset windows-msvc-debug -R "ccd|broad|introspect"` green (the targeted slice tests — the full sweep
   is the flagship-end verify.ps1). Clean under `windows-msvc-asan` (SEPARATE build + test).
2. **proofs + visual:** `--ccd-toi-shot` on Vulkan: the 4 proofs + exit 0, under the conan validation layer →
   ZERO VUID. **VERIFY the diagnostic shows each mover's swept path arrested at the TOI pose against the obstacle
   (the receding ones travel free), coherent, no garbage/NaN.**
3. Metal: `visual_test --ccd-toi` → new golden `tests/golden/metal/ccd_toi.png`; two runs DIFF 0.0000. **Confirm
   `visual_test.mm` in the diff; confirm `ccd_toi.comp` is NOT in `hf_gen_msl`.** Cross-vendor STRICT ZERO.
4. **Render-invariance:** ONLY `ccd_toi.png` added; the other 215 byte-identical (+ controller introspect rebake).
5. Introspect: exactly `+deterministic-ccd-toi` + `--ccd-toi-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` + gjk.h/broad.h/convex.h/fpx.h/fric.h/persist.h/grain.h + ALL other sim headers + ALL
   existing shaders byte-unchanged; one new Vulkan-only shader, no RHI change). `ccd_toi.comp` NOT in `hf_gen_msl`.
