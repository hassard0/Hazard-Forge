# Slice PS3 — Deterministic Persistent Contacts: THE WARM-STARTED CONE SOLVER — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The THIRD slice of FLAGSHIP #21
> (DETERMINISTIC WARM-STARTED CONTACT CACHING + SLEEPING ISLANDS, `hf::sim::persist`). PS1 built the contact key,
> PS2 the persistent cache. PS3 is the **warm-started solver**: an ACCUMULATED sequential-impulse friction solve
> that seeds each contact's impulse accumulators from last tick's cached values and re-applies them, so a resting
> stack converges in far fewer iterations and rests tighter — the convergence + scale aid every shipping engine
> uses, here made bit-identical CPU/Vulkan/Metal. INTEGER-bit-exact. int64 → the `persist_warm.comp` shader is
> Vulkan-only + a Metal CPU reference. PS1/PS2's `persist.h` code + CX/FC's `convex.h`/`fric.h` are BYTE-FROZEN
> (PS3 is additive). Branch: `slice-ps3`. See [[hazard-forge-persist-roadmap]].

**Goal:** Extend `engine/sim/persist.h` (additive — PS1/PS2 + fric.h byte-unchanged) with the warm-started solver:
`SolveFrictionWarm` (the accumulated cone solver) + `StepWarmWorld` (the FC4 step with the warm solver + the cache
maintained tick-to-tick) + `StepWarmWorldN`. Add the new int64 shader `shaders/persist_warm.comp.hlsl` +
`--persist-warm-shot` (Vulkan) / `--persist-warm` (Metal). Bake the integer golden `persist_warm`. **NO new RHI.**

## Design call: accumulated sequential impulses with warm-start priming

FC3's `SolveFrictionImpulse` is NON-accumulated — it computes a fresh impulse each sweep and applies it. That cannot
be warm-started (re-applying last tick's impulse on top of a fresh full impulse would DOUBLE-COUNT). PS3 uses the
proper **accumulated** form: each contact point's `normalImpulse`/`tangentImpulse1`/`tangentImpulse2` are
ACCUMULATORS holding the TOTAL impulse so far; the cone clamps the TOTAL; each sweep applies only the DELTA.
- **Seed (warm-start):** the accumulators arrive seeded — from the PS2 cache (`MatchCache`) for a matched contact,
  or zero for a cold one. **PRIME ONCE** at the start of the solve: for each point apply the seeded total impulse
  `J = n·normalImpulse + t1·tangentImpulse1 + t2·tangentImpulse2` to both bodies (the standard warm-start — re-inject
  last tick's converged impulse so the velocities start near the solved state).
- **Accumulated sweeps:** `iters` Gauss-Seidel sweeps; per point, NORMAL then t1 then t2 (the FC3 order), but each
  is accumulated:
  - NORMAL: `djn = fxdiv(−fxmul(kOne+restitution, vn), kn)` (the FC3 numerator); `newTotal = max(0,
    normalImpulse + djn)`; `applied = newTotal − normalImpulse`; apply `J = n·applied`; `normalImpulse = newTotal`.
  - TANGENT t (t1 then t2): `djt = fxdiv(−vt, kt)`; `newTotal = clamp(tangentImpulse + djt, −fxmul(mu,
    normalImpulse), +fxmul(mu, normalImpulse))` (the cone vs the ACCUMULATED normal); `applied = newTotal −
    tangentImpulse`; apply `Jt = t·applied`; `tangentImpulse = newTotal`. (Recompute the contact-point velocities
    after each apply — sequential impulse.) The effective masses `kn`/`kt`, the lever arms, the inertia-tensor
    apply are EXACTLY FC3's (`fric::SolveFrictionImpulse`, reused in form).
  After the sweeps the accumulators hold the TOTAL impulse — ready for `UpdateCache` to store for next tick.
- **`SolveFrictionWarm(bodyA, bodyB, invIaW, invIbW, keyedManifold, restitution, mu, iters)`** — the prime + the
  accumulated sweeps, mutating body vel/angVel + the `keyed.fm` accumulators.
- **`StepWarmWorld(world, cache, cfg)`** — the FC4 `fric::StepFrictionWorld` tick REPRODUCED with two changes: (1)
  per overlapping pair, `BuildKeyedManifold` → `MatchCache(cache, keyed)` (seed the accumulators from last tick) →
  `SolveFrictionWarm` (instead of `SolveFrictionImpulse`); (2) AFTER the solve sweeps + the position de-penetration,
  rebuild the cache: a fresh `PersistentCache` accumulating every pair's `UpdateCache`-style entries, swapped into
  `cache` for next tick. The cache PERSISTS across ticks (passed in/out). `StepWarmWorldN(world, cache, cfg, ticks)`.

**DESIGN NOTE (the honest control, NOT "==FC3"):** the accumulated GS is a DIFFERENT algorithm from FC3's
non-accumulated GS (they share the cone + inertia + effective-mass math). So PS3 is NOT byte-identical to FC3 even
cold. The make-or-break controls are instead: **(a) WARM-START BENEFIT** — at a fixed LOW iteration count, the
warm-started stack's max residual relative velocity is STRICTLY LESS than the cold-started one (warm converges
faster); **(b) CONSISTENCY** — warm and cold reach the SAME state byte-identically at a HIGH iteration count
(accumulated GS has a unique fixed point; once converged the impulses stop changing).

**THE int64 REALITY:** the whole chain is int64. `persist_warm.comp` is **VULKAN-SPIR-V-ONLY (NOT in hf_gen_msl)**,
single-thread over the small world (the convex_step/fric_step convention); the Metal `--persist-warm` runs the CPU
`StepWarmWorldN` — byte-identical to the Vulkan GPU result BY CONSTRUCTION, the Vulkan side carries the GPU==CPU
memcmp.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **PS2 `engine/sim/persist.h` (read it; APPEND only after `MeasureCache`):** `PersistentCache`,
  `KeyedFrictionManifold`, `BuildKeyedManifold`, `MatchCache`, `UpdateCache`, `CachedContact`. PS1/PS2 byte-frozen.
- **fric.h (read-only — do NOT edit):** `fric::SolveFrictionImpulse` (`fric.h:307` — the cone + inertia + effective-
  mass body PS3's accumulated solver reproduces with accumulation), `fric::FrictionManifold`/`FrictionPoint` (the
  accumulators), `fric::StepFrictionWorld` (`fric.h:443` — the FC4 5-pass shell PS3's `StepWarmWorld` reproduces),
  `fric::FrictionStepConfig`. **DO NOT modify fric.h.**
- **convex.h (read-only):** `convex::ConvexWorld`, `IsDynamic`, `BoxSatStable`, `FxBoxInvInertiaBody`/
  `WorldInvInertia`, `FxMat3MulVec`, the position de-penetration loop (`convex.h:907-932`). **DO NOT modify.**
- **The shader + showcase precedent:** FC4's `shaders/fric_step.comp.hlsl` (the int64 Vulkan-only single-thread
  whole-world step + the GPU==CPU final-state memcmp) and the `--fric-stack-shot`/`--fric-stack` showcases.
  `persist_warm.comp` is the twin (copies `StepWarmWorldN` + the per-tick cache). Confirm `persist_warm` NOT in
  hf_gen_msl.
- **Registration:** `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller
  rebakes the JSON golden — do NOT**), append to `tests/persist_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/sim/persist.h`** (PS1/PS2 byte-frozen): `SolveFrictionWarm` (the accumulated cone solver with
   priming), `StepWarmWorld(world, cache, cfg)` (the FC4 step + the warm solver + the per-tick cache match/update),
   `StepWarmWorldN(world, cache, cfg, ticks)`, a `WarmMeasure` (max residual vel). `cfg` reuses
   `fric::FrictionStepConfig` (or a thin alias). Pure integer, FIXED orders. **NEW shader** `persist_warm.comp.hlsl`
   (int64, Vulkan-only, one thread runs the whole `StepWarmWorldN` + cache — copies it VERBATIM). NOT in hf_gen_msl.
2. **Showcase `--persist-warm-shot <out>` (Vulkan) AND `--persist-warm` (Metal) — WIRE BOTH** (standalone
   arg-parse). The SCENE: the FC4 friction stack (static floor + 3-4 dynamic boxes). Run TWO sims K ticks: WARM
   (`StepWarmWorldN` — the cache persists) and COLD (`StepWarmWorldN` with the cache FORCE-CLEARED each tick, OR the
   accumulated solver with a zero seed each tick — same accumulated algorithm, no warm-start). At a fixed LOW K
   (e.g. K=30, low solveIters) the WARM stack rests tighter (lower max speed); at a HIGH K both converge to the
   same rest. Vulkan: the GPU `persist_warm.comp` (warm) → **memcmp the GPU final body world vs the CPU
   `StepWarmWorldN`** (NO tolerance). Metal: the CPU reference. Render the settled warm stack (the FC4 side-view).
   Golden = `tests/golden/metal/persist_warm.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU warm final body world == the CPU `StepWarmWorldN` byte-for-byte. Print
     `persist-warm: {bodies:<N>, ticks:<K>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `persist-warm determinism: two runs BYTE-IDENTICAL`.
   - **(3) warm-start benefit:** at the fixed low K, the WARM stack's max residual speed is STRICTLY LESS than the
     COLD stack's (warm converged tighter/faster). Print `persist-warm benefit: {warmMaxSpeed:<w>, coldMaxSpeed:<c>,
     warmTighter:true}`; assert `w < c`.
   - **(4) consistency:** warm and cold reach a BYTE-IDENTICAL body world at a high K (the unique fixed point —
     accumulated GS converged). Print `persist-warm consistency: {warmEqualsColdAtConvergence:true}`; assert the
     memcmp. (If exact byte-identity at high K proves flaky in fixed point, fall back to "within a tight integer
     epsilon" and DOCUMENT it honestly — the controller judges; prefer the clean byte-identity if it holds.)
   - **Golden discipline: ONLY `tests/golden/metal/persist_warm.png`; do NOT commit it.** Existing 198 image goldens
     UNTOUCHED.
4. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels.
5. **Tests — APPEND to `tests/persist_test.cpp` (pure CPU):** `SolveFrictionWarm` with a zero seed leaves the
   accumulators as the solved totals; a warm seed primes the bodies (the prime moves velocity); a stack solved warm
   at low K has lower residual than cold; warm and cold agree at high K; the accumulated normal impulse stays ≥0 and
   the tangent within `±μ·jn`; two runs byte-identical. Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-persist-warm` (features) + `--persist-warm-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + read-back path. `rhi.h` + backend dirs UNCHANGED.
  `engine/sim/convex.h` + `fric.h` + `fpx.h` + **PS1/PS2's persist.h code + persist_key.comp + persist_cache.comp**
  + all other sim headers + `engine/nav/` + `engine/anim/` + `engine/physics/` + all EXISTING shaders UNCHANGED. The
  ONLY new shader is `persist_warm.comp.hlsl` (int64, Vulkan-only, NOT in hf_gen_msl). `persist.h` APPEND-only.
  Report the seam empty.

## Out of scope (YAGNI — later PS slices)
Sleeping islands (PS4 — the new physics; PS3's warm world step always integrates+solves every body), lockstep (PS5),
the lit 3D render (PS6). PS3 claims ONLY: a deterministic integer accumulated warm-started cone solver + warm world
step that converges a stack tighter at fixed iterations and agrees with the cold solve at convergence, bit-identical
CPU↔Vulkan↔Metal, with the integer golden + the four proofs. NOTE: boxes only; the warm-start is a CONVERGENCE aid
(matched contacts only — a sliding contact's changed key cold-starts, the documented caveat); accumulated GS residual
not analytic (the within-a-band honesty); int64 → Vulkan-GPU + Metal-CPU-ref.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 107 incl. PS1/PS2's `persist_test` + the appended PS3 cases).
   Clean under `windows-msvc-asan` (build+run `persist_test` + `introspect_test`).
2. **proofs + visual:** `--persist-warm-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID. **VERIFY the image shows a coherent settled warm stack (a resting tower, not collapsed/scattered).**
3. Metal: `visual_test --persist-warm` → new golden `tests/golden/metal/persist_warm.png`; two runs DIFF 0.0000.
   **Confirm `visual_test.mm` in the diff; confirm `persist_warm.comp` NOT in `hf_gen_msl`.** Cross-vendor STRICT
   ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `persist_warm.png` added; the other
   198 byte-identical. `git diff master --stat -- tests/golden` = ONLY `persist_warm.png` (metal) + the introspect
   json (controller rebake, post-gate).
5. Introspect: exactly `+deterministic-persist-warm` + `--persist-warm-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/convex.h`/`fric.h`/`fpx.h` + **PS1/PS2's persist.h code +
   persist_key.comp + persist_cache.comp** + ALL other sim headers + `engine/nav/` + `engine/anim/` +
   `engine/physics/` + ALL existing shaders byte-unchanged). `scripts/verify.ps1` updated: `persist_warm` golden in
   the Mac loop + `--persist-warm-shot` in `$vkShots`. **The ONLY new shader is `persist_warm.comp.hlsl` (int64, NOT
   in `hf_gen_msl`).**
