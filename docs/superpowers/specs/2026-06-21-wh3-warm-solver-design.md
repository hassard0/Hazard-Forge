# Slice WH3 — Warm-Started Hull Contacts: THE ACCUMULATED WARM-STARTED SOLVER (the core solve) — Design

> Autonomous-session spec. Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The THIRD slice of
> FLAGSHIP #26 (WARM-STARTED HULL CONTACTS + ROBUST DETERMINISTIC STACKING, `hf::sim::warmhull`). WH1 built the
> feature ID; WH2 the persistent cache that matches this tick's manifold points to last tick's accumulated
> impulses. WH3 is the engine of the flagship: an **ACCUMULATED, warm-started** normal sequential-impulse solver
> that primes each contact point from its cached impulse and applies only the per-iteration DELTA (clamping the
> ACCUMULATED total ≥ 0). This is the lever that closes the #25 gap: the documented #25 tower collapse
> (convex.h:758-763) is caused by the NON-accumulated Gauss-Seidel re-deriving a fresh, slightly-inconsistent
> impulse each tick whose fixed-point asymmetry leaks a residual torque; the accumulated form converges to a
> consistent island equilibrium across ticks, removing the torque SOURCE instead of damping its symptom (the #25
> global angular drag). It is the direct hull-normal generalization of `persist::SolveFrictionWarm` (the box
> friction warm solver, #21). int64 → Vulkan + Metal CPU-reference (GPU==CPU memcmp), chunked 1 tick/dispatch
> (TDR-safe). APPEND to `engine/sim/warmhull.h` (WH1/WH2 + manifold/gjk/persist/convex/fric/fpx BYTE-FROZEN).
> Branch: `slice-wh3`. See [[hazard-forge-warmhull-roadmap]], [[hazard-forge-manifold-roadmap]],
> [[hazard-forge-gpu-tdr-chunking]], [[hazard-forge-metal-showcase-gate]], [[hazard-forge-docs-style]].

**Goal:** Extend `engine/sim/warmhull.h` (additive — WH1/WH2 byte-unchanged) with `SolveHullManifoldWarm(bodyA,
bodyB, invIaW, invIbW, KeyedHullManifold&, restitution, iters)` — the accumulated normal sequential-impulse solver
(prime each point from its `normalImpulse`, then `iters` Gauss-Seidel sweeps applying only the impulse DELTA and
clamping the accumulated total ≥ 0, writing the converged accumulated impulse back into the `KeyedHullManifold`) —
and `StepWarmHullWorld(world, cache, cfg)` / `StepWarmHullWorldN` — the `manifold::StepHullWorldHardened` shell
with the impulse pass replaced by: build keyed manifolds (WH2) → `MatchHullCache` (warm-seed) → `SolveHullManifoldWarm`
→ `UpdateHullCache` (persist the accumulated impulses). Add `shaders/warmhull_warm.comp.hlsl` (int64 — the GPU
mirror of `StepWarmHullWorldN`, chunked 1 tick/dispatch). Add the showcase `--wh3-warm-shot <out>` (Vulkan: GPU
step → memcmp vs CPU; + the convergence comparison) / `--wh3-warm` (Metal: CPU `StepWarmHullWorldN`). Bake the
integer golden `wh3_warm`. **NO new render RHI**; ONE new compute shader (int64 Vulkan-only).

## Design call: the accumulated form is what removes the residual-torque source

`convex::SolveManifoldImpulse` (convex.h:651) is the NON-accumulated per-point normal-impulse kernel: each call
computes a fresh impulse `jn = -(1+e)·vn / k` clamped ≥ 0, applied immediately. It "cannot be warm-started" (the
persist.h:284-290 note) because it keeps no running total. WH3's `SolveHullManifoldWarm` is the ACCUMULATED form
(the `persist::SolveFrictionWarm` idiom, persist.h:335-422, generalized from box+friction to the hull multi-point
normal manifold):
- **Prime:** seed each point's running accumulated impulse from its `KeyedHullManifold::normalImpulse[p]` (the
  WH2-matched value — last tick's converged total; 0 if cold).
- **Apply the seed as an impulse** to the bodies ONCE before the sweeps (the warm-start kick — last tick's solution
  is approximately this tick's, so the solver starts near the answer).
- **Accumulated Gauss-Seidel:** for `iters` sweeps, for each point `p` in FIXED order: compute the desired total
  `jnTotal = accum[p] + (-(1+e)·vn_p / k_p)`; CLAMP `jnTotal ≥ 0` (a contact only pushes); apply the DELTA
  `dJ = jnTotal − accum[p]` to the bodies; set `accum[p] = jnTotal`. (The effective-mass `k` + the
  contact-point velocity math are `convex::SolveManifoldImpulse`'s, convex.h:645-651, reused as the per-point
  kernel — with the full `WorldInvInertiaFull` tensor, manifold.h:771.)
- **Write back:** store the converged `accum[p]` into `KeyedHullManifold::normalImpulse[p]` so `UpdateHullCache`
  persists it for next tick. FIXED order, pure integer, deterministic.

**`StepWarmHullWorld`** is `manifold::StepHullWorldHardened` (manifold.h:809) with step (3) — the impulse solve —
replaced: instead of re-deriving a fresh `HullContactMulti` + `SolveManifoldImpulse` per pair per sweep, it builds
the keyed manifold ONCE per pair, `MatchHullCache` to warm-seed, runs `SolveHullManifoldWarm`, and `UpdateHullCache`.
Steps (1) integrate, (2) full inertia, (4) position de-pen are UNCHANGED from the hardened step. The `cfg.angDamp`
stays available but the HEADLINE is that the warm solve holds the stack with `angDamp = kOne` (OFF) — the damping
crutch is no longer needed because the residual-torque source is gone.

> NOTE: `StepWarmHullWorld` runs the GJK/EPA/clip narrowphase + the accumulated solve = HEAVIER than the hardened
> step. The GPU dispatch is chunked **1 tick/dispatch** (the documented Windows ~2s TDR rule) → TDR impossible by
> construction; verify ~3 runs GPU==CPU (NOT 30x). The cache is per-tick mutable replayable state (carried in the
> world/cache, snapshotted in WH5).

## Reuse map (file:line)
- **WH1/WH2 `engine/sim/warmhull.h` (APPEND after WH2):** `KeyedHullManifold`, `BuildKeyedHullManifold`,
  `HullCache`, `MatchHullCache`, `UpdateHullCache`. WH1/WH2 frozen.
- **persist.h (read-only — the accumulated-form TEMPLATE):** `persist::SolveFrictionWarm` (persist.h:335-422 — the
  prime/seed-kick/accumulated-delta/clamp/write-back idiom to mirror for the hull normal manifold), the
  "cannot be warm-started" rationale (persist.h:284-290).
- **convex.h (read-only):** `convex::SolveManifoldImpulse` (convex.h:651 — the effective-mass `k` + contact-point
  velocity per-point kernel math, the NON-accumulated reference WH3's accumulated form is built from),
  `convex::ContactManifold`, `convex::ConvexStepConfig`, `convex::FxMat3`.
- **manifold.h (read-only):** `manifold::StepHullWorldHardened` (manifold.h:809 — the 5-pass shell WH3 extends),
  `manifold::HullContactMulti`, `manifold::FxHullInertiaBodyFull`/`WorldInvInertiaFull` (manifold.h:674/771 — the
  full inertia), `manifold::MeasureHullStack`/the residual metric.
- **The int64 GPU step precedent:** `shaders/hull_step_hardened.comp.hlsl` (MF4 — the int64 Vulkan-only chunked
  1-tick/dispatch step mirror) + `warmhull_cache.comp` (WH2). Mirror for `warmhull_warm.comp`.
- **Registration:** `samples/hello_triangle/CMakeLists.txt` (`shaders/warmhull_warm.comp.hlsl:cs` — Vulkan only,
  int64 → NOT in `hf_gen_msl`), `scripts/verify.ps1` (`wh3_warm` + `--wh3-warm-shot`), `engine/editor/introspect.cpp`
  + `tests/introspect_test.cpp` (**controller rebakes**), append to `tests/warmhull_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/sim/warmhull.h`** (WH1/WH2 byte-frozen): `SolveHullManifoldWarm`, `StepWarmHullWorld`,
   `StepWarmHullWorldN`, a `WarmHullMeasure`/residual helper. **ONE new shader `shaders/warmhull_warm.comp.hlsl`**
   (int64, Vulkan-only — NOT in `hf_gen_msl`; Metal CPU-ref). **NO new render RHI.** manifold.h/persist.h/ALL
   other sim headers + ALL OTHER shaders BYTE-UNCHANGED.
2. **Showcase `--wh3-warm-shot <out>` (Vulkan) AND `--wh3-warm` (Metal) — WIRE BOTH (grep your own
   `visual_test.mm` for `--wh3-warm` BEFORE reporting DONE).** Build a deterministic stack scene (the MF4
   box-on-support, or a 2-3 hull stack), step it with `StepWarmHullWorldN` (Vulkan: GPU `warmhull_warm.comp`
   chunked 1-tick/dispatch, memcmp vs CPU). Render the settled world (pure-integer side-view → strict-zero
   cross-vendor). Golden = `tests/golden/metal/wh3_warm.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact stdout lines):**
   - **(1) the convergence headline:** `wh3-warm: warm residual < cold residual {warm:<W>, cold:<C>} at iters:<I>`
     — at the SAME low solve-iteration budget, the WARM solver's post-step residual (penetration / constraint
     violation, the `MeasureHullStack` metric) is strictly LESS than the non-accumulated cold solver's. Assert
     `warm < cold`. **This is the lever the whole flagship rests on — warm-start converges faster.**
   - **(2) GPU==CPU:** `wh3-warm: {bodies:<N>, ticks:<K>} GPU==CPU BIT-EXACT` — the GPU `StepWarmHullWorldN` final
     world+cache memcmp-equals the CPU reference (Vulkan).
   - **(3) damping-off hold:** `wh3-warm: {angDampOff:true, settled:true}` — with `cfg.angDamp = kOne` (OFF), the
     warm-stepped stack settles (residual angular speed in band) where the frozen hardened step with damping OFF
     does NOT (the removed-torque-source proof).
   - **(4) determinism:** `wh3-warm determinism: two runs BYTE-IDENTICAL` (~3 runs — efficient, TDR-safe).
   - Golden discipline: ONLY `tests/golden/metal/wh3_warm.png`; do NOT commit it. Existing 229 goldens UNTOUCHED.
4. **Cross-backend bar (int64 → strict on the integer world):** Vulkan GPU==CPU bit-exact (~3× clean, chunked
   1-tick/dispatch TDR-safe); Metal CPU-ref byte-identical. The settled world is strict-zero cross-vendor; the
   pure-integer render strict-zero.
5. **Tests — APPEND to `tests/warmhull_test.cpp` (CPU):** `SolveHullManifoldWarm` accumulated total stays ≥ 0 and
   converges (residual decreases vs the non-accumulated solve at equal iters); a warm-seeded solve reaches a lower
   residual than a cold one; `StepWarmHullWorldN` is deterministic (two runs byte-equal); the cache carries the
   converged impulse across ticks (a settled stack's impulses stabilize). Clean under `windows-msvc-asan`.
6. **Introspect.** Add EXACTLY `warmhull-solve` (features) + `--wh3-warm-shot` (showcases) + update
   `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the controller does.**

## RHI seam additions (summary)
- **ONE new compute shader, NO new render RHI.** `shaders/warmhull_warm.comp.hlsl` (int64, Vulkan-only — NOT in
  `hf_gen_msl`; Metal CPU-ref), chunked 1 tick/dispatch. Reuses the existing compute-dispatch + SSBO seam.
  `engine/sim/warmhull.h` APPEND-only (WH1/WH2 frozen); manifold.h/persist.h/convex.h + ALL other sim headers +
  ALL OTHER shaders UNCHANGED. Report the seam: warmhull.h APPEND-only + ONE new Vulkan-only shader; NO rhi.h
  change, NO MSL entry, NO frozen-file edit.

## Out of scope (YAGNI — later slices)
Sleeping islands + the falsifiable stable-stack DELTA (WH4 — WH3 proves warm CONVERGES better; WH4 adds sleep +
the height-N stability inequality). Lockstep (WH5 — WH3's cache is replayable state WH5 snapshots). Render capstone
(WH6). Hull friction / tangential warm-start (a separate future flagship — WH3 warm-starts the NORMAL impulse
only). WH3 claims ONLY: a deterministic accumulated warm-started normal solver that converges to a lower residual
than the non-accumulated solver at equal iters and holds a stack with global damping OFF, bit-identical
CPU/Vulkan/Metal, with the integer golden + the four proofs. CAVEAT: "converges better / holds" is a within-band
residual improvement, not analytic zero; the tall-stack stability DELTA is WH4's headline (WH3 is the engine, WH4
the demonstration).

## Verification gate (controller)
1. `ctest --preset windows-msvc-debug -R "warmhull|introspect"` green. Clean under `windows-msvc-asan` (SEPARATE
   build + test).
2. **proofs + visual:** `--wh3-warm-shot` on Vulkan: the 4 proof lines (incl. warm<cold + GPU==CPU) + exit 0 under
   the conan validation layer → ZERO VUID. **~3 runs all GPU==CPU (chunked 1-tick/dispatch — TDR-safe). VERIFY the
   settled stack render is coherent.**
3. Metal: `visual_test --wh3-warm` → `tests/golden/metal/wh3_warm.png`; two runs DIFF 0.0000. **Confirm `--wh3-warm`
   wired in `visual_test.mm` (grep it) BEFORE the Mac bake; confirm NO `hf_gen_msl` entry (int64).** Cross-vendor
   STRICT ZERO on the integer world + pure-integer render.
4. **Render-invariance:** ONLY `wh3_warm.png` added; the other 229 byte-identical (+ controller introspect rebake).
5. Introspect: exactly `+warmhull-solve` + `--wh3-warm-shot`; `tests/introspect_test.cpp` updated.
6. Seam grep clean (`rhi.h` + WH1/WH2 warmhull.h code + manifold.h/persist.h/convex.h + ALL other sim headers +
   ALL OTHER shaders byte-unchanged; warmhull.h APPEND-only; exactly ONE new shader `warmhull_warm.comp.hlsl`,
   Vulkan-only). `wh3_warm` in the Mac loop + `--wh3-warm-shot` in `$vkShots`.
