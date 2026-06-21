# Slice WH4 — Warm-Started Hull Contacts: SLEEPING ISLANDS → THE STABLE STACK (the new-physics beat) — Design

> Autonomous-session spec. Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FOURTH slice (THE
> HEADLINE BEAT) of FLAGSHIP #26 (WARM-STARTED HULL CONTACTS + ROBUST DETERMINISTIC STACKING, `hf::sim::warmhull`).
> WH1-WH3 built the feature ID, the persistent cache, and the accumulated warm solver (which converges better than
> the non-accumulated one and holds a single box with the angular damping OFF). WH4 is the PAYOFF the whole
> flagship was built for: **a deterministic, bit-identical, lockstep-ready STABLE STACK of convex polyhedra** —
> the documented #25 tall-stack gap, CLOSED. It adds sleeping islands (the persist.h #21 recipe) on top of the
> warm solver: a settled island of hulls freezes at EXACTLY zero residual velocity before the within-band residual
> torque can re-accumulate and topple it, and a wake-impulse re-energizes the whole island atomically. THE
> HEADLINE is a FALSIFIABLE STABILITY DELTA: at a fixed tick budget, the warm+sleep step holds an N-high tower the
> frozen `StepHullWorldHardened` (#25) topples at the same budget. **Honest scope (rigorous): NOT an unconditional
> "stable arbitrarily-tall stack" — the claim is (a) a measured, reproducible, golden-verified stability INEQUALITY
> vs #25, and (b) provable rest to a DEMONSTRATED N (a specific tower that goes fully asleep and stays asleep). A
> mainstream float engine reaches resting stacks only non-deterministically; this one is byte-identical across
> CPU/Vulkan/Metal — the differentiator independent of the exact N.** int64 → Vulkan + Metal CPU-reference, chunked
> 1 tick/dispatch. APPEND to `engine/sim/warmhull.h` (WH1-WH3 + manifold/persist/etc BYTE-FROZEN). Branch:
> `slice-wh4`. See [[hazard-forge-warmhull-roadmap]], [[hazard-forge-manifold-roadmap]],
> [[hazard-forge-gpu-tdr-chunking]], [[hazard-forge-metal-showcase-gate]], [[hazard-forge-docs-style]].

**Goal:** Extend `engine/sim/warmhull.h` (additive — WH1-WH3 byte-unchanged) with `HullSleepState` /
`HullSleepConfig`, `KineticEnergyHull` (the per-body deterministic energy metric), `PropagateWakeHull` (the contact
adjacency from `manifold::HullContactMulti` overlaps → islands), `StepWarmSleepHullWorld(world, cache, sleep, cfg)`
/ `StepWarmSleepHullWorldN` (the WH3 warm step + the island sleep/freeze/skip logic), and `HullSleepMeasure` /
`MeasureHullSleep`. Add `shaders/warmhull_sleep.comp.hlsl` (int64 — the GPU mirror, chunked 1 tick/dispatch). Add
the showcase `--wh4-stack-shot <out>` (Vulkan: GPU step → memcmp vs CPU; + the stability-delta comparison) /
`--wh4-stack` (Metal: CPU `StepWarmSleepHullWorldN`). Bake the integer golden `wh4_stack`. **NO new render RHI**;
ONE new compute shader (int64 Vulkan-only).

## Design call: sleep freezes the island at exactly zero before the residual torque can topple it

The WH3 warm solver REDUCES the residual torque but does not provably ELIMINATE it in fixed point; on a TALL tower
the tiny per-tick residual still integrates over many ticks. The persist.h #21 cure (for boxes) is SLEEP: once an
island's kinetic energy stays below a threshold for a hysteresis window, FREEZE it — zero its velocities EXACTLY
and SKIP its solve — so the residual torque has nothing to integrate. WH4 generalizes that recipe to hull islands:
- **`KineticEnergyHull(body)`** — the deterministic per-body energy (the `persist::KineticEnergy` idiom): a
  fixed-point combination of linear + angular speed. Pure integer.
- **`HullSleepState`** — per-body `{ asleep, lowEnergyTicks }` (the persist.h sleep-state shape). A body whose
  energy stays below `cfg.sleepThreshold` for `cfg.sleepTicks` consecutive ticks is a sleep CANDIDATE.
- **`PropagateWakeHull(world, sleep)`** — build the contact-adjacency graph from `manifold::HullContactMulti`
  overlaps (two bodies in contact are in the same island), then an island sleeps ONLY if EVERY member is a
  candidate (a whole island sleeps or none of it does — the persist.h island rule); a wake event (an external
  impulse, or a new contact from a moving body) wakes the ENTIRE island atomically (`PropagateWake`, persist.h:
  ~659-720). FIXED body/island order.
- **Sleep FREEZE (persist.h:720-729 idiom):** an asleep body's `vel`/`angVel` are set to EXACTLY zero and its
  solve + integrate are SKIPPED (it does not move, accumulates no residual). On wake, it rejoins the warm solve.
- **`StepWarmSleepHullWorld`** = the WH3 `StepWarmHullWorld` with: (a) asleep bodies skipped in integrate + solve;
  (b) after the solve, update `KineticEnergyHull` + `lowEnergyTicks`; (c) `PropagateWakeHull` to sleep converged
  islands / wake disturbed ones. The cache + sleep state are the per-tick mutable replayable state.

**THE HEADLINE — the stable stack + the falsifiable delta:** build a deterministic **N-high tower** (N ≥ 3 — a
genuine multi-body stack, NOT the WH3 single box; e.g. 3-5 stacked boxes/mixed hulls on a static support). Step it
with `StepWarmSleepHullWorldN`: the tower settles, the island goes FULLY ASLEEP (`asleepCount == dynamicCount`),
and the awake-body `maxSpeed` is EXACTLY 0 (frozen). Step the IDENTICAL tower with the frozen
`manifold::StepHullWorldHardenedN` (#25, damping off) at the SAME tick budget: it does NOT hold (the residual
torque topples it — `maxAngVel`/the tower height drifts out of band). The two final states DIFFER, and the
warm+sleep one passes a "tower-standing" predicate the frozen one fails. **This is the closed-#25-gap proof: a
deterministic stable stack, by a measured and bit-identical margin.**

> NOTE: `StepWarmSleepHullWorld` over a multi-body tower (GJK/EPA/clip per pair + the warm solve) is HEAVIER than
> WH3 → the GPU dispatch is chunked **1 tick/dispatch** (the Windows ~2s TDR rule). Verify ~3 runs GPU==CPU (NOT
> 30x). The sleep state + cache are snapshotted in WH5 (the replayable triple).

## Reuse map (file:line)
- **WH1-WH3 `engine/sim/warmhull.h` (APPEND after WH3):** `StepWarmHullWorld`/`SolveHullManifoldWarm`,
  `BuildKeyedHullManifold`, `HullCache`/`MatchHullCache`/`UpdateHullCache`. WH1-WH3 frozen.
- **persist.h (read-only — the sleeping-island TEMPLATE):** `persist::SleepState`/`SleepConfig`,
  `persist::KineticEnergy`, `persist::PropagateWake`/`StepWarmSleepWorld` (persist.h:~659-829 — the island +
  freeze + skip recipe to mirror for hulls), the sleep-freeze velocity-zeroing (persist.h:720-729). persist's
  version is box-only — WH4's hull version is new (adjacency from `HullContactMulti`).
- **manifold.h (read-only):** `manifold::StepHullWorldHardened`/`…N` (manifold.h:809 — the frozen #25 step, the
  TEETER CONTROL the delta proof beats), `manifold::HullContactMulti` (the contact adjacency), `MeasureHullStack`.
- **The int64 GPU step precedent:** `shaders/warmhull_warm.comp.hlsl` (WH3) + `hull_step_hardened.comp.hlsl` (MF4)
  — mirror for `warmhull_sleep.comp` (int64 Vulkan-only, chunked 1 tick/dispatch).
- **Registration:** `samples/hello_triangle/CMakeLists.txt` (`shaders/warmhull_sleep.comp.hlsl:cs` — Vulkan only),
  `scripts/verify.ps1` (`wh4_stack` + `--wh4-stack-shot`), `engine/editor/introspect.cpp` +
  `tests/introspect_test.cpp` (**controller rebakes**), append to `tests/warmhull_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/sim/warmhull.h`** (WH1-WH3 byte-frozen): `HullSleepState`, `HullSleepConfig`,
   `KineticEnergyHull`, `PropagateWakeHull`, `StepWarmSleepHullWorld`, `StepWarmSleepHullWorldN`, `HullSleepMeasure`,
   `MeasureHullSleep`. **ONE new shader `shaders/warmhull_sleep.comp.hlsl`** (int64, Vulkan-only — NOT in
   `hf_gen_msl`; Metal CPU-ref). **NO new render RHI.** manifold.h/persist.h/ALL other sim headers + ALL OTHER
   shaders BYTE-UNCHANGED.
2. **Showcase `--wh4-stack-shot <out>` (Vulkan) AND `--wh4-stack` (Metal) — WIRE BOTH (grep your own
   `visual_test.mm` for `--wh4-stack` BEFORE reporting DONE).** Build a deterministic **N≥3 tower** scene; step
   with `StepWarmSleepHullWorldN` (Vulkan: GPU `warmhull_sleep.comp` chunked 1-tick/dispatch, memcmp vs CPU) AND
   the frozen `manifold::StepHullWorldHardenedN` control. Render the SETTLED ASLEEP tower (pure-integer side-view →
   strict-zero cross-vendor). Golden = `tests/golden/metal/wh4_stack.png` (Mac-baked by the CONTROLLER — DO NOT
   commit).
3. **PROOFS (fail loudly; exact stdout lines):**
   - **(1) THE STABLE STACK:** `wh4-stack: {tower:<N>, asleep:<A>, awakeMaxSpeed:<S>} fully-asleep` — the N≥3
     tower goes FULLY asleep (`asleep == dynamicCount`) and the awake-body `maxSpeed` is EXACTLY 0 (the freeze).
     Assert `asleep == dynamicCount` AND the tower is still standing (each body within its rest band of its
     stacked pose).
   - **(2) THE FALSIFIABLE DELTA (the closed-#25-gap headline):** `wh4-stack: {warmSleepHolds:true,
     frozenTopples:true} at N:<N>, ticks:<K>` — at the SAME budget, warm+sleep HOLDS the N-tower (standing,
     asleep) while the frozen `StepHullWorldHardenedN` does NOT (toppled / out of rest band). Assert BOTH. **This
     is the deterministic stable stack the single-point/non-accumulated #25 step cannot hold.**
   - **(3) wake:** `wh4-stack wake: island re-energized atomically (awoke:<W>)` — a wake-impulse on one body wakes
     the WHOLE island (all members `asleep=false`), and the disturbed tower re-solves.
   - **(4) GPU==CPU + determinism:** `wh4-stack: GPU==CPU BIT-EXACT {bodies:<N>, ticks:<K>}` + `wh4-stack
     determinism: two runs BYTE-IDENTICAL` (~3 runs).
   - Golden discipline: ONLY `tests/golden/metal/wh4_stack.png`; do NOT commit it. Existing 230 goldens UNTOUCHED.
4. **Cross-backend bar (int64 → strict):** Vulkan GPU==CPU bit-exact (~3× clean, chunked 1-tick/dispatch); Metal
   CPU-ref byte-identical. The settled asleep tower is strict-zero cross-vendor; the pure-integer render strict-zero.
5. **Tests — APPEND to `tests/warmhull_test.cpp` (CPU):** the N≥3 tower goes fully asleep under
   `StepWarmSleepHullWorldN` (asleep==dynamicCount, awake maxSpeed==0); the IDENTICAL tower under frozen
   `StepHullWorldHardenedN` does NOT stay in the rest band (the delta — assert the inequality, do NOT fake it); a
   wake-impulse wakes the whole island; `KineticEnergyHull` monotonically gates sleep; two-run byte-equal;
   determinism across K. Clean under `windows-msvc-asan`.
6. **Introspect.** Add EXACTLY `warmhull-sleep` (features) + `--wh4-stack-shot` (showcases) + update
   `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the controller does.**

## RHI seam additions (summary)
- **ONE new compute shader, NO new render RHI.** `shaders/warmhull_sleep.comp.hlsl` (int64, Vulkan-only — NOT in
  `hf_gen_msl`; Metal CPU-ref), chunked 1 tick/dispatch. Reuses the existing compute-dispatch + SSBO seam.
  `engine/sim/warmhull.h` APPEND-only (WH1-WH3 frozen); manifold.h/persist.h + ALL other sim headers + ALL OTHER
  shaders UNCHANGED. Report the seam: warmhull.h APPEND-only + ONE new Vulkan-only shader; NO rhi.h change, NO MSL
  entry, NO frozen-file edit.

## Out of scope (YAGNI — honest scope discipline)
**The stability claim is a measured DELTA + rest to a DEMONSTRATED N, NOT unconditional unbounded stability** — the
within-band Gauss-Seidel + linear-only de-pen cannot give a provably stable arbitrarily-tall stack (sleep freezes a
SETTLED island; a tower too tall to settle within the budget before the residual unseats it is a documented limit,
not a target). Lockstep (WH5 — WH4's sleep state + cache are the replayable triple WH5 snapshots). Render capstone
(WH6). Hull friction (a separate future flagship). A continuous (non-hysteresis) sleep model. WH4 claims ONLY: a
deterministic, bit-identical (CPU↔Vulkan↔Metal) warm+sleep step under which a demonstrated N≥3 tower goes fully
asleep and stands, where the frozen #25 step topples it at the same budget — with the integer golden + the four
proofs. CAVEAT: "asleep/standing" = within a fixed rest band (the freeze makes awake-speed exactly 0, but the
settle that PRECEDES sleep is within-band); sleep is a hysteresis heuristic (defensible — real solvers sleep
resting bodies — and fully deterministic).

## Verification gate (controller)
1. `ctest --preset windows-msvc-debug -R "warmhull|introspect"` green. Clean under `windows-msvc-asan` (SEPARATE
   build + test).
2. **proofs + visual:** `--wh4-stack-shot` on Vulkan: the 4 proof lines (incl. the stable-stack + the
   warmSleepHolds/frozenTopples DELTA + GPU==CPU) + exit 0 under the conan validation layer → ZERO VUID. **~3 runs
   all GPU==CPU (chunked 1-tick/dispatch — TDR-safe). VERIFY the image shows the N≥3 tower STANDING (asleep,
   settled), not toppled.**
3. Metal: `visual_test --wh4-stack` → `tests/golden/metal/wh4_stack.png`; two runs DIFF 0.0000. **Confirm
   `--wh4-stack` wired in `visual_test.mm` (grep it) BEFORE the Mac bake; confirm NO `hf_gen_msl` entry (int64).**
   Cross-vendor STRICT ZERO on the integer tower + pure-integer render.
4. **Render-invariance:** ONLY `wh4_stack.png` added; the other 230 byte-identical (+ controller introspect rebake).
5. Introspect: exactly `+warmhull-sleep` + `--wh4-stack-shot`; `tests/introspect_test.cpp` updated.
6. Seam grep clean (`rhi.h` + WH1-WH3 warmhull.h code + manifold.h/persist.h + ALL other sim headers + ALL OTHER
   shaders byte-unchanged; warmhull.h APPEND-only; exactly ONE new shader `warmhull_sleep.comp.hlsl`, Vulkan-only).
   `wh4_stack` in the Mac loop + `--wh4-stack-shot` in `$vkShots`. **This is the new-physics headline of flagship
   #26 — the deterministic stable stack; do NOT accept a fake/2-body demo, demand the real N≥3 tower delta.**
