# Slice CD3 — Deterministic Integer CCD: THE SUBSTEPPED CCD WORLD STEP (the mechanism beat) — Design

> Autonomous-session spec. Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The THIRD slice of
> FLAGSHIP #24 (DETERMINISTIC INTEGER CCD, `hf::sim::ccd`). CD1 built the time-of-impact (`ConservativeAdvance`);
> CD2 built the swept-AABB broadphase (`BuildSweptBroadphasePairs`, a proven superset of the discrete set). CD3
> assembles them into a **substepped CCD world step**: each tick, sweep-broadphase the candidate pairs, compute
> each pair's TOI, advance ALL bodies to the EARLIEST impact, resolve that contact via the frozen solver, then
> consume the remaining dt — so a fast body STOPS at the first thing it would hit instead of tunneling through it
> in one big tick. The make-or-break is the **NO-TUNNEL proof**: in a scene with a fast mover, `StepHullWorldCCD`
> lands the body on the correct side of the wall, while the discrete `broad::StepHullWorldBP` on the SAME scene
> tunnels it through — the two final states DIFFER, and the CCD one passes a `noTunnel` predicate. APPEND to
> `engine/sim/ccd.h` (CD1+CD2 + broad.h/gjk.h/convex.h/fpx.h/fric.h/persist.h/grain.h BYTE-FROZEN). Branch:
> `slice-cd3`. See [[hazard-forge-ccd-roadmap]], [[hazard-forge-gpu-tdr-chunking]], [[hazard-forge-docs-style]].

**Goal:** Extend `engine/sim/ccd.h` (additive — CD1+CD2 byte-unchanged) with `CcdStepConfig` (a
`broad::BroadStepConfig` + `maxSubsteps`) + `StepHullWorldCCD` / `StepHullWorldCCDN` (the substepped tick) +
`CcdMeasure` / `MeasureCcd`. Add the int64 GPU shader `shaders/ccd_step.comp.hlsl` (Vulkan-only, **chunked at 1
tick/dispatch from the start** — the heaviest single-thread dispatch in the suite) + the showcase
`--ccd-step-shot` (Vulkan) / `--ccd-step` (Metal). Bake the integer golden `ccd_step`.

## Design call: substep to the earliest TOI, resolve, repeat — bounded, fixed-order, integer

`StepHullWorldCCD(world, cfg)` — one tick, with a bounded substep budget `cfg.maxSubsteps`:
1. **Swept broadphase:** `BuildSweptBroadphasePairs(world, remainingDt, cfg.cellSize)` (CD2) → the candidate pair
   list, (i,j)-sorted (the BP3 Gauss-Seidel-order discipline).
2. **Per-pair TOI:** for each candidate pair, `ConservativeAdvance(hullA, bodyA, hullB, bodyB, remainingDt)` (CD1)
   → its `FxToi`. Track the GLOBAL-MINIMUM TOI over all `hit` pairs (a deterministic min over the fixed-order pair
   list; ties keep the lowest pair index).
3. **Advance to the earliest impact:** integrate ALL dynamic bodies forward by `minToi` (`fpx::IntegrateBodyFull`,
   the tick's gravity, sub-dt = `minToi`). If NO pair hit within `remainingDt` → integrate all bodies by the FULL
   `remainingDt` and the tick is done (no impact this tick).
4. **Resolve at the contact instant:** at `minToi` the impacting pair(s) are touching — run the frozen contact
   resolve (`gjk::HullContact` → `convex::SolveManifoldImpulse` + the position de-pen, the BP4/`StepHullWorldBP`
   solve passes) over the touching pairs so the impulse stops/bounces the body. Reuse the `StepHullWorldBP` solve
   machinery for this (document the exact reuse — it is the same solver, applied at the TOI pose).
5. **Consume remaining dt + repeat:** `remainingDt -= minToi`; loop from (1) until `remainingDt ≤ 0` OR
   `maxSubsteps` is hit (the bounded budget — a body past the cap can still tunnel, a documented deterministic
   limit). All integer, FIXED substep + pair + body order → bit-identical CPU/Vulkan/Metal.

`StepHullWorldCCDN(world, cfg, ticks)` runs `ticks` CCD ticks. `CcdStepConfig` = `broad::BroadStepConfig` (the
gravity/dt/iters/slop/cellSize) + `uint32_t maxSubsteps`.

> **TDR (apply up front — the heaviest dispatch in the suite, [[hazard-forge-gpu-tdr-chunking]]):** a CCD tick =
> swept broadphase + N TOI loops (each an iterated `Gjk`) + a resolve, ×`maxSubsteps` — far heavier than BP4's
> discrete tick. **Chunk the GPU `ccd_step.comp` at 1 TICK/DISPATCH from the start** (the BP4 precedent), barrier
> between, bit-identical to one big dispatch. Keep the scene MODERATE (e.g. ~8-16 bodies incl. the fast mover, a
> small `maxSubsteps` like 8) so one tick is provably ≪ the ~2s watchdog. **Verify GPU determinism with a TIMING
> argument (one tick-dispatch time ≪ watchdog) + ~3 runs, NOT a 30× loop.** If one tick is still too heavy, go
> finer to 1 substep/dispatch OR shrink the scene/`maxSubsteps` — report it.

## Reuse map (file:line)
- **CD1+CD2 `engine/sim/ccd.h` (APPEND after CD2's `MeasureSweptPairs`):** `ConservativeAdvance` (the TOI),
  `BuildSweptBroadphasePairs` (the swept candidates), `FxToi`, `kContactEps`. CD1+CD2 frozen.
- **broad.h (read-only — REUSE verbatim):** `broad::StepHullWorldBP` (broad.h:736 — READ IT; CD3 reuses its solve
  + de-pen passes for the at-TOI resolve, and its (i,j)-sort), `broad::BroadStepConfig`, `gjk::HullWorld`/
  `HullContact`, `convex::SolveManifoldImpulse`, `gjk::MeasureHullStack`. `fpx::IntegrateBodyFull` (the capped
  sub-dt advance). Do NOT modify broad.h/gjk.h/fpx.h/etc — BYTE-FROZEN.
- **Proof-tier:** int64 → `ccd_step.comp` VULKAN-ONLY (NOT in `hf_gen_msl`); Metal `--ccd-step` runs CPU
  `StepHullWorldCCDN`. The `broad_hull_step.comp` chunked-dispatch precedent.
- **The showcase precedent:** BP4's `--broad-hull-shot` (the chunked GPU step + GPU==CPU memcmp + the settled
  side-view render) + BP3's `--broad-convex-shot` (the comparison-to-a-baseline proof). Mirror for `--ccd-step`.
- **Registration:** `scripts/verify.ps1` (append `ccd_step` + `--ccd-step-shot` to `$vkShots`),
  `metal_headless/CMakeLists.txt` (`hf_gen_msl` — do NOT add `ccd_step.comp`), `engine/editor/introspect.cpp` +
  `tests/introspect_test.cpp` (**controller rebakes the JSON golden**), append to `tests/ccd_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/sim/ccd.h`** (CD1+CD2 byte-frozen): `CcdStepConfig`, `StepHullWorldCCD`,
   `StepHullWorldCCDN`, `CcdMeasure`, `MeasureCcd`. Pure integer, FIXED order, bounded `maxSubsteps`. NO new RHI;
   one new Vulkan-only shader (chunked at 1 tick/dispatch).
2. **New shader `shaders/ccd_step.comp.hlsl` (int64, VULKAN-ONLY)** — runs `StepHullWorldCCDN` verbatim, single
   thread, **chunked at 1 tick/dispatch** with a barrier between. NOT in `hf_gen_msl`.
3. **Showcase `--ccd-step-shot <out>` (Vulkan) AND `--ccd-step` (Metal) — WIRE BOTH (grep your own visual_test.mm
   for `--ccd-step` before reporting DONE).** A MODERATE scene with a FAST MOVER aimed at a thin static wall (its
   per-tick travel ≫ the wall thickness) plus a few slow settling hulls. Vulkan dispatches `ccd_step.comp`
   (chunked, 1 tick/dispatch) + memcmps GPU final world vs CPU `StepHullWorldCCDN`; Metal runs the CPU path. BOTH
   render the final world (a side-view — the fast mover arrested at the wall). Golden =
   `tests/golden/metal/ccd_step.png` (Mac-baked by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU:** `ccd-step: {bodies:<N>, ticks:<K>} GPU==CPU BIT-EXACT` — the GPU final bodies == the CPU
     `StepHullWorldCCDN` byte-for-byte; assert. (Run the shot ~3× in the gate; 1 tick/dispatch is TDR-safe by
     construction.)
   - **(2) determinism:** `ccd-step determinism: two runs BYTE-IDENTICAL`.
   - **(3) THE NO-TUNNEL PROOF:** `ccd-step no-tunnel: {ccdSide:correct, discreteTunneled:true}` — run the SAME
     fast-mover scene through `StepHullWorldCCD` AND the discrete `broad::StepHullWorldBP`; the CCD final has the
     fast body on the CORRECT (near) side of the wall (a `noTunnel` predicate: the body's final position is on the
     approach side, gap ≥ 0), while the discrete final has it TUNNELED through (the far side); assert the two
     final states DIFFER and the CCD one passes `noTunnel`. **This is the headline: CCD provably prevents the
     tunnel the discrete solver allows.**
   - **Golden discipline: ONLY `tests/golden/metal/ccd_step.png`; do NOT commit it.** Existing 217 goldens
     UNTOUCHED.
5. **Cross-backend bar (int64 → strict):** Vulkan GPU==CPU bit-exact (~3× clean, no TDR); Metal CPU-ref
   byte-identical; cross-vendor ZERO differing pixels.
6. **Tests — APPEND to `tests/ccd_test.cpp`:** `StepHullWorldCCDN` brings a moderate scene to a settled state;
   the no-tunnel case (`StepHullWorldCCD` keeps the fast body on the near side; discrete `StepHullWorldBP`
   tunnels it — the two final states differ); deterministic (two runs byte-equal); a slow scene with no fast
   movers steps identically to the discrete `StepHullWorldBP` (CCD reduces to the discrete step when no substep
   triggers — the within-tick-TOI ≥ dt fallback). Clean under `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-ccd-step` (features) + `--ccd-step-shot` (showcases) + update
   `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the controller does.**

## RHI seam additions (summary)
- **None.** The shot rides the existing compute-dispatch + SSBO path (the `broad_hull_step`/`gjk_settle` seam,
  incl. the chunked-dispatch barrier). `engine/sim/ccd.h` APPEND-only (CD1+CD2 frozen); broad.h/gjk.h/convex.h/
  fpx.h/etc + ALL other sim headers + ALL existing shaders UNCHANGED. NEW file: `shaders/ccd_step.comp.hlsl` only.

## Out of scope (YAGNI — later slices)
The bullet-wall headline beat (CD4 — a dedicated bullet-through-thin-wall scene + the discrete control), lockstep
(CD5), lit render (CD6). CD3 claims ONLY: a deterministic, bit-exact (CPU↔Vulkan↔Metal) substepped CCD world step
that PROVABLY prevents tunneling a fast mover the discrete step allows, with the integer golden + the three
proofs. CAVEATS: conservative advancement within-band (the CD1 TOI band); `maxSubsteps` cap (a very-fast body
past the budget can still tunnel — a documented deterministic limit); the swept broadphase inflates the candidate
set; inherited GJ within-band (single-point manifold, diagonal inertia).

## Verification gate
1. `ctest --preset windows-msvc-debug -R "ccd|broad|introspect"` green. Clean under `windows-msvc-asan` (separate
   build + test).
2. **proofs + visual:** `--ccd-step-shot` on Vulkan: the 3 proofs (incl. no-tunnel) + exit 0 under the conan
   validation layer → ZERO VUID. **~3 runs all GPU==CPU (1 tick/dispatch TDR-safe by construction — NO 30× loop).**
   VERIFY the image shows the fast mover ARRESTED at the wall (not through it).
3. Metal: `visual_test --ccd-step` → `tests/golden/metal/ccd_step.png`; two runs DIFF 0.0000. Confirm
   `visual_test.mm` in the diff; `ccd_step.comp` NOT in `hf_gen_msl`. Cross-vendor STRICT ZERO.
4. **Render-invariance:** ONLY `ccd_step.png` added; the other 217 byte-identical (+ controller introspect rebake).
5. Introspect: exactly `+deterministic-ccd-step` + `--ccd-step-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` + CD1+CD2 ccd.h code + broad.h/gjk.h/convex.h/fpx.h/fric.h/persist.h/grain.h + ALL
   other sim headers + ALL existing shaders byte-unchanged; ccd.h APPEND-only; one new Vulkan-only shader, no RHI
   change). `ccd_step.comp` NOT in `hf_gen_msl`.
