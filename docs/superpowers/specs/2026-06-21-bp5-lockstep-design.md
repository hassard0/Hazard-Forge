# Slice BP5 — Deterministic Integer Broadphase: LOCKSTEP + ROLLBACK (the netcode beat) — Design

> Autonomous-session spec. Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIFTH slice of
> FLAGSHIP #23 (DETERMINISTIC INTEGER BROADPHASE, `hf::sim::broad`). BP4 made the HULL world step
> broadphase-driven and proven byte-identical to all-pairs. BP5 proves the whole broadphase-driven sim is
> **lockstep- and rollback-replayable**: two peers fed only an input-command stream re-derive the entire world —
> INCLUDING the per-tick re-broadphase — byte-identical, and a rollback re-sims from a snapshot bit-for-bit. THE
> BP5 HEADLINE: the broadphase is re-derived each tick from the current positions, so two peers re-broadphase to
> the SAME candidate-pair list → the lockstep holds **through the broadphase**, not just the narrowphase — the
> first pipeline stage that is itself replayable, not merely the contacts downstream of it. Because
> `StepHullWorldBP` is a pure deterministic integer function, this falls out by retargeting the frozen GJ5
> lockstep harness over it — PURE CPU, no shader, no RHI → both backends run the IDENTICAL harness → the golden is
> bit-identical BY CONSTRUCTION (cross-vendor 0 px). APPEND to `engine/sim/broad.h` (BP1-BP4 + gjk.h/convex.h/
> fric.h/persist.h/fpx.h/grain.h BYTE-FROZEN). Branch: `slice-bp5`. See [[hazard-forge-broad-roadmap]],
> [[hazard-forge-docs-style]].

**Goal:** Extend `engine/sim/broad.h` (additive — BP1-BP4 byte-unchanged) with `SimBroadTick` (`gjk::ApplyHullCommands`
+ `StepHullWorldBP`) + `RunBroadLockstep` + `RunBroadRollback`, reusing the frozen GJ5 command/snapshot/equality
machinery (`convex::ConvexCommand`, `gjk::HullSnapshot`/`SnapshotHull`/`RestoreHull`/`HullBodiesEqual`). Add the
showcase `--broad-lockstep-shot` (Vulkan) / `--broad-lockstep` (Metal) — both run the SAME pure-CPU harness. Bake
the integer golden `broad_lockstep`. **NO new shader, NO new RHI.**

## Design call: the pure-CPU lockstep harness over `StepHullWorldBP` (the GJ5/BP twin)

`StepHullWorldBP` (BP4) is fully deterministic; its mutable replayable state is the body vector (the `hulls` are
immutable, the grid + pair list are RE-DERIVED each tick from positions — so they are NOT state to snapshot, they
are RECOMPUTED, which is exactly why the lockstep holds through the broadphase). So BP5 is the direct GJ5 twin —
the same harness with `StepHullWorld` → `StepHullWorldBP`.
- **Commands — REUSE frozen `convex::ConvexCommand` + `gjk::ApplyHullCommands`** (GJ5): a launch impulse / a spin.
- **`SimBroadTick(world, cfg, commands, tick)`** = `gjk::ApplyHullCommands(world, commands, tick)` then
  `StepHullWorldBP(world, cfg)`. ONE deterministic broadphase-driven tick (commands BEFORE the step).
- **Snapshot/restore/equality — REUSE frozen `gjk::HullSnapshot`/`SnapshotHull`/`RestoreHull`/`HullBodiesEqual`**
  (GJ5; bodies only, hulls immutable). NO new snapshot type needed (the grid/pairs are recomputed, not stored).
- **`RunBroadLockstep(world0, cfg, commands, ticks, outIdentical)`** → two peers (authority + replica) BOTH start
  from `world0`, BOTH run `SimBroadTick` for `ticks` with the SAME command stream (each re-broadphases each tick
  independently → the same pair list → byte-identical) → `*outIdentical = HullBodiesEqual(...)`; return the
  converged authority world. (Mirror `gjk::RunHullLockstep` with the step swapped.)
- **`RunBroadRollback(world0, cfg, authStream, mispredictStream, ticks, rollbackAt, outCorrectedEqAuthority,
  outMispredictDiverged)`** → the GJ5/CX5 rollback control flow over `SimBroadTick`: advance to `rollbackAt`,
  `SnapshotHull`, speculatively mispredict (≤3 ticks, diverges), `RestoreHull`, re-sim the correct stream; set
  the corrected==authority + mispredict-diverged flags. cfg + streams CONSTANT.

**The golden scene:** the BP4 mixed-hull settle scene + a deterministic command stream (launch-impulses knock the
pile around → it re-settles). Render the converged authority world (the BP4 side-view). Both backends produce the
identical image BY CONSTRUCTION. PURE CPU (no GPU dispatch → no TDR concern).

## Reuse map (file:line)
- **BP1-BP4 `engine/sim/broad.h` (APPEND after BP4's `StepHullWorldBPN`):** `StepHullWorldBP`, `BroadStepConfig`,
  `gjk::HullWorld`. BP1-BP4 frozen.
- **gjk.h GJ5 machinery (read-only — REUSE verbatim):** `gjk::ApplyHullCommands`, `gjk::SimHullTick` (the shape
  to mirror), `gjk::HullSnapshot`/`SnapshotHull`/`RestoreHull`/`HullBodiesEqual`, `gjk::RunHullLockstep`/
  `RunHullRollback` (the control flow to mirror with `StepHullWorld` → `StepHullWorldBP`). `convex::ConvexCommand`.
- **The showcase precedent:** `gjk::`'s `--gjk-lockstep-shot`/`--gjk-lockstep` (the pure-CPU harness + the 4 proofs
  + the converged-world render, NO GPU dispatch). Mirror for `--broad-lockstep`.
- **Registration:** `scripts/verify.ps1` (append `broad_lockstep` + `--broad-lockstep-shot` to `$vkShots`),
  `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller rebakes the JSON golden**), append to
  `tests/broad_test.cpp`. (No shader → nothing for `hf_gen_msl`.)

## Design decisions (locked)
1. **APPEND to `engine/sim/broad.h`** (BP1-BP4 byte-frozen): `SimBroadTick`, `RunBroadLockstep`, `RunBroadRollback`.
   Pure integer, FIXED command + peer order. **NO new shader, NO new RHI** (the seam incl. shaders — EMPTY).
2. **Showcase `--broad-lockstep-shot <out>` (Vulkan) AND `--broad-lockstep` (Metal) — WIRE BOTH (grep your own
   visual_test.mm for `--broad-lockstep` before reporting DONE).** BOTH run the IDENTICAL pure-CPU
   `RunBroadLockstep` + `RunBroadRollback` over the BP4 mixed-hull scene + a fixed command stream — NO GPU
   dispatch. Render the converged authority world. Golden = `tests/golden/metal/broad_lockstep.png` (Mac-baked by
   the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - (1) `broad-lockstep: {bodies:<N>, ticks:<K>, commands:<C>} authority==replica BIT-IDENTICAL`
   - (2) `broad-lockstep determinism: two runs BYTE-IDENTICAL`
   - (3) `broad-lockstep rollback: corrected==authority BIT-EXACT`
   - (4) `broad-lockstep mispredict: diverged before rollback (real divergence corrected)`
   - Golden discipline: ONLY `tests/golden/metal/broad_lockstep.png`; do NOT commit it. Existing 213 goldens
     UNTOUCHED.
4. **Cross-backend bar (PURE CPU → strict):** both backends run the identical harness → bit-identical BY
   CONSTRUCTION; cross-vendor ZERO differing pixels.
5. **Tests — APPEND to `tests/broad_test.cpp` (pure CPU):** `RunBroadLockstep` authority==replica;
   `RunBroadRollback` corrected==authority AND mispredict diverged; two runs byte-identical; a command stream
   moved the pile non-trivially (re-broadphase exercised — not a frozen no-op). Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-broadphase-lockstep` (features) + `--broad-lockstep-shot`
   (showcases) + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the controller does.**

## RHI seam additions (summary)
- **None — and NO new shader.** Pure CPU. `engine/sim/broad.h` APPEND-only (BP1-BP4 frozen); gjk.h/convex.h/etc +
  ALL other sim headers + ALL existing shaders UNCHANGED. Report the seam empty (only the broad.h APPEND + the
  showcase/test/introspect are new/changed; NO shaders/ change at all).

## Out of scope (YAGNI — BP6)
The lit 3D render capstone (BP6). Real network transport. BP5 claims ONLY: the broadphase-driven hull sim is
lockstep-deterministic (two peers converge from inputs alone, re-broadphasing each tick) and rollback-replayable,
bit-identical CPU↔Vulkan↔Metal, with the integer golden + the four proofs. CAVEATS inherited: the BP4/GJ within-band.

## Verification gate
1. `ctest --preset windows-msvc-debug -R "broad|introspect"` green. Clean under `windows-msvc-asan` (separate
   build + test).
2. **proofs + visual:** `--broad-lockstep-shot` on Vulkan: the 4 proofs + exit 0. VERIFY a coherent converged
   pile (no GPU compute → no TDR/VUID risk from this slice; confirm no validation regression).
3. Metal: `visual_test --broad-lockstep` → `tests/golden/metal/broad_lockstep.png`; two runs DIFF 0.0000. Confirm
   `visual_test.mm` in the diff; NO shader added (pure CPU). Cross-vendor STRICT ZERO.
4. **Render-invariance:** ONLY `broad_lockstep.png` added; the other 213 byte-identical (+ controller introspect
   rebake).
5. Introspect: exactly `+deterministic-broadphase-lockstep` + `--broad-lockstep-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` + BP1-BP4 broad.h code + gjk.h/convex.h/fric.h/persist.h/fpx.h/grain.h + ALL other sim
   headers + ALL existing shaders byte-unchanged; broad.h APPEND-only; NO shaders/ change). `broad_lockstep` in
   the Mac loop + `--broad-lockstep-shot` in `$vkShots`.
