# Slice CD5 — Deterministic Integer CCD: LOCKSTEP + ROLLBACK (the netcode headline) — Design

> Autonomous-session spec. Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIFTH slice of
> FLAGSHIP #24 (DETERMINISTIC INTEGER CCD, `hf::sim::ccd`). CD1-CD4 built the TOI, the swept broadphase, the
> substepped CCD step, and the bullet-wall beat. CD5 proves the whole CCD sim is **lockstep- and
> rollback-replayable**: two peers fed only an input-command stream re-derive the entire world — INCLUDING the
> per-substep time-of-impact — byte-identical, and a rollback re-sims from a snapshot bit-for-bit. THE HEADLINE:
> the TOI is re-derived each substep from the current positions/velocities, so two peers re-compute the same
> impact times → **the lockstep holds through the swept continuous solve**, not just the discrete step. This is
> the deterministic-CCD moat: mainstream float engines' CCD is non-deterministic root-finding and is DISABLED in
> lockstep netcode; HF's is bit-identical and replayable. Because `StepHullWorldCCD` is a pure deterministic
> integer function, this falls out by retargeting the frozen BP5/GJ5 lockstep harness over it — PURE CPU, no
> shader, no RHI → both backends run the IDENTICAL harness → the golden is bit-identical BY CONSTRUCTION
> (cross-vendor 0 px). APPEND to `engine/sim/ccd.h` (CD1-CD4 + broad.h/gjk.h/convex.h/fpx.h/fric.h/persist.h/
> grain.h BYTE-FROZEN). Branch: `slice-cd5`. See [[hazard-forge-ccd-roadmap]], [[hazard-forge-docs-style]].

**Goal:** Extend `engine/sim/ccd.h` (additive — CD1-CD4 byte-unchanged) with `SimCcdTick` (`gjk::ApplyHullCommands`
+ `StepHullWorldCCD`) + `RunCcdLockstep` + `RunCcdRollback`, reusing the frozen command/snapshot/equality
machinery (`convex::ConvexCommand`, `gjk::HullSnapshot`/`SnapshotHull`/`RestoreHull`/`HullBodiesEqual`). Add the
showcase `--ccd-lockstep-shot` (Vulkan) / `--ccd-lockstep` (Metal) — both run the SAME pure-CPU harness. Bake the
integer golden `ccd_lockstep`. **NO new shader, NO new RHI.**

## Design call: the pure-CPU lockstep harness over `StepHullWorldCCD` (the BP5/GJ5 twin)

`StepHullWorldCCD` (CD3) is a fully deterministic integer tick; its mutable replayable state is the body vector
(the `hulls` are immutable; the swept grid + pair list + the per-substep TOIs are RE-DERIVED each tick from
positions/velocities — NOT state to snapshot, they are RECOMPUTED, which is exactly why the lockstep holds through
the swept solve). So CD5 is the direct BP5/GJ5 twin — the same harness with the step swapped to
`StepHullWorldCCD`.
- **Commands — REUSE frozen `convex::ConvexCommand` + `gjk::ApplyHullCommands`.** `SimCcdTick(world, cfg,
  commands, tick)` = `gjk::ApplyHullCommands(world, commands, tick)` then `StepHullWorldCCD(world, cfg)`. (A
  command can FIRE the projectile — an add-impulse — making the bullet-wall the lockstep scene.)
- **Snapshot/restore/equality — REUSE frozen `gjk::HullSnapshot`/`SnapshotHull`/`RestoreHull`/`HullBodiesEqual`**
  (bodies only). NO new snapshot type (the swept grid/TOIs are recomputed, not stored).
- **`RunCcdLockstep(world0, cfg, commands, ticks, outIdentical)`** → two peers (authority + replica) BOTH from
  `world0`, BOTH run `SimCcdTick` for `ticks` with the SAME command stream (each re-derives the swept broadphase
  + the per-substep TOIs independently → the same result) → `*outIdentical = HullBodiesEqual(...)`; return the
  converged authority world. (Mirror `broad::RunBroadLockstep` with the step swapped.)
- **`RunCcdRollback(world0, cfg, authStream, mispredictStream, ticks, rollbackAt, outCorrectedEqAuthority,
  outMispredictDiverged)`** → the BP5/CX5 rollback control flow over `SimCcdTick`: advance to `rollbackAt`,
  `SnapshotHull`, speculatively mispredict (≤3 ticks, diverges), `RestoreHull`, re-sim the correct stream; set
  the corrected==authority + mispredict-diverged flags. cfg + streams CONSTANT.

**The golden scene:** the CD4 bullet-wall scene (or a CCD scene with movers) + a deterministic command stream (a
launch-impulse fires the projectile; the wall arrests it; a later command perturbs). Render the converged
authority world (the CD3/CD4 side-view — the projectile arrested at the wall). Both backends produce the identical
image BY CONSTRUCTION. PURE CPU (no GPU dispatch → no TDR concern).

## Reuse map (file:line)
- **CD1-CD4 `engine/sim/ccd.h` (APPEND after CD4's `MeasureBullet`):** `StepHullWorldCCD`/`StepHullWorldCCDN`,
  `CcdStepConfig`, `MakeBulletWallScene`, `gjk::HullWorld`. CD1-CD4 frozen.
- **broad.h / gjk.h BP5 machinery (read-only — REUSE verbatim):** `broad::SimBroadTick`/`RunBroadLockstep`/
  `RunBroadRollback` (broad.h:835-866 — the SHAPE to mirror with `StepHullWorldBP` → `StepHullWorldCCD`),
  `gjk::ApplyHullCommands`, `gjk::HullSnapshot`/`SnapshotHull`/`RestoreHull`/`HullBodiesEqual`,
  `convex::ConvexCommand`. Do NOT modify broad.h/gjk.h/etc — BYTE-FROZEN.
- **The showcase precedent:** `broad::`'s `--broad-lockstep-shot`/`--broad-lockstep` (the pure-CPU harness + the 4
  proofs + the converged-world render, NO GPU dispatch). Mirror for `--ccd-lockstep`.
- **Registration:** `scripts/verify.ps1` (append `ccd_lockstep` + `--ccd-lockstep-shot` to `$vkShots`),
  `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller rebakes the JSON golden**), append to
  `tests/ccd_test.cpp`. (No shader → nothing for `hf_gen_msl`.)

## Design decisions (locked)
1. **APPEND to `engine/sim/ccd.h`** (CD1-CD4 byte-frozen): `SimCcdTick`, `RunCcdLockstep`, `RunCcdRollback`. Pure
   integer, FIXED command + peer order. **NO new shader, NO new RHI** (the seam incl. shaders — EMPTY).
2. **Showcase `--ccd-lockstep-shot <out>` (Vulkan) AND `--ccd-lockstep` (Metal) — WIRE BOTH (grep your own
   visual_test.mm for `--ccd-lockstep` before reporting DONE).** BOTH run the IDENTICAL pure-CPU `RunCcdLockstep`
   + `RunCcdRollback` over the bullet-wall (or CCD) scene + a fixed command stream — NO GPU dispatch. Render the
   converged authority world. Golden = `tests/golden/metal/ccd_lockstep.png` (Mac-baked by the CONTROLLER — DO NOT
   commit).
3. **PROOFS (fail loudly; exact lines):**
   - (1) `ccd-lockstep: {bodies:<N>, ticks:<K>, commands:<C>} authority==replica BIT-IDENTICAL`
   - (2) `ccd-lockstep determinism: two runs BYTE-IDENTICAL`
   - (3) `ccd-lockstep rollback: corrected==authority BIT-EXACT`
   - (4) `ccd-lockstep mispredict: diverged before rollback (real divergence corrected)`
   - Golden discipline: ONLY `tests/golden/metal/ccd_lockstep.png`; do NOT commit it. Existing 219 goldens
     UNTOUCHED.
4. **Cross-backend bar (PURE CPU → strict):** both backends run the identical harness → bit-identical BY
   CONSTRUCTION; cross-vendor ZERO differing pixels.
5. **Tests — APPEND to `tests/ccd_test.cpp` (pure CPU):** `RunCcdLockstep` authority==replica; `RunCcdRollback`
   corrected==authority AND mispredict diverged; two runs byte-identical; a command stream moved the scene
   non-trivially (the CCD step + re-broadphase exercised — not a frozen no-op). Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-ccd-lockstep` (features) + `--ccd-lockstep-shot` (showcases) +
   update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the controller does.**

## RHI seam additions (summary)
- **None — and NO new shader.** Pure CPU. `engine/sim/ccd.h` APPEND-only (CD1-CD4 frozen); broad.h/gjk.h/etc +
  ALL other sim headers + ALL existing shaders UNCHANGED. Report the seam empty (only the ccd.h APPEND + the
  showcase/test/introspect are new/changed; NO shaders/ change at all).

## Out of scope (YAGNI — CD6)
The lit 3D render capstone (CD6). Real network transport. CD5 claims ONLY: the CCD sim is lockstep-deterministic
(two peers converge from inputs alone, re-deriving the TOI each substep) and rollback-replayable, bit-identical
CPU↔Vulkan↔Metal, with the integer golden + the four proofs. CAVEATS inherited: the CD1-CD4 within-band.

## Verification gate
1. `ctest --preset windows-msvc-debug -R "ccd|introspect"` green. Clean under `windows-msvc-asan` (separate build
   + test).
2. **proofs + visual:** `--ccd-lockstep-shot` on Vulkan: the 4 proofs + exit 0. VERIFY a coherent converged world
   (the projectile arrested at the wall). (PURE CPU — no GPU compute, no TDR/VUID risk from this slice.)
3. Metal: `visual_test --ccd-lockstep` → `tests/golden/metal/ccd_lockstep.png`; two runs DIFF 0.0000. Confirm
   `visual_test.mm` in the diff; NO shader added (pure CPU). Cross-vendor STRICT ZERO.
4. **Render-invariance:** ONLY `ccd_lockstep.png` added; the other 219 byte-identical (+ controller introspect
   rebake).
5. Introspect: exactly `+deterministic-ccd-lockstep` + `--ccd-lockstep-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` + CD1-CD4 ccd.h code + broad.h/gjk.h/convex.h/fpx.h/etc + ALL other sim headers + ALL
   existing shaders byte-unchanged; ccd.h APPEND-only; NO shaders/ change). `ccd_lockstep` in the Mac loop +
   `--ccd-lockstep-shot` in `$vkShots`.
